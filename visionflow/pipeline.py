import tensorflow as tf
import numpy as np
import os

# --- 1. Import your project's modules ---
# The preprocessor is now the key import for data handling.
from .preprocessing import Preprocessor, create_dataset_from_directory
from .model_zoo import get_model
from .trainer import Trainer
from .postprocessing import decode_predictions, plot_confusion_matrix, classification_report

# ==============================================================================
# The Main Pipeline Class (UPDATED)
# ==============================================================================

class ClassificationPipeline:
    """
    Automates the end-to-end image classification workflow using a
    configurable preprocessor.
    """
    def __init__(self, config: dict):
        """
        Initializes the pipeline with a configuration dictionary.
        The config MUST now contain a 'preprocessor' object.
        """
        if "preprocessor" not in config:
            raise ValueError("Configuration must include a 'preprocessor' object.")
            
        self.config = config
        self.preprocessor = config["preprocessor"]
        
        # Get class names from the data path
        self.class_names = sorted([d for d in os.listdir(config["data_path"]) if os.path.isdir(os.path.join(config["data_path"], d))])
        self.config["num_classes"] = len(self.class_names)
        
        print("✅ Pipeline initialized. Found classes:", self.class_names)

    def _prepare_datasets(self):
        """Loads and splits data into training and validation sets using the preprocessor."""
        print("\n--- STAGE 1: Preparing Datasets ---")
        
        # The preprocessor object is now passed directly to the dataset creation function
        full_dataset = create_dataset_from_directory(
            data_path=self.config["data_path"],
            batch_size=self.config["batch_size"],
            preprocessor=self.preprocessor
        )
        
        # Split the data into training (80%) and validation (20%)
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int(0.8 * dataset_size)
        
        train_dataset = full_dataset.take(train_size)
        validation_dataset = full_dataset.skip(train_size)
        
        print(f"   - Training dataset size: {train_size} batches")
        print(f"   - Validation dataset size: {dataset_size - train_size} batches")
        return train_dataset, validation_dataset

    def run(self):
        """
        Executes the entire pipeline: data prep, training, and evaluation.
        """
        print("\n🚀 Starting Visionflow Pipeline...")
        
        # --- STAGE 1: PREPROCESSING ---
        train_ds, val_ds = self._prepare_datasets()
        
        # --- STAGE 2: MODEL TRAINING ---
        print("\n--- STAGE 2: Model Training ---")
        model = get_model(
            model_name=self.config["model_name"],
            input_shape=(*self.preprocessor.config["resize"].values(), 3), # Get img size from preprocessor
            num_classes=self.config["num_classes"]
        )
        
        trainer = Trainer(model=model, config=self.config)
        history = trainer.train(train_ds, val_ds)
        
        # --- STAGE 3: POSTPROCESSING & EVALUATION ---
        print("\n--- STAGE 3: Evaluating on Validation Data ---")
        all_true_labels = []
        all_raw_predictions = []
        for images, labels in val_ds:
            all_true_labels.extend(labels.numpy())
            preds = model.predict(images, verbose=0)
            all_raw_predictions.extend(preds)
            
        true_class_names = [self.class_names[i] for i in all_true_labels]
        pred_class_names, _ = decode_predictions(np.array(all_raw_predictions), self.class_names)

        print("\n--- Classification Report ---")
        print(classification_report(true_class_names, pred_class_names, target_names=self.class_names))
        
        print("\n--- Displaying Confusion Matrix ---")
        plot_confusion_matrix(true_class_names, pred_class_names, self.class_names)
        
        print("\n🎉 Pipeline execution finished successfully!")
        return history

# ==============================================================================
# EXAMPLE USAGE (how a user would run your package with the new system)
# ==============================================================================
if __name__ == '__main__':
    # --- Create dummy data for a self-contained demonstration ---
    DUMMY_DATA_PATH = "temp_data_pipeline"
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "dogs"), exist_ok=True)
    for i in range(50):
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/cats/cat{i}.jpg", np.random.rand(100, 100, 3) * 255)
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/dogs/dog{i}.jpg", np.random.rand(100, 100, 3) * 255)

    # 1. Define a custom preprocessing configuration using YOUR function names
    my_preprocessor_config = {
        "resize": {"height": 128, "width": 128},
        "median_filter": {"ksize": 3},
        "flip_horizontal": True, # Will be applied randomly during dataset creation
        "normalize": True
    }

    # 2. Create an instance of the Preprocessor
    custom_preprocessor = Preprocessor(config=my_preprocessor_config)

    # 3. The main experiment config now includes the preprocessor object
    experiment_config = {
        "data_path": DUMMY_DATA_PATH,
        "model_name": "MobileNetV2",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.001,
        "checkpoint_path": "checkpoints/best_model.keras",
        "preprocessor": custom_preprocessor # <-- Pass the configured object
    }

    # 4. Create and run the pipeline
    pipeline = ClassificationPipeline(config=experiment_config)
    pipeline.run()
