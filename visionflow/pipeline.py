import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report

# --- 1. Import your actual project modules ---
from .preprocessing import create_dataset_from_directory
from .postprocessing import decode_predictions, plot_confusion_matrix
from .model_zoo import get_model
from .trainer import Trainer

# ==============================================================================
# THE MAIN PIPELINE CLASS
# ==============================================================================

class ClassificationPipeline:
    """
    Automates the end-to-end image classification workflow, from data loading
    and preprocessing to model training and post-training evaluation.
    """
    def __init__(self, config: dict):
        """
        Initializes the pipeline with a user-defined configuration.

        Args:
            config (dict): A dictionary containing all necessary parameters, such as
                           data_path, model_name, img_size, epochs, etc.
        """
        self.config = config
        
        # Automatically determine class names from the data directory
        self.class_names = sorted([d for d in os.listdir(config["data_path"]) if os.path.isdir(os.path.join(config["data_path"], d))])
        self.config["num_classes"] = len(self.class_names)
        
        print("✅ Pipeline initialized. Found classes:", self.class_names)

    def _prepare_datasets(self):
        """
        Loads the full dataset and splits it into training and validation sets.
        """
        print("\n--- STAGE 1: Preparing Datasets ---")
        full_dataset = create_dataset_from_directory(
            data_path=self.config["data_path"],
            img_size=self.config["img_size"],
            batch_size=self.config["batch_size"],
            augment=True  # Augmentation is enabled by default
        )
        
        # Split the data into training (80%) and validation (20%)
        # This is a standard practice to evaluate the model on unseen data.
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int(0.8 * dataset_size)
        
        train_dataset = full_dataset.take(train_size)
        validation_dataset = full_dataset.skip(train_size)
        
        print(f"   - Training dataset size: {train_size} batches")
        print(f"   - Validation dataset size: {dataset_size - train_size} batches")
        return train_dataset, validation_dataset

    def run(self):
        """
        Executes the entire pipeline: data prep, model training, and evaluation.
        """
        print("\n🚀 Starting Visionflow Pipeline...")
        
        # --- STAGE 1: PREPROCESSING ---
        train_ds, val_ds = self._prepare_datasets()
        
        # --- STAGE 2: MODEL CREATION & TRAINING ---
        print("\n--- STAGE 2: Model Training ---")
        model = get_model(
            model_name=self.config["model_name"],
            input_shape=(*self.config["img_size"], 3),
            num_classes=self.config["num_classes"]
        )
        
        trainer = Trainer(model=model, config=self.config)
        history = trainer.train(train_ds, val_ds)
        
        # --- STAGE 3: POSTPROCESSING & EVALUATION ---
        print("\n--- STAGE 3: Evaluating on Validation Data ---")
        # Get all true labels and make predictions on the validation set
        all_true_labels = []
        all_raw_predictions = []
        for images, labels in val_ds:
            all_true_labels.extend(labels.numpy())
            preds = model.predict(images, verbose=0)
            all_raw_predictions.extend(preds)
            
        # Use your postprocessing functions to get human-readable results
        true_class_names = [self.class_names[i] for i in all_true_labels]
        pred_class_names, _ = decode_predictions(np.array(all_raw_predictions), self.class_names)

        # Display the results
        print("\n--- Classification Report ---")
        print(classification_report(true_class_names, pred_class_names))
        
        print("\n--- Displaying Confusion Matrix ---")
        plot_confusion_matrix(true_class_names, pred_class_names, self.class_names)
        
        print("\n🎉 Pipeline execution finished successfully!")
        return history

# ==============================================================================
# EXAMPLE USAGE (how a user would run your package)
# ==============================================================================
if __name__ == '__main__':
    # --- Create dummy data for a self-contained demonstration ---
    DUMMY_DATA_PATH = "temp_data_pipeline"
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "dogs"), exist_ok=True)
    for i in range(50): # Create 50 dummy images per class for a decent split
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/cats/cat{i}.jpg", np.random.rand(100, 100, 3) * 255)
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/dogs/dog{i}.jpg", np.random.rand(100, 100, 3) * 255)

    # 1. This is what a user would configure for their experiment
    experiment_config = {
        "data_path": DUMMY_DATA_PATH,
        "model_name": "MobileNetV2",
        "img_size": (128, 128),
        "epochs": 3, 
        "batch_size": 8,
        "learning_rate": 0.001,
        "checkpoint_path": "checkpoints/best_model.keras"
    }

    # 2. The user creates and runs the pipeline
    pipeline = ClassificationPipeline(config=experiment_config)
    pipeline.run()
