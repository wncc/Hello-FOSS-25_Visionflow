import tensorflow as tf
import numpy as np
import os

# --- 1. Import your project's modules (with new postprocessing functions) ---
from .preprocessing import Preprocessor, create_dataset_from_directory
from .model_zoo import get_model
from .trainer import Trainer
from .postprocessing import (
    get_predictions, 
    compute_accuracy,
    classification_report, 
    plot_confusion_matrix, 
    plot_sample_predictions,
    save_predictions_to_csv,
    save_annotated_images
)

# ==============================================================================
# The Main Pipeline Class (UPDATED)
# ==============================================================================

class ClassificationPipeline:
    """
    Automates the end-to-end image classification workflow using a
    configurable preprocessor and generating a comprehensive evaluation report.
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
        
        self.class_names = sorted([d for d in os.listdir(config["data_path"]) if os.path.isdir(os.path.join(config["data_path"], d))])
        self.config["num_classes"] = len(self.class_names)
        
        print("✅ Pipeline initialized. Found classes:", self.class_names)

    def _prepare_datasets(self):
        """Loads and splits data into training and validation sets using the preprocessor."""
        print("\n--- STAGE 1: Preparing Datasets ---")
        
        full_dataset = create_dataset_from_directory(
            data_path=self.config["data_path"],
            batch_size=self.config["batch_size"],
            preprocessor=self.preprocessor
        )
        
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
            input_shape=(*self.preprocessor.config["resize"].values(), 3),
            num_classes=self.config["num_classes"]
        )
        
        trainer = Trainer(model=model, config=self.config)
        history = trainer.train(train_ds, val_ds)
        
        # --- STAGE 3: POSTPROCESSING & EVALUATION (Now more comprehensive) ---
        print("\n--- STAGE 3: Evaluating on Validation Data ---")
        
        # 3.1 Get all predictions and labels using the new helper function
        pred_indices, true_indices = get_predictions(model, val_ds)
        
        # 3.2 Compute and print overall accuracy
        accuracy = compute_accuracy(true_indices, pred_indices)
        print(f"\nOverall Validation Accuracy: {accuracy:.4f}")

        # 3.3 Print detailed classification report
        print("\n--- Classification Report ---")
        print(classification_report(true_indices, pred_indices, class_names=self.class_names))
        
        # 3.4 Display Confusion Matrix
        print("\n--- Displaying Confusion Matrix ---")
        # Convert indices to class names for plotting
        true_class_names = [self.class_names[i] for i in true_indices]
        pred_class_names = [self.class_names[i] for i in pred_indices]
        plot_confusion_matrix(true_class_names, pred_class_names, self.class_names)
        
        # 3.5 Visualize some sample predictions
        print("\n--- Visualizing Sample Predictions ---")
        # Get a single batch of images and labels to visualize
        sample_images, sample_labels = next(iter(val_ds))
        sample_preds = predict_batch(model, sample_images)
        plot_sample_predictions(sample_images, sample_preds, sample_labels.numpy(), self.class_names)

        # 3.6 Save results to disk
        print("\n--- Saving Results to Disk ---")
        save_predictions_to_csv(pred_indices, class_names=self.class_names)
        save_annotated_images(sample_images, sample_preds, out_dir="./results", class_names=self.class_names)
        
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

    # 1. Define a custom preprocessing configuration
    my_preprocessor_config = {
        "resize": {"height": 128, "width": 128},
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
        "preprocessor": custom_preprocessor
    }

    # 4. Create and run the pipeline
    pipeline = ClassificationPipeline(config=experiment_config)
    pipeline.run()

