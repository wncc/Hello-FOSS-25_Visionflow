import tensorflow as tf
import numpy as np
import os

# --- 1. Import your project's modules ---
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
# The Main Pipeline Class (No changes needed here)
# ==============================================================================

class ClassificationPipeline:
    """
    Automates the end-to-end image classification workflow using a
    configurable preprocessor and generating a comprehensive evaluation report.
    """
    def __init__(self, config: dict):
        if "preprocessor" not in config:
            raise ValueError("Configuration must include a 'preprocessor' object.")
            
        self.config = config
        self.preprocessor = config["preprocessor"]
        
        self.class_names = sorted([d for d in os.listdir(config["data_path"]) if os.path.isdir(os.path.join(config["data_path"], d))])
        self.config["num_classes"] = len(self.class_names)
        
        print("✅ Pipeline initialized. Found classes:", self.class_names)

    def _prepare_datasets(self):
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
        print("\n🚀 Starting Visionflow Pipeline...")
        
        train_ds, val_ds = self._prepare_datasets()
        
        print("\n--- STAGE 2: Model Training ---")
        model = get_model(
            model_name=self.config["model_name"],
            input_shape=(*self.preprocessor.config["resize"].values(), 3),
            num_classes=self.config["num_classes"]
        )
        
        trainer = Trainer(model=model, config=self.config)
        history = trainer.train(train_ds, val_ds)
        
        print("\n--- STAGE 3: Evaluating on Validation Data ---")
        pred_indices, true_indices = get_predictions(model, val_ds)
        
        accuracy = compute_accuracy(true_indices, pred_indices)
        print(f"\nOverall Validation Accuracy: {accuracy:.4f}")

        print("\n--- Classification Report ---")
        print(classification_report(true_indices, pred_indices, class_names=self.class_names))
        
        print("\n--- Displaying Confusion Matrix ---")
        true_class_names = [self.class_names[i] for i in true_indices]
        pred_class_names = [self.class_names[i] for i in pred_indices]
        plot_confusion_matrix(true_class_names, pred_class_names, self.class_names)
        
        print("\n--- Visualizing Sample Predictions ---")
        sample_images, sample_labels = next(iter(val_ds))
        sample_preds = predict_batch(model, sample_images)
        plot_sample_predictions(sample_images, sample_preds, sample_labels.numpy(), self.class_names)

        print("\n--- Saving Results to Disk ---")
        save_predictions_to_csv(pred_indices, class_names=self.class_names)
        save_annotated_images(sample_images, sample_preds, out_dir="./results", class_names=self.class_names)
        
        print("\n🎉 Pipeline execution finished successfully!")
        return history

# ==============================================================================
# EXAMPLE USAGE (UPDATED to show new augmentation features)
# ==============================================================================
if __name__ == '__main__':
    # --- Create dummy data for a self-contained demonstration ---
    DUMMY_DATA_PATH = "temp_data_pipeline"
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "dogs"), exist_ok=True)
    for i in range(50):
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/cats/cat{i}.jpg", np.random.rand(100, 100, 3) * 255)
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/dogs/dog{i}.jpg", np.random.rand(100, 100, 3) * 255)

    # 1. Define a custom preprocessing configuration that now includes augmentations
    my_preprocessor_config = {
        "resize": {"height": 128, "width": 128},
        "normalize": True,
        # --- Demonstrate how to enable augmentations from the new module ---
        "flip_horizontal": True, # This will be applied randomly
        "adjust_brightness": {"value": 40}, # Randomly adjust brightness by +/- 40
        "gaussian_blur": {"ksize": 3, "sigma": 1.0}
    }

    # 2. Create an instance of the Preprocessor
    custom_preprocessor = Preprocessor(config=my_preprocessor_config)

    # 3. The main experiment config includes the configured preprocessor object
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

