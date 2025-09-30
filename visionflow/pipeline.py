import tensorflow as tf
import numpy as np
import os

#Importing all our modules
from .preprocessing import Preprocessor, create_dataset_from_directory
from .models import get_model
from .trainer import Trainer
from .postprocessing import Postprocessor


class ClassificationPipeline:
    
    def __init__(self, config: dict):
    
        if "preprocessor" not in config or "postprocessor" not in config:
            raise ValueError("Config must include 'preprocessor' and 'postprocessor' objects.")
            
        self.config = config
        self.preprocessor = config["preprocessor"]
        self.postprocessor = config["postprocessor"]
        
        self.class_names = sorted([d for d in os.listdir(config["data_path"]) if os.path.isdir(os.path.join(config["data_path"], d))])
        self.config["num_classes"] = len(self.class_names)
        
        print("Pipeline initialized. Found classes:", self.class_names)

    def _prepare_datasets(self):
        print("\n--- STAGE 1: Preparing Datasets ---")

        #creating dataset from directory containing classes and preprocessing each image
        full_dataset = create_dataset_from_directory(
            data_path=self.config["data_path"],
            batch_size=self.config["batch_size"],
            preprocessor=self.preprocessor
        )

        #Training set we take as 80% of dataset and rest validation set
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        train_size = int(0.8 * dataset_size)
        train_dataset = full_dataset.take(train_size)
        validation_dataset = full_dataset.skip(train_size)
        print(f"   - Training dataset size: {train_size} batches")
        print(f"   - Validation dataset size: {dataset_size - train_size} batches")
        return train_dataset, validation_dataset

    def run(self):
        print("\n Starting Visionflow Pipeline...")
        
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
        
        # --- STAGE 3: POSTPROCESSING ---
        self.postprocessor.run_report(model, val_ds, self.class_names)
        
        print("\n Pipeline execution finished successfully!")
        return history

# ==============================================================================
# EXAMPLE USAGE (how a user would now run package)
# ==============================================================================
if __name__ == '__main__':
    # --- Create dummy data ---
    DUMMY_DATA_PATH = "temp_data_pipeline"
    # ... (dummy data creation code remains the same)
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "cats"), exist_ok=True)
    os.makedirs(os.path.join(DUMMY_DATA_PATH, "dogs"), exist_ok=True)
    for i in range(50):
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/cats/cat{i}.jpg", np.random.rand(100, 100, 3) * 255)
        tf.keras.utils.save_img(f"{DUMMY_DATA_PATH}/dogs/dog{i}.jpg", np.random.rand(100, 100, 3) * 255)

    # 1. Define the PRE-processor configuration
    preprocessor_config = {
        "resize": {"height": 128, "width": 128},
        "normalize": True
    }
    custom_preprocessor = Preprocessor(config=preprocessor_config)

    # 2. Define the POST-processor configuration
    postprocessor_config = {
        "accuracy": True,
        "classification_report": True,
        "confusion_matrix": True,
        "save_csv": "results/final_predictions.csv",
        "save_images": "results/annotated_images"
    }
    custom_postprocessor = Postprocessor(config=postprocessor_config)

    # 3. The main experiment config now includes BOTH processor objects
    experiment_config = {
        "data_path": DUMMY_DATA_PATH,
        "model_name": "MobileNetV2",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.001,
        "checkpoint_path": "checkpoints/best_model.keras",
        "preprocessor": custom_preprocessor,
        "postprocessor": custom_postprocessor # <-- Add the new object
    }

    # 4. Create and run the pipeline
    pipeline = ClassificationPipeline(config=experiment_config)
    pipeline.run()

