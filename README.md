# Visionflow

Visionflow is an open-source Python package designed to automate and streamline the entire image classification workflow using TensorFlow and Keras. From data loading and custom preprocessing to training diverse model architectures and generating detailed evaluation reports, Visionflow makes experimenting with deep learning for computer vision effortless.

## Installation

**Note:** Visionflow requires TensorFlow. It's recommended to install it first.

1.  **Install TensorFlow:**
    ```bash
    pip install tensorflow
    ```

2.  **Install the Visionflow package from GitHub:**
    ```bash
    pip install https://github.com/MithraBijumon/visionflow/
    ```

## Usage

### Preparing Your Data

Before training, organize your images into a specific folder structure. Create a main data directory, and inside it, create one sub-directory for each of your classes.

---

### Basic Training (Transfer Learning)

This is the recommended approach for most problems. It uses a powerful, pre-trained model for fast and effective training. Create a `train.py` script and run it.

```python
# Import the main classes from their specific modules
from visionflow.pipeline import ClassificationPipeline
from visionflow.preprocessing import Preprocessor
from visionflow.postprocessing import Postprocessor

# 1. Configure the preprocessor
preprocessor_config = {
    "resize": {"height": 128, "width": 128},
    "normalize": True
}
custom_preprocessor = Preprocessor(config=preprocessor_config)

# 2. Configure the postprocessor
postprocessor_config = {
    "accuracy": True,
    "classification_report": True,
    "confusion_matrix": True
}
custom_postprocessor = Postprocessor(config=postprocessor_config)

# 3. Configure the main experiment
experiment_config = {
    "data_path": "my_dataset/",
    "model_name": "MobileNetV2",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "checkpoint_path": "checkpoints/best_model.keras",
    "preprocessor": custom_preprocessor,
    "postprocessor": custom_postprocessor
}

# 4. Run the pipeline
pipeline = ClassificationPipeline(config=experiment_config)
pipeline.run()
```
### Advanced Training (Custom Augmentation & MixUp)
For more challenging datasets, you can add custom preprocessing steps and enable advanced training techniques like MixUp.


```
from visionflow.pipeline import ClassificationPipeline
from visionflow.preprocessing import Preprocessor
from visionflow.postprocessing import Postprocessor

# Add custom augmentations like blur and flips
preprocessor_config = {
    "resize": {"height": 128, "width": 128},
    "gaussian_blur": {"ksize": 3, "sigma": 1.0},
    "flip_horizontal": True,
    "normalize": True
}
custom_preprocessor = Preprocessor(config=preprocessor_config)

# Generate a more detailed report
postprocessor_config = {
    "accuracy": True,
    "classification_report": True,
    "confusion_matrix": True,
    "save_csv": "results/predictions.csv"
}
custom_postprocessor = Postprocessor(config=postprocessor_config)

experiment_config = {
    "data_path": "my_dataset/",
    "model_name": "EfficientNetB0",
    "epochs": 20, # Train for longer with augmentation
    "batch_size": 32,
    "learning_rate": 0.001,
    "checkpoint_path": "checkpoints/best_augmented_model.keras",
    "preprocessor": custom_preprocessor,
    "postprocessor": custom_postprocessor,
    "use_mixup": True # Enable MixUp
}

pipeline = ClassificationPipeline(config=experiment_config)
pipeline.run()
