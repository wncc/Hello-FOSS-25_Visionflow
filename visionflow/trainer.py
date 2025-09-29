# visionflow/trainer.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime
import os

class Trainer:
    """
    Handles the model compilation and the entire training process.
    
    This class encapsulates the logic for training, including setting up
    the optimizer, loss function, callbacks, and executing the `fit` method.
    """
    def __init__(self, model: tf.keras.Model, config: dict):
        """
        Initializes the Trainer and compiles the model.

        Args:
            model (tf.keras.Model): The Keras model to be trained.
            config (dict): A dictionary containing training parameters, such as
                           'learning_rate', 'epochs', and 'checkpoint_path'.
        """
        self.model = model
        self.config = config
        self.history = None

        # 1. Define the optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.get("learning_rate", 0.001)
        )
        
        # 2. Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        print("✅ Trainer initialized and model compiled.")

    def _get_callbacks(self) -> list:
        """
        Prepares a list of Keras callbacks for a robust training session.
        """
        checkpoint_path = self.config.get("checkpoint_path", "checkpoints/best_model.keras")
        # Ensure the directory for the checkpoint exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Callback to save the best model based on validation accuracy
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        # Callback to stop training early if validation accuracy doesn't improve
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Number of epochs to wait for improvement
            restore_best_weights=True,
            verbose=1
        )

        # (Optional) Callback to log training for TensorBoard visualization
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return [model_checkpoint, early_stopping, tensorboard_callback]

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset):
        """
        Executes the model training loop using the provided data.

        Args:
            train_dataset (tf.data.Dataset): The dataset for training.
            validation_dataset (tf.data.Dataset): The dataset for validation.
        """
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...")
        callbacks = self._get_callbacks()

        self.history = self.model.fit(
            train_dataset,
            epochs=self.config['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        
        print("🎉 Training finished!")
        return self.history
