import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report as sk_classification_report
import os


def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Run inference on a dataset and return predictions and ground truth labels.

    Args:
        model (tf.keras.Model): The trained Keras model.
        dataset (tf.data.Dataset): A dataset yielding (images, labels) batches.

    Returns:
        tuple: A tuple containing (predicted_indices, true_labels).
    """
    model.evaluate # Ensure model is in inference mode
    all_preds = []
    all_labels = []

    for images, labels in dataset:
        raw_preds = model.predict(images, verbose=0)
        predicted_indices = np.argmax(raw_preds, axis=1)
        all_preds.extend(predicted_indices)
        all_labels.extend(labels.numpy())

    return all_preds, all_labels

def predict_single(model: tf.keras.Model, image: np.ndarray, class_names: list = None):
    """
    Predict a class for a single image.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image (np.ndarray): A single image array (H, W, C).
        class_names (list, optional): A list to map the output index to a class name.

    Returns:
        str or int: The predicted class name or index.
    """
    # Add a batch dimension and get prediction
    image_batch = np.expand_dims(image, axis=0)
    raw_pred = model.predict(image_batch, verbose=0)[0]
    pred_idx = np.argmax(raw_pred)

    if class_names:
        return class_names[pred_idx]
    return pred_idx

def predict_batch(model: tf.keras.Model, images: np.ndarray, class_names: list = None):
    """
    Predict classes for a batch of images.

    Args:
        model (tf.keras.Model): The trained Keras model.
        images (np.ndarray): A batch of images (B, H, W, C).
        class_names (list, optional): A list to map output indices to class names.

    Returns:
        list: A list of predicted class names or indices.
    """
    raw_preds = model.predict(images, verbose=0)
    pred_indices = np.argmax(raw_preds, axis=1)

    if class_names:
        return [class_names[i] for i in pred_indices]
    return list(pred_indices)


def compute_accuracy(y_true, y_pred):
    """Computes classification accuracy from lists or arrays."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def classification_report(y_true, y_pred, class_names=None):
    """Generate a precision/recall/F1 report using scikit-learn."""
    return sk_classification_report(y_true, y_pred, target_names=class_names)

# ==============================================================================
# Visualization and Export Functions
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10,8), cmap="Blues"):
    """Plots and displays a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

def plot_sample_predictions(images, preds, labels=None, class_names=None, n=6):
    """
    Plot a batch of images with predicted (and optionally true) labels.
    """
    images = images[:n]
    preds = preds[:n]
    if labels is not None:
        labels = labels[:n]

    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i]) # Assumes images are in a displayable format (e.g., [0,1] or [0,255])
        plt.axis("off")
        
        # Get prediction name
        pred_name = class_names[preds[i]] if class_names else preds[i]
        title = f"Pred: {pred_name}"

        if labels is not None:
            true_name = class_names[labels[i]] if class_names else labels[i]
            title += f"\nTrue: {true_name}"
            if pred_name == true_name:
                plt.title(title, color="green")
            else:
                plt.title(title, color="red")
        else:
            plt.title(title)
            
    plt.show()

def save_predictions_to_csv(preds, filename="predictions.csv", class_names=None):
    """Saves predictions to a CSV file."""
    if class_names:
        preds = [class_names[p] for p in preds]

    df = pd.DataFrame({"prediction": preds})
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")


def save_annotated_images(images, preds, out_dir="./results", class_names=None):
    """Saves images to disk with their predicted labels as titles."""
    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(images):
        plt.imshow(img)
        plt.axis("off")
        pred_name = class_names[preds[i]] if class_names else preds[i]
        plt.title(f"Pred: {pred_name}")
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"))
        plt.close()

    print(f"Saved annotated images to {out_dir}")

