import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Run inference on a dataset and return predictions and ground truth labels.
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

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report(y_true, y_pred, class_names=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    cm = confusion_matrix_manual(y_true, y_pred, n_classes)
    
    report = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        cls_name = class_names[i] if class_names else str(cls)
        report[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": cm[i, :].sum()
        }
    
    # Add overall accuracy
    report["accuracy"] = np.trace(cm) / np.sum(cm)
    return report


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


class Postprocessor:
    """
    Postprocessing with custom config
    """
    def __init__(self, config: dict):
        self.config = config

    def run_report(self, model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list):
        
        print("\n--- STAGE 3: Evaluating on Validation Data ---")

        # This is a common step for most tasks, so we do it once.
        pred_indices, true_indices = get_predictions(model, dataset)

        if self.config.get("accuracy"):
            accuracy = compute_accuracy(true_indices, pred_indices)
            print(f"\nOverall Validation Accuracy: {accuracy:.4f}")

        if self.config.get("classification_report"):
            print("\n--- Classification Report ---")
            report = classification_report(true_indices, pred_indices, class_names=class_names)
            for cls, metrics in report.items():
            print(f"{cls}: {metrics}")
    
        if self.config.get("confusion_matrix"):
            print("\n--- Displaying Confusion Matrix ---")
            cm = confusion_matrix(true_indices, pred_indices, len(class_names))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
            
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()
        if "plot_samples" in self.config:
            print("\n--- Visualizing Sample Predictions ---")
            sample_images, sample_labels = next(iter(dataset))
            sample_preds = predict_batch(model, sample_images)
            n_samples = self.config["plot_samples"].get("n", 6)
            plot_sample_predictions(sample_images, sample_preds, sample_labels.numpy(), class_names, n=n_samples)

        if "save_csv" in self.config:
            print("\n--- Saving Predictions to CSV ---")
            save_predictions_to_csv(pred_indices, filename=self.config["save_csv"], class_names=class_names)

        if "save_images" in self.config:
            print("\n--- Saving Annotated Images ---")
            sample_images, _ = next(iter(dataset))
            sample_preds = predict_batch(model, sample_images)
            save_annotated_images(sample_images, sample_preds, out_dir=self.config["save_images"], class_names=class_names)


