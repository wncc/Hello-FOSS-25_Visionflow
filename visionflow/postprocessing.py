import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional, Tuple, Any, Dict


def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    
    raw_preds = model.predict(dataset, verbose=0)
    preds_np = np.argmax(raw_preds, axis=1)

    label_tensors = []
    for _, labels in dataset:
        if not tf.is_tensor(labels):
            labels = tf.convert_to_tensor(labels)
        label_tensors.append(labels)
        
    if label_tensors:
        labels_concat = tf.concat(label_tensors, axis=0)
        labels_np = labels_concat.numpy()
        if labels_np.ndim > 1:
            labels_np = np.argmax(labels_np, axis=1)
    else:
        labels_np = np.array([], dtype=int)

    return preds_np, labels_np



def predict_single(model: tf.keras.Model, image: np.ndarray, class_names: Optional[List[str]] = None) -> Any:
    """
    Predict a class for a single image. Returns class index or name (if class_names provided).
    """
    image_np = np.asarray(image)
    image_batch = np.expand_dims(image_np, axis=0)
    raw_pred = model.predict(image_batch, verbose=0)[0]
    pred_idx = int(np.argmax(raw_pred))

    if class_names:
        return class_names[pred_idx]
    return pred_idx


def predict_batch(model: tf.keras.Model, images: np.ndarray, class_names: Optional[List[str]] = None) -> List[Any]:
    """
    Predict classes for a batch of images. Returns list of indices or names (if class_names provided).
    """
    images_np = np.asarray(images)
    raw_preds = model.predict(images_np, verbose=0)
    pred_indices = np.argmax(raw_preds, axis=1).tolist()

    if class_names:
        return [class_names[i] for i in pred_indices]
    return pred_indices


def compute_accuracy(y_true, y_pred) -> float:
    """Computes classification accuracy from lists or arrays. Returns 0.0 if empty."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0:
        return 0.0
    correct = np.sum(y_true == y_pred)
    return float(correct) / len(y_true)


def confusion_matrix(y_true, y_pred, labels : Optional[List] = None):
    """ Compute a confusion matrix from true and predicted labels."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    num_classes = len(labels)
    # Label to index mapping
    label_to_index = {label : i for i  , label in enumerate(labels)}

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        true_index = label_to_index.get(t)
        pred_index = label_to_index.get(p)

        if true_index is not None and pred_index is not None:
            cm[true_index, pred_index] += 1

    return cm , labels


def classification_report(y_true, y_pred, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate precision, recall, F1-score, and support for each class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Decide label ordering: use provided class_names if they match labels, else infer
    if class_names is not None:
        labels = np.arange(len(class_names))
        # if y_true/y_pred are strings or not 0..n-1, try to use class_names as label names
        # We support both numeric label indices and string label names.
        # If y_true contains strings, use class_names directly as labels.
        if y_true.dtype.type is np.str_ or y_pred.dtype.type is np.str_:
            labels = np.array(class_names)
    else:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    cm, labels = confusion_matrix(y_true, y_pred, labels=labels)
    report = {}
    for i, cls in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # label name to put in report
        if class_names is not None and i < len(class_names):
            cls_name = class_names[i]
        else:
            cls_name = str(cls)

        report[cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": support
        }

    # overall accuracy
    overall_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
    report["accuracy"] = overall_acc
    return report


# ==============================================================================
# Visualization and Export Functions
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names: Optional[List[str]] = None, figsize=(10, 8), cmap="Blues"):
    """Plots and displays a confusion matrix (rows = actual, cols = predicted)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # passing labels as none to letting confusion matrix handle labels completely
    cm, labels = confusion_matrix(y_true, y_pred, labels=None)

    plt.figure(figsize=figsize)
    # if labels are numeric, convert to strings for ticklabels
    if class_names is not None:
        ticklabels = class_names
    else:
        ticklabels = [str(l) for l in labels]
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=ticklabels, yticklabels=ticklabels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()


def plot_sample_predictions(images, preds, labels=None, class_names: Optional[List[str]] = None, n: int = 6):
    """
    Plot a batch of images with predicted (and optionally true) labels.
    images: tf.Tensor or np.ndarray of shape (B, H, W, C) or (B, H, W).
    preds: list/array of indices or strings.
    labels: optional list/array of true indices or strings.
    """
    images_np = np.asarray(images)
    preds_list = list(preds)
    if labels is not None:
        labels_list = list(labels)
    else:
        labels_list = None

    n = min(n, len(images_np), len(preds_list))
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = images_np[i]
        # if tensor images are in [0,1] floats or [0,255] ints, matplotlib will handle common cases
        plt.imshow(img.astype(np.uint8) if img.dtype != np.float32 and img.dtype != np.float64 else img)
        plt.axis("off")

        # prediction name
        pred_val = preds_list[i]
        if class_names is not None and isinstance(pred_val, (int, np.integer)) and pred_val < len(class_names):
            pred_name = class_names[int(pred_val)]
        else:
            pred_name = str(pred_val)

        title = f"Pred: {pred_name}"

        if labels_list is not None:
            true_val = labels_list[i]
            if class_names is not None and isinstance(true_val, (int, np.integer)) and true_val < len(class_names):
                true_name = class_names[int(true_val)]
            else:
                true_name = str(true_val)

            title += f"\nTrue: {true_name}"
            if pred_name == true_name:
                plt.title(title, color="green")
            else:
                plt.title(title, color="red")
        else:
            plt.title(title)

    plt.show()


def save_predictions_to_csv(preds, filename: Optional[str] = "predictions.csv", class_names: Optional[List[str]] = None):
    """Saves predictions to a CSV file."""
    if filename is True or filename is None:
        filename = "predictions.csv"
    if class_names:
        preds_to_save = [class_names[p] if isinstance(p, (int, np.integer)) else p for p in preds]
    else:
        preds_to_save = preds

    df = pd.DataFrame({"prediction": preds_to_save})
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")


def save_annotated_images(images, preds, out_dir="./results", class_names: Optional[List[str]] = None):
    """Saves images to disk with their predicted labels as titles."""
    images_np = np.asarray(images)
    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(images_np):
        plt.figure()
        plt.imshow(img.astype(np.uint8) if img.dtype != np.float32 and img.dtype != np.float64 else img)
        plt.axis("off")
        pred_name = class_names[preds[i]] if (class_names and isinstance(preds[i], (int, np.integer)) and preds[i] < len(class_names)) else str(preds[i])
        plt.title(f"Pred: {pred_name}")
        save_path = os.path.join(out_dir, f"sample_{i}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"Saved annotated images to {out_dir}")


class Postprocessor:
    """
    Postprocessing with custom config.
    config can include keys:
      - "accuracy": True
      - "classification_report": True
      - "confusion_matrix": True
      - "plot_samples": {"n": 6}
      - "save_csv": "path/to/file.csv" or True
      - "save_images": "path/to/out_dir" or True
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_report(self, model: tf.keras.Model, dataset: tf.data.Dataset, class_names: Optional[List[str]] = None):
        print("\n--- STAGE 3: Evaluating on Validation Data ---")

        pred_indices, true_indices = get_predictions(model, dataset)

        if self.config.get("accuracy"):
            accuracy = compute_accuracy(true_indices, pred_indices)
            print(f"\nOverall Validation Accuracy: {accuracy:.4f}")

        if self.config.get("classification_report"):
            print("\n--- Classification Report ---")
            report = classification_report(true_indices, pred_indices, class_names=class_names)
            # print per-class metrics (skip the "accuracy" key here for formatting)
            for cls, metrics in report.items():
                if cls == "accuracy":
                    continue
                print(f"{cls}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1-score']:.4f}, support={metrics['support']}")
            print(f"Overall accuracy: {report.get('accuracy', 0.0):.4f}")

        if self.config.get("confusion_matrix"):
            print("\n--- Displaying Confusion Matrix ---")
            plot_confusion_matrix(true_indices, pred_indices, class_names)

        if "plot_samples" in self.config:
            print("\n--- Visualizing Sample Predictions ---")
            try:
                sample_images, sample_labels = next(iter(dataset))
            except Exception as ex:
                print(f"Could not fetch a batch from dataset for sample plotting: {ex}")
                sample_images, sample_labels = None, None

            if sample_images is not None:
                sample_preds = predict_batch(model, sample_images)
                n_samples = int(self.config["plot_samples"].get("n", 6)) if isinstance(self.config["plot_samples"], dict) else int(self.config["plot_samples"])
                plot_sample_predictions(sample_images, sample_preds, sample_labels.numpy() if isinstance(sample_labels, tf.Tensor) else sample_labels, class_names, n=n_samples)

        if "save_csv" in self.config:
            print("\n--- Saving Predictions to CSV ---")
            filename = self.config.get("save_csv")
            if filename is True or filename is None:
                filename = "predictions.csv"
            save_predictions_to_csv(pred_indices, filename=filename, class_names=class_names)

        if "save_images" in self.config:
            print("\n--- Saving Annotated Images ---")
            try:
                sample_images, _ = next(iter(dataset))
                sample_preds = predict_batch(model, sample_images)
                out_dir = self.config.get("save_images") if self.config.get("save_images") not in (True, None) else "./results"
                save_annotated_images(sample_images, sample_preds, out_dir=out_dir, class_names=class_names)
            except Exception as ex:
                print(f"Could not save annotated images: {ex}")
