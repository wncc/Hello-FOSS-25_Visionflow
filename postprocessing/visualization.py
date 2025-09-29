import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10,8), cmap="Blues"):
    """
    Plot a confusion matrix.

    Args:
        y_true (list/array): Ground truth labels.
        y_pred (list/array): Predicted labels.
        class_names (list, optional): Class names for labeling axes.
        figsize (tuple): Figure size.
        cmap (str): Color map.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()

def plot_sample_predictions(images, preds, labels=None, class_names=None, n=6):
    """
    Plot a batch of images with predicted (and optionally true) labels.

    Args:
        images (torch.Tensor or list): Batch of images (B, C, H, W).
        preds (list): Predicted indices or names.
        labels (list, optional): True labels.
        class_names (list, optional): Map indices to class names.
        n (int): Number of images to display.
    """
    import torch
    import matplotlib.pyplot as plt

    images = images[:n]
    preds = preds[:n]
    if labels:
        labels = labels[:n]

    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()  # CHW → HWC
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis("off")
        title = f"Pred: {preds[i]}"
        if labels:
            title += f"\nTrue: {labels[i]}"
        plt.title(title)
    plt.show()
