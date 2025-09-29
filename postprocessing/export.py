import pandas as pd

def save_predictions_to_csv(preds, filename="predictions.csv", class_names=None):
    """
    Save predictions to a CSV file.

    Args:
        preds (torch.Tensor or list): Predicted class indices.
        filename (str): Path to save CSV.
        class_names (list, optional): If provided, map indices to class names.
    """
    preds = preds.cpu().numpy() if hasattr(preds, "cpu") else preds
    if class_names:
        preds = [class_names[p] for p in preds]

    df = pd.DataFrame({"prediction": preds})
    df.to_csv(filename, index=False)
    print(f"[INFO] Saved predictions to {filename}")


import os
import matplotlib.pyplot as plt

def save_annotated_images(images, preds, out_dir="./results", class_names=None):
    """
    Save images with predicted labels overlaid.

    Args:
        images (torch.Tensor): Batch of images.
        preds (torch.Tensor or list): Predicted indices.
        out_dir (str): Output directory.
        class_names (list, optional): If provided, use names instead of indices.
    """
    os.makedirs(out_dir, exist_ok=True)

    preds = preds.cpu().numpy() if hasattr(preds, "cpu") else preds
    if class_names:
        preds = [class_names[p] for p in preds]

    for i, img in enumerate(images):
        plt.imshow(img.permute(1, 2, 0))  # CHW → HWC
        plt.axis("off")
        plt.title(f"Pred: {preds[i]}")
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"))
        plt.close()

    print(f"[INFO] Saved annotated images to {out_dir}")
