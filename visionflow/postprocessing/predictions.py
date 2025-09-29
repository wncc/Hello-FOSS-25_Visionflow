import torch

def get_predictions(model, dataloader, device="cpu"):
    """
    Run inference on a dataloader and return predictions + labels.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader for test/val dataset.
        device (str): Device to run on ("cpu" or "cuda").

    Returns:
        preds (list[int]): Predicted class indices.
        labels (list[int]): Ground truth class indices.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return preds, labels


def predict_single(model, image, device="cpu", class_names=None):
    """
    Predict class for a single image.

    Args:
        model (torch.nn.Module): Trained model.
        image (torch.Tensor): Single image tensor (C, H, W).
        device (str): Device.
        class_names (list[str], optional): Class name mapping.

    Returns:
        str or int: Predicted class name (if provided) or index.
    """
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # add batch dim
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()

    if class_names:
        return class_names[pred_idx]
    return pred_idx


def predict_batch(model, images, device="cpu", class_names=None):
    """
    Predict classes for a batch of images.

    Args:
        model (torch.nn.Module): Trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (str): Device.
        class_names (list[str], optional): Class name mapping.

    Returns:
        list[str] or list[int]: Predicted class names or indices.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds = predicted.cpu().numpy()

    if class_names:
        preds = [class_names[p] for p in preds]
    return preds
