from sklearn.metrics import classification_report as sk_classification_report

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu()
    correct = (y_true == y_pred).sum().item()
    return correct / len(y_true)


def classification_report(y_true, y_pred, class_names=None):
    """
    Generate a precision/recall/F1 report.
    """
    target_names = class_names if class_names else None
    return sk_classification_report(y_true, y_pred, target_names=target_names)


def top_k_accuracy(outputs, y_true, k=5):
    """
    Compute top-k accuracy from model outputs.
    """
    _, topk_preds = outputs.topk(k, dim=1)
    correct = topk_preds.eq(y_true.view(-1, 1).expand_as(topk_preds))
    return correct.sum().item() / len(y_true)
