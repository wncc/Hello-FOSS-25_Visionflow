import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def decode_predictions(raw_predictions, class_names):
    """
    Converts raw model outputs (logits) into class names and confidences.

    Args:
        raw_predictions (np.array): The raw output from the model for a batch of images.
        class_names (list): A list of strings representing the class names.

    Returns:
        tuple: A tuple containing:
            - list: The predicted class names for each image.
            - list: The confidence scores for each prediction.
    """
    # Use softmax to convert the raw logits into a probability distribution
    probabilities = tf.nn.softmax(raw_predictions).numpy()
    
    # Find the index of the class with the highest probability for each prediction
    predicted_indices = np.argmax(probabilities, axis=1)
    
    # Get the confidence scores (the highest probability) and corresponding class names
    confidences = np.max(probabilities, axis=1)
    predicted_class_names = [class_names[i] for i in predicted_indices]
    
    return predicted_class_names, confidences

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates, plots, and displays a confusion matrix to evaluate model accuracy.
    

    Args:
        y_true (list): The ground truth labels.
        y_pred (list): The labels predicted by the model.
        class_names (list): A list of all possible class names for labeling the axes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names, 
        cmap='Blues'
    )
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def display_prediction(image, true_label, predicted_label, confidence):
    """
    Displays a single image with its true and predicted labels.

    The title is colored green for a correct prediction and red for an incorrect one.

    Args:
        image (np.array): The image to display (should be in [0, 1] range).
        true_label (str): The correct label for the image.
        predicted_label (str): The model's predicted label.
        confidence (float): The model's confidence in the prediction.
    """
    plt.imshow(image)
    
    title_color = 'green' if true_label == predicted_label else 'red'
    
    plt.title(
        f"True: {true_label}\nPredicted: {predicted_label} ({confidence:.2f})",
        color=title_color
    )
    plt.axis('off')
    plt.show()
