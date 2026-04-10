import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

from emotion_modeling_utils import (
    build_efficientnet_model,
    prepare_efficientnet_dataloaders,
    get_device
)

def main():
    device = get_device()
    IMAGE_SIZE = 260
    BATCH_SIZE = 32
    
    # Load Data
    print("Loading test data...")
    data = prepare_efficientnet_dataloaders(
        train_dir='archive/train',
        test_dir='archive/test',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    test_loader = data['test_loader']
    idx_to_class = data['idx_to_class']
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Load the Saved Model
    print("Rebuilding model and loading saved weights...")
    model = build_efficientnet_model(num_classes=7)
    weights_path = 'model_weights/efficientnet_b2_emotion_v2.pth'
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Collect All Predictions
    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate Metrics Table (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n--- CLASSIFICATION REPORT ---")
    print(report)

    # Build Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Emotion Classification')
    
    # Highlight your focus areas in the save name
    plt.savefig('confusion_matrix_v2.png')
    print("\nConfusion Matrix saved as 'confusion_matrix_v2.png'")
    plt.show()

if __name__ == "__main__":
    main()
