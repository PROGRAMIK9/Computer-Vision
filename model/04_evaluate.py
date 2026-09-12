import os
import argparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(data_dir, model_path, img_size, batch_size):
    test_dir = os.path.join(data_dir, "Test")
    
    if not os.path.exists(test_dir):
        print(f"Error: Could not find Test directory in {data_dir}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading test dataset...")
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False # Keep false for evaluation to match predictions with labels
    )
    
    class_names = test_ds_raw.class_names
    if "Normal" in class_names:
        normal_idx = class_names.index("Normal")
    else:
        normal_idx = [i for i, c in enumerate(class_names) if c.lower() == 'normal'][0]
        
    print(f"Classes: {class_names}")
    print(f"Normal class index: {normal_idx}")

    # Map to binary for evaluation
    def to_binary(image, label):
        is_anomaly = tf.cast(label != normal_idx, tf.float32)
        return image, is_anomaly, label # Return original label too for per-class analysis

    test_ds = test_ds_raw.map(to_binary)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating...")
    
    y_true_binary = []
    y_true_multi = []
    y_pred_probs = []
    
    for images, binary_labels, multi_labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())
        y_true_binary.extend(binary_labels.numpy())
        y_true_multi.extend(multi_labels.numpy())
        
    y_pred_binary = (np.array(y_pred_probs) > 0.5).astype(int)
    y_true_binary = np.array(y_true_binary)
    y_true_multi = np.array(y_true_multi)
    
    print("\n--- Overall Binary Classification Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Recall:    {recall_score(y_true_binary, y_pred_binary):.4f}")
    print(f"F1 Score:  {f1_score(y_true_binary, y_pred_binary):.4f}")
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Binary Confusion Matrix')
    cm_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")
    
    print("\n--- Per-Class Anomaly Detection Rate ---")
    # How well did we detect anomalies for each specific class?
    for i, cls in enumerate(class_names):
        mask = (y_true_multi == i)
        total_class = np.sum(mask)
        if total_class == 0:
            continue
            
        if i == normal_idx:
            # For Normal, we want to see how many were correctly classified as Normal (0)
            correct = np.sum(y_pred_binary[mask] == 0)
            print(f"{cls} (Normal): {correct}/{total_class} correctly identified as Normal ({correct/total_class*100:.2f}%)")
        else:
            # For Anomalies, we want to see how many were correctly classified as Anomaly (1)
            correct = np.sum(y_pred_binary[mask] == 1)
            print(f"{cls} (Anomaly): {correct}/{total_class} correctly identified as Anomaly ({correct/total_class*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the anomaly detection model.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the dataset is located")
    parser.add_argument("--model_path", type=str, default="./saved_models/model_final.keras", help="Path to the saved model")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.model_path, args.img_size, args.batch_size)
