import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import VideoDeepfakeCNN, AudioDeepfakeCNN, MultimodalDeepfakeModel

def evaluate_model(model_type, test_csv, model_path, device='cpu'):
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Dataset
    print(f"Loading test dataset from {test_csv}...")
    try:
        dataset = DeepfakeDataset(test_csv)
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
        
    # 2. Load Model
    print(f"Initializing {model_type} model...")
    if model_type == "video":
        model = VideoDeepfakeCNN(pretrained=False, feature_extraction=False)
    elif model_type == "audio":
        model = AudioDeepfakeCNN(feature_extraction=False)
    elif model_type == "multimodal":
        model = MultimodalDeepfakeModel()
    else:
        raise ValueError("Invalid model type. Choose from: video, audio, multimodal")
        
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Inference Loop
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("Running predictions on test set...")
    with torch.no_grad():
        for faces, audios, labels in test_loader:
            faces, audios = faces.to(device), audios.to(device)
            
            if model_type == "video":
                outputs = model(faces)
            elif model_type == "audio":
                outputs = model(audios)
            elif model_type == "multimodal":
                outputs = model(faces, audios)
                
            probs = outputs.squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Model Type: {model_type.upper()}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print("====================================================")
    
    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    
    cm_path = f"results/{model_type}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")
    
    # 6. Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_type.upper()}')
    plt.legend(loc="lower right")
    
    roc_path = f"results/{model_type}_roc_auc_curve.png"
    plt.savefig(roc_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"ROC-AUC Curve saved to {roc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    parser.add_argument("--model", type=str, default="multimodal", choices=["video", "audio", "multimodal"],
                        help="Model type to evaluate")
    parser.add_argument("--csv", type=str, required=True, help="Path to test/validation CSV")
    parser.add_argument("--weight", type=str, required=True, help="Path to saved .pth weight checkpoint")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_model(
        model_type=args.model,
        test_csv=args.csv,
        model_path=args.weight,
        device=device
    )
