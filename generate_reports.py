import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_report_for_run(predictions, class_names, title, out_text_file, out_image_path):
    y_true = [p["label"] for p in predictions]
    y_pred = [p["predicted"] for p in predictions]
    
    # Calculate confusion matrix
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Calculate metrics per class
    report = []
    report.append(f"Classification Report - {title}\n")
    report.append(f"{'Class Name':20s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    report.append("-" * 68)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        report.append(f"{class_names[i]:20s} | {precision:10.4f} | {recall:10.4f} | {f1:10.4f} | {support:8d}")
        
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    total_support = np.sum(supports)
    
    report.append("-" * 68)
    report.append(f"{'Accuracy':20s} | {'':10s} | {'':10s} | {accuracy:10.4f} | {total_support:8d}")
    report.append(f"{'Macro Avg':20s} | {macro_precision:10.4f} | {macro_recall:10.4f} | {macro_f1:10.4f} | {total_support:8d}")
    
    # Save text report
    with open(out_text_file, "w") as f:
        f.write("\n".join(report))
    print(f"Saved classification report to: {out_text_file}")
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names, yticklabels=class_names,
           title=f"Confusion Matrix - {title}",
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate class names on x axis
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate matrix cells with values
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(out_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to: {out_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrix and Classification Reports")
    parser.add_argument("--fold", type=int, default=1, help="Fold index")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (e.g. crossentropy, msle)")
    args = parser.parse_args()
    
    exp_dir = f"experiments/{args.exp_name}/fold_{args.fold}"
    preds_path = f"{exp_dir}/predictions.json"
    
    if not os.path.exists(preds_path):
        print(f"Error: Predictions file '{preds_path}' not found!")
        return
        
    with open(preds_path, "r") as f:
        preds_data = json.load(f)
        
    class_names = [
        'air_conditioner','car_horn','children_playing','dog_bark',
        'drilling','engine_idling','gun_shot','jackhammer','siren','street_music'
    ]
    
    # 1. Best Validation Model
    if "best_val_model_predictions" in preds_data and len(preds_data["best_val_model_predictions"]) > 0:
        generate_report_for_run(
            preds_data["best_val_model_predictions"],
            class_names,
            f"{args.exp_name.upper()} (Best Val Model, Fold {args.fold})",
            f"{exp_dir}/classification_report_best.txt",
            f"{exp_dir}/confusion_matrix_best.png"
        )
        
    # 2. Ensemble Model
    if "ensemble_model_predictions" in preds_data and len(preds_data["ensemble_model_predictions"]) > 0:
        generate_report_for_run(
            preds_data["ensemble_model_predictions"],
            class_names,
            f"{args.exp_name.upper()} (Ensemble, Fold {args.fold})",
            f"{exp_dir}/classification_report_ensemble.txt",
            f"{exp_dir}/confusion_matrix_ensemble.png"
        )

if __name__ == "__main__":
    main()
