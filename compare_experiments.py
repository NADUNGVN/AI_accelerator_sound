import os
import json
import matplotlib.pyplot as plt

def main():
    msle_hist_path = "experiments/msle/fold_1/history.json"
    ce_hist_path = "experiments/crossentropy/fold_1/history.json"
    
    if not os.path.exists(msle_hist_path) or not os.path.exists(ce_hist_path):
        print("Error: Missing history.json in experiments/msle/fold_1/ or experiments/crossentropy/fold_1/")
        return
        
    with open(msle_hist_path, "r") as f:
        msle_hist = json.load(f)
    with open(ce_hist_path, "r") as f:
        ce_hist = json.load(f)
        
    epochs = range(1, len(msle_hist["train_loss"]) + 1)
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Train Loss Plot
    color_msle = "#1f77b4"
    color_ce = "#d62728"
    
    ax1.set_title("Training Loss Comparison (Fold 1)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs")
    
    ax1.set_ylabel("MSLE Loss Value", color=color_msle)
    line1 = ax1.plot(epochs, msle_hist["train_loss"], label="MSLE Loss", color=color_msle, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_msle)
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Cross Entropy Loss Value", color=color_ce)
    line2 = ax1_twin.plot(epochs, ce_hist["train_loss"], label="CrossEntropy Loss", color=color_ce, linewidth=2, linestyle="--")
    ax1_twin.tick_params(axis='y', labelcolor=color_ce)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 2. Validation Accuracy Plot
    msle_val_pct = [x * 100 for x in msle_hist["val_clip_acc"]]
    ce_val_pct = [x * 100 for x in ce_hist["val_clip_acc"]]
    
    ax2.plot(epochs, msle_val_pct, label="MSLE Val Acc", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, ce_val_pct, label="CrossEntropy Val Acc", color="#ff7f0e", linewidth=2, linestyle="--")
    ax2.set_title("Validation Clip Accuracy Comparison (Fold 1)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    output_path = "results/figures/fold_1_loss_acc_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved comparison curves to: {output_path}")

if __name__ == "__main__":
    main()
