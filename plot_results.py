import os
import argparse
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot TCAM1DCNN Training Curves")
    parser.add_argument("--fold", type=int, default=1, help="Fold index to plot")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name suffix (e.g. crossentropy, msle)")
    args = parser.parse_args()
    
    if args.exp_name:
        json_path = f"experiments/{args.exp_name}/fold_{args.fold}/history.json"
    else:
        json_path = f"logs/fold_{args.fold}_history.json"
        
    if not os.path.exists(json_path):
        print(f"Error: Log file '{json_path}' not found!")
        return
        
    with open(json_path, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    os.makedirs("results", exist_ok=True)
    
    # Setup aesthetic plotting parameters
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.set_title(f"TCAM1DCNN Training Loss (Fold {args.fold})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss Value", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # 2. Accuracy Plot
    train_acc_pct = [x * 100 for x in history["train_acc"]]
    val_acc_pct = [x * 100 for x in history["val_clip_acc"]]
    
    ax2.plot(epochs, train_acc_pct, label="Train Acc (Frame-level)", color="#ff7f0e", linewidth=2)
    ax2.plot(epochs, val_acc_pct, label="Val Acc (Clip-level)", color="#2ca02c", linewidth=2)
    ax2.set_title(f"TCAM1DCNN Accuracy Curves (Fold {args.fold})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    if args.exp_name:
        output_path = f"experiments/{args.exp_name}/fold_{args.fold}/curves.png"
    else:
        output_path = f"results/figures/fold_{args.fold}_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved training curves to: {output_path}")

if __name__ == "__main__":
    main()
