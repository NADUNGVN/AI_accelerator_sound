import os
import shutil

def remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")

def main():
    # Remove nested redundant folders
    remove_if_exists("checkpoints/baseline_v1/baseline_v1")
    remove_if_exists("results/baseline_v1/baseline_v1")
    remove_if_exists("logs/baseline_v1/baseline_v1")
    
    # Clean up empty .gitkeep files inside backup folders
    remove_if_exists("checkpoints/baseline_v1/.gitkeep")
    remove_if_exists("logs/baseline_v1/.gitkeep")
    
    # Ensure correct empty directories for results
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    
    print("Clean up completed successfully!")

if __name__ == "__main__":
    main()
