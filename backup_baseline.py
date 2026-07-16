import os
import shutil

def backup_dir(src, dest):
    if os.path.exists(src):
        os.makedirs(dest, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"Backed up '{src}' to '{dest}'")
    else:
        print(f"Source directory '{src}' does not exist, skipping backup.")

def main():
    print("Backing up old baseline_v1 artifacts (archived 56.82% accuracy)...")
    backup_dir("checkpoints", "checkpoints/baseline_v1")
    backup_dir("logs", "logs/baseline_v1")
    backup_dir("results", "results/baseline_v1")
    print("Backup completed successfully!")

if __name__ == "__main__":
    main()
