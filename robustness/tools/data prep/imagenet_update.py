import os
import subprocess
import json

# ==== CONFIG ====
IMAGENET_USER = "YOUR_USERNAME"  # <-- replace with your ImageNet username
IMAGENET_PASS = "YOUR_PASSWORD"  # <-- replace with your ImageNet password

DATA_DIR = "/mnt/data/datasets/imagenet"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TRAIN_TAR_DIR = os.path.join(DATA_DIR, "train_tars")

IMAGENET_TRAIN_TAR_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
IMAGENET_VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

# ==== Ensure dirs ====
os.makedirs(TRAIN_TAR_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==== Get official ImageNet 1K synset IDs ====
imagenet_ids_url = "https://raw.githubusercontent.com/pytorch/tutorials/main/_static/imagenet_class_index.json"
ids_text = subprocess.check_output(["wget", "-qO-", imagenet_ids_url]).decode()
imagenet_class_index = json.loads(ids_text)
synset_ids = [imagenet_class_index[str(i)][0] for i in range(len(imagenet_class_index))]

# ==== Download the BIG training tar once ====
train_big_tar = os.path.join(DATA_DIR, "ILSVRC2012_img_train.tar")
if not os.path.exists(train_big_tar):
    print("📥 Downloading training set tar (this is ~138 GB)...")
    subprocess.run([
        "wget", "--continue",
        "--user", IMAGENET_USER, "--password", IMAGENET_PASS,
        IMAGENET_TRAIN_TAR_URL, "-O", train_big_tar
    ], check=True)

# ==== Extract 1,000 per-class tarballs from the big tar ====
# This will create files like train_tars/n01440764.tar, ..., n15075141.tar
if not os.listdir(TRAIN_TAR_DIR):
    print("📦 Extracting per-class tarballs from the training tar...")
    subprocess.run(["tar", "-xf", train_big_tar, "-C", TRAIN_TAR_DIR], check=True)

# ==== Find missing classes (by folder) and extract only those ====
existing_classes = set(os.listdir(TRAIN_DIR))
missing_classes = [cls for cls in synset_ids if cls not in existing_classes]
print(f"Found {len(missing_classes)} missing classes out of {len(synset_ids)} total.")

for cls in missing_classes:
    per_class_tar = os.path.join(TRAIN_TAR_DIR, f"{cls}.tar")
    if not os.path.exists(per_class_tar):
        print(f"⚠️ Expected {per_class_tar} not found inside the big tar; skipping {cls}.")
        continue
    extract_dir = os.path.join(TRAIN_DIR, cls)
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"📂 Extracting {cls}...")
        subprocess.run(["tar", "-xf", per_class_tar, "-C", extract_dir], check=True)

# ==== Download and prepare validation set ====
val_tar_path = os.path.join(DATA_DIR, "ILSVRC2012_img_val.tar")
if not os.path.exists(os.path.join(VAL_DIR, "n01440764")):  # check if already sorted
    if not os.path.exists(val_tar_path):
        print("📥 Downloading validation set...")
        subprocess.run([
            "wget", "--continue",
            "--user", IMAGENET_USER, "--password", IMAGENET_PASS,
            IMAGENET_VAL_URL, "-O", val_tar_path
        ], check=True)
    if not os.listdir(VAL_DIR):
        print("📦 Extracting validation set...")
        subprocess.run(["tar", "-xf", val_tar_path, "-C", VAL_DIR], check=True)
    # fetch valprep.sh if needed and organize val images into class subfolders
    if not os.path.exists("valprep.sh"):
        subprocess.run([
            "wget", "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
        ], check=True)
    print("🗂 Organizing validation set...")
    subprocess.run(["bash", "valprep.sh", VAL_DIR], check=True)

print("✅ ImageNet dataset is now complete.")
