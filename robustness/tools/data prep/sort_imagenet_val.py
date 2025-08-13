from pathlib import Path
import tarfile, shutil, os
import numpy as np
from scipy.io import loadmat

IMAGENET_ROOT = Path("/mnt/data/datasets/imagenet")
VAL_TAR = IMAGENET_ROOT / "ILSVRC2012_img_val.tar"
VAL_DIR = IMAGENET_ROOT / "val"                      # already exists
TMP_DIR = IMAGENET_ROOT / "_val_tmp"                 # staging
DEVKIT = IMAGENET_ROOT / "ILSVRC2012_devkit_t12"     # extracted devkit root
GT_TXT = DEVKIT / "data" / "ILSVRC2012_validation_ground_truth.txt"
META_MAT = DEVKIT / "data" / "meta.mat"

def ensure(p, what): 
    if not p.exists(): raise FileNotFoundError(f"Expected {what} at: {p}")

def extract_val_tar():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(VAL_TAR, "r") as tf:
        tf.extractall(TMP_DIR)

def build_clsidx_to_wnid_from_meta():
    # meta.mat has a 'synsets' struct array with fields including 'WNID' and 'ILSVRC2012_ID'
    meta = loadmat(META_MAT, squeeze_me=True)
    synsets = meta['synsets']  # array of structs
    # some entries are non-leaf; ILSVRC2012 classes are the 1000 with ILSVRC2012_ID=1..1000
    wnid_by_id = {}
    for s in np.atleast_1d(synsets):
        wnid = s['WNID'].item() if isinstance(s['WNID'], np.ndarray) else s['WNID']
        ilsvrc_id = int(s['ILSVRC2012_ID']) if 'ILSVRC2012_ID' in s.dtype.names else None
        if ilsvrc_id is not None and 1 <= ilsvrc_id <= 1000:
            wnid_by_id[ilsvrc_id] = wnid
    if len(wnid_by_id) != 1000:
        raise RuntimeError(f"Expected 1000 ILSVRC2012_ID entries, got {len(wnid_by_id)}")
    # return list of 1000 wnids in 1..1000 order
    return [wnid_by_id[i] for i in range(1, 1001)]

def number_from_val_filename(p):
    # ILSVRC2012_val_00000001.JPEG -> 1
    return int(p.stem.split("_")[-1])

def main():
    # sanity
    ensure(VAL_TAR, "validation tar")
    ensure(VAL_DIR, "val directory")
    ensure(DEVKIT, "extracted devkit directory")
    ensure(GT_TXT, "validation ground-truth file")
    ensure(META_MAT, "meta.mat")

    # 1) extract validation tar into a temp flat folder
    print("Extracting validation tar ...")
    extract_val_tar()

    # 2) read ground truth (50k lines: class indices 1..1000)
    print("Reading ground truth ...")
    gt = [int(x.strip()) for x in open(GT_TXT) if x.strip()]
    if len(gt) != 50000:
        raise RuntimeError(f"Ground truth lines != 50000 (got {len(gt)})")

    # 3) build mapping from class index (1..1000) -> WNID
    print("Building class-index -> WNID from meta.mat ...")
    clsidx_to_wnid = build_clsidx_to_wnid_from_meta()

    # 4) make per-class folders
    print("Creating val/<WNID> folders ...")
    for wnid in clsidx_to_wnid:
        (VAL_DIR / wnid).mkdir(parents=True, exist_ok=True)

    # 5) move images into their target folders
    print("Sorting images ...")
    images = sorted([p for p in TMP_DIR.iterdir() if p.suffix.lower() in [".jpeg", ".jpg"]])
    if len(images) != 50000:
        print(f"Warning: expected 50,000 images extracted, found {len(images)}")

    for img in images:
        k = number_from_val_filename(img)   # 1..50000
        cls_idx = gt[k - 1]                 # 1..1000
        wnid = clsidx_to_wnid[cls_idx - 1]
        dst = VAL_DIR / wnid / img.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(img), str(dst))

    # 6) verify counts (should be 50 per class)
    print("Verifying per-class counts ...")
    bad = 0
    for wnid in clsidx_to_wnid:
        n = len(list((VAL_DIR / wnid).glob("*.JP*G")))
        if n != 50:
            bad += 1
            print(f"  {wnid}: {n} (expected 50)")
    print("DONE. ✅" if bad == 0 else f"Completed with {bad} class count issues.")

    # 7) cleanup temp
    try:
        TMP_DIR.rmdir()
    except OSError:
        shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    main()
