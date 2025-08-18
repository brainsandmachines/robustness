# dryrun_loader.py
from robustness import datasets
ds = datasets.DATASETS['imagenet']('/mnt/data/datasets/imagenet')
train_loader, val_loader = ds.make_loaders(workers=16, batch_size=256, data_aug=True)

for split, loader in [('train', train_loader), ('val', val_loader)]:
    print(f"Checking {split} loader…")
    for _i, (_x, _y) in enumerate(loader):
        pass
    print(f"Done {split}.")
