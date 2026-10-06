"""Fine-tune ResNet-50 on Food-101.

Uses the OFFICIAL Food-101 split (75,750 train / 25,250 test), so the accuracy
printed here is comparable to published numbers: the official test set is
human-cleaned, while the train set carries deliberate label noise.

Data comes from the HuggingFace mirror rather than
`torchvision.datasets.Food101`. Same images and same split, but torchvision
points at data.vision.ee.ethz.ch, which measured 0.17 MB/s during development
against 3.4 MB/s from HuggingFace.

The whole backbone is trained, not just the classifier head. A frozen backbone
is a linear probe over ImageNet features and plateaus around 45% on these 101
fine-grained classes; unfreezing is what reaches the 83.46% reported here.

    python train.py --cache-dir ./data --epochs 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from labels import FOOD_101_CLASSES, NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Food101Torch(Dataset):
    """Applies transforms and rewrites labels into labels.py order.

    Must live at module scope: on Windows the DataLoader spawns workers rather
    than forking, so this class gets pickled, and a class defined inside a
    function cannot be.
    """

    def __init__(self, hf_split, transform, remap: list[int]):
        self.split = hf_split
        self.transform = transform
        self.remap = remap

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[index]
        return self.transform(row["image"].convert("RGB")), self.remap[row["label"]]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
            _, top5_idx = outputs.float().topk(5, dim=1)
            hits = top5_idx.eq(labels.view(-1, 1))
            top1 += hits[:, 0].sum().item()
            top5 += hits.any(dim=1).sum().item()
            total += labels.size(0)
    return 100 * top1 / total, 100 * top5 / total


def load_splits(cache_dir: str):
    """Import `datasets` only after HF_HOME is set, or it pins the cache elsewhere."""
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_dir) / "datasets"))

    # Some machines run a TLS-intercepting antivirus whose root is in the OS
    # trust store but not in certifi, which fails the download with
    # CERTIFICATE_VERIFY_FAILED. Route TLS through the OS store when available.
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    from datasets import load_dataset

    dataset = load_dataset("ethz/food101")
    # HuggingFace calls the official test split "validation"; same 25,250 images.
    return dataset["train"], dataset["validation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="./data",
                        help="Where the HuggingFace dataset cache lives (needs ~10GB)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    # Each Windows worker respawns and re-imports torch, costing ~1GB RSS. Train
    # workers are persistent, so during validation both pools are alive at once:
    # the peak is (workers + val_workers) GB.
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--val-workers", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=0.01)
    parser.add_argument("--backbone-lr", type=float, default=0.001)
    parser.add_argument("--out", default="weights/resnet50_food101.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | device {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    hf_train, hf_test = load_splits(args.cache_dir)

    # The checkpoint's index-to-name mapping must match what inference ships, or
    # every prediction is silently mislabeled. Rather than trust the order, remap
    # HuggingFace label ids onto labels.py order. These really do differ:
    # HuggingFace sorts cheesecake before cheese_plate.
    hf_names = hf_train.features["label"].names
    assert sorted(hf_names) == sorted(FOOD_101_CLASSES), "HF classes != labels.py"
    remap = [FOOD_101_CLASSES.index(name) for name in hf_names]
    differing = sum(1 for i, r in enumerate(remap) if i != r)
    print(f"label remap: {differing} of {NUM_CLASSES} positions differ", flush=True)

    train_tf, eval_tf = build_transforms()
    train_ds = Food101Torch(hf_train, train_tf, remap)
    test_ds = Food101Torch(hf_test, eval_tf, remap)
    print(f"train {len(train_ds)} | test {len(test_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    # Not persistent: the eval pool is torn down as soon as the epoch is scored,
    # so it does not sit in RAM alongside the training pool for the next epoch.
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.val_workers, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"resnet50: {trainable:,} trainable parameters", flush=True)

    # Pretrained features need gentler updates than the randomly-initialised head.
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}],
        momentum=0.9, weight_decay=1e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.backbone_lr, args.head_lr],
        epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.25,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_top1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = seen = 0
        epoch_start = time.perf_counter()

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            seen += labels.size(0)

            if step % 100 == 0 or step == len(train_loader):
                rate = seen / (time.perf_counter() - epoch_start)
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / seen:.3f} acc={100 * correct / seen:.1f}% "
                      f"({rate:.0f} img/s)", flush=True)

        train_loss = running_loss / seen
        train_acc = 100 * correct / seen
        val_top1, val_top5 = evaluate(model, test_loader, device)
        mins = (time.perf_counter() - epoch_start) / 60

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc:.1f}%  val_top1={val_top1:.2f}%  "
              f"val_top5={val_top5:.2f}%  ({mins:.1f} min)", flush=True)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 2), "val_top1": round(val_top1, 2),
                        "val_top5": round(val_top5, 2), "minutes": round(mins, 2)})

        if val_top1 > best_top1:
            best_top1 = val_top1
            # Labels travel WITH the weights, so inference cannot pair a
            # checkpoint with a different class order by accident.
            torch.save({
                "arch": "resnet50",
                "num_classes": NUM_CLASSES,
                "classes": FOOD_101_CLASSES,
                "state_dict": model.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "epoch": epoch,
            }, out_path)
            print(f"  saved {out_path} (best top-1 {best_top1:.2f}%)", flush=True)

    Path("training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone. Best val top-1: {best_top1:.2f}%  ->  {out_path}")


if __name__ == "__main__":
    main()
