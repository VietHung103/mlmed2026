import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from configs import TrainConfig
from data.yolo_dataset import YOLODetection, collate_fn
from data.transforms import Compose, Resize, RandomHorizontalFlip, Normalize
from models.retinanet import RetinaNet


# =========================================================
# Build COCO GT once
# =========================================================
def build_coco_gt(dataset, num_classes):
    coco_dict = {"images": [], "annotations": [], "categories": []}

    for img_id in range(len(dataset)):
        img, target = dataset[img_id]
        h, w = img.shape[1:]

        coco_dict["images"].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": f"{img_id}.jpg"
        })

        for i in range(len(target["boxes"])):
            x1, y1, x2, y2 = target["boxes"][i]
            coco_dict["annotations"].append({
                "id": len(coco_dict["annotations"]),
                "image_id": img_id,
                "category_id": int(target["labels"][i]),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })

    for c in range(num_classes):
        coco_dict["categories"].append({
            "id": c,
            "name": f"class_{c}"
        })

    with open("temp_gt.json", "w") as f:
        json.dump(coco_dict, f)

    return COCO("temp_gt.json")


# =========================================================
# Validation
# =========================================================
@torch.no_grad()
def validate(model, val_loader, coco_gt, device):
    model.eval()

    results = []
    img_id = 0
    total_val_loss = 0.0

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for images, targets in pbar:
        images = [img.to(device) for img in images]

        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)

        # ---- Validation loss ----
        losses = model(images, targets)
        val_loss = losses["loss_total"]
        total_val_loss += val_loss.item()

        # ---- Inference ----
        outputs = model(images)

        for b in range(len(outputs)):
            boxes = outputs[b]["boxes"].cpu().numpy()
            scores = outputs[b]["scores"].cpu().numpy()
            labels = outputs[b]["labels"].cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                results.append({
                    "image_id": img_id,
                    "category_id": int(labels[i]),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                    ],
                    "score": float(scores[i])
                })
            img_id += 1

    avg_val_loss = total_val_loss / len(val_loader)

    if len(results) == 0:
        return avg_val_loss, 0, 0, 0, 0, 0, 0

    with open("temp_pred.json", "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes("temp_pred.json")

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return (
        avg_val_loss,
        coco_eval.stats[0],  # AP 0.50:0.95
        coco_eval.stats[1],  # AP 0.50
        coco_eval.stats[2],  # AP 0.75
        coco_eval.stats[8],  # AR 0.50:0.95
        coco_eval.stats[6],  # AR (maxDet=100)
        coco_eval.stats[7],  # AR mid
    )


# =========================================================
# TRAIN
# =========================================================
def main():
    cfg = TrainConfig()

    train_tf = Compose([
        Resize(cfg.img_size),
        RandomHorizontalFlip(0.5),
        Normalize()
    ])

    val_tf = Compose([
        Resize(cfg.img_size),
        Normalize()
    ])

    train_ds = YOLODetection("dataset/train/images",
                             "dataset/train/labels",
                             transforms=train_tf)

    val_ds = YOLODetection("dataset/valid/images",
                           "dataset/valid/labels",
                           transforms=val_tf)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RetinaNet(
        num_classes=cfg.num_classes,
        backbone_name="convnext_tiny",
        pretrained=True,
        fpn_out_channels=cfg.fpn_out_channels,
        anchor_sizes=cfg.anchor_sizes,
        anchor_ratios=cfg.anchor_ratios,
        anchor_scales=cfg.anchor_scales,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=(device == "cuda"))

    coco_gt = build_coco_gt(val_ds, cfg.num_classes)

    os.makedirs("runs", exist_ok=True)

    # Metric history
    train_loss_history = []
    val_loss_history = []

    ap_5095_history = []
    ap_50_history = []
    ap_75_history = []

    ar_5095_history = []
    ar_50_history = []
    ar_75_history = []

    best_map = 0.0

    for epoch in range(cfg.epochs):

        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{cfg.epochs}",
                    leave=False)

        for images, targets in pbar:
            images = [img.to(device) for img in images]

            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device == "cuda")):
                losses = model(images, targets)
                loss = losses["loss_total"]

            if torch.isnan(loss):
                print("NaN detected. Stop.")
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        (val_loss,
         ap_5095,
         ap_50,
         ap_75,
         ar_5095,
         ar_50,
         ar_75) = validate(model, val_loader, coco_gt, device)

        # Save history
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)

        ap_5095_history.append(ap_5095)
        ap_50_history.append(ap_50)
        ap_75_history.append(ap_75)

        ar_5095_history.append(ar_5095)
        ar_50_history.append(ar_50)
        ar_75_history.append(ar_75)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"AP50-95: {ap_5095:.4f} | AP50: {ap_50:.4f} | AP75: {ap_75:.4f}")

        # ---------------- PLOT ----------------
        plt.figure(figsize=(16,8))

        plt.subplot(2,3,1)
        plt.plot(train_loss_history)
        plt.title("Train Loss")

        plt.subplot(2,3,2)
        plt.plot(val_loss_history)
        plt.title("Val Loss")

        plt.subplot(2,3,3)
        plt.plot(ap_5095_history, label="AP50-95")
        plt.plot(ap_50_history, label="AP50")
        plt.plot(ap_75_history, label="AP75")
        plt.legend()
        plt.title("Average Precision")

        plt.subplot(2,3,4)
        plt.plot(ar_5095_history, label="AR50-95")
        plt.plot(ar_50_history, label="AR")
        plt.plot(ar_75_history, label="AR mid")
        plt.legend()
        plt.title("Average Recall")

        plt.tight_layout()
        plt.savefig("runs/metrics.png")
        plt.close()

        # ---------------- SAVE ----------------
        torch.save(model.state_dict(), "runs/last.pt")

        if ap_5095 > best_map:
            best_map = ap_5095
            torch.save(model.state_dict(), "runs/best.pt")

    print("\nTraining finished")
    print("Best AP50-95:", best_map)


if __name__ == "__main__":
    main()
