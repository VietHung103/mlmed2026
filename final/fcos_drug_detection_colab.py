
import os, math, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision.ops import nms, generalized_box_iou_loss

import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:
    BASE_DIR      = "/content/dataset/SAP_BABA_CLEAN/SAP_BABA_CLEAN"
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
    TRAIN_LBL_DIR = os.path.join(BASE_DIR, "train", "labels")
    VAL_IMG_DIR   = os.path.join(BASE_DIR, "valid", "images")
    VAL_LBL_DIR   = os.path.join(BASE_DIR, "valid", "labels")
    TEST_IMG_DIR  = os.path.join(BASE_DIR, "test",  "images")
    TEST_LBL_DIR  = os.path.join(BASE_DIR, "test",  "labels")

    TRAIN_LBL_BBOX = "/content/working/labels_bbox/train"
    VAL_LBL_BBOX   = "/content/working/labels_bbox/valid"
    TEST_LBL_BBOX  = "/content/working/labels_bbox/test"

    NUM_CLASSES    = 12        
    IMG_SIZE       = 512          

    FPN_STRIDES    = [8, 16, 32, 64, 128]   
    SCALE_RANGES   = [
        [0,   64],
        [64,  128],
        [128, 256],
        [256, 512],
        [512, 1e8],
    ]

    EPOCHS         = 30
    BATCH_SIZE     = 8            
    LR             = 2e-4        
    BACKBONE_LR    = 2e-5
    WEIGHT_DECAY   = 1e-4
    FREEZE_EPOCHS  = 2
    GRAD_CLIP      = 1.0
    USE_AMP        = True       

    FOCAL_ALPHA    = 0.25
    FOCAL_GAMMA    = 2.0

    SCORE_THRESH   = 0.05
    NMS_THRESH     = 0.5
    MAX_DETS       = 100

    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    SEED           = 42
    NUM_WORKERS    = 4          
    SAVE_DIR       = "/content/drive/MyDrive/fcos_checkpoints"
    SAVE_EVERY     = 5

    EARLY_STOP_PATIENCE  = 7     
    EARLY_STOP_MIN_DELTA = 1e-4  

    CLASS_NAMES = [
        "Aspirin", "Augmentin", "Cipro",    "Coumadin",
        "Doxycycline", "Flagyl", "Lantus",  "Lipitor",
        "Nexium",  "Pradaxa",   "Zithromax","Zyrtec",
    ]


cfg = Config()
for d in [cfg.SAVE_DIR, cfg.TRAIN_LBL_BBOX,
          cfg.VAL_LBL_BBOX, cfg.TEST_LBL_BBOX]:
    os.makedirs(d, exist_ok=True)

random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

def _parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        vals   = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    n = len(vals)
    if n == 4:                   
        cx, cy, w, h = vals
        if w < 1e-4 or h < 1e-4:
            return None
        return (cls_id,
                float(np.clip(cx,0,1)), float(np.clip(cy,0,1)),
                float(np.clip(w, 0,1)), float(np.clip(h, 0,1)))
    if n >= 5:                          
        if n % 2 != 0:
            vals = vals[:-1]
        if len(vals) < 4:
            return None
        xs = np.clip(vals[0::2], 0, 1)
        ys = np.clip(vals[1::2], 0, 1)
        x1,x2 = float(xs.min()), float(xs.max())
        y1,y2 = float(ys.min()), float(ys.max())
        w = x2-x1; h = y2-y1
        if w < 1e-4 or h < 1e-4:
            return None
        return (cls_id, x1+w/2, y1+h/2, w, h)
    return None


def preprocess_labels():
    splits = [
        (cfg.TRAIN_LBL_DIR, cfg.TRAIN_LBL_BBOX, "train"),
        (cfg.VAL_LBL_DIR,   cfg.VAL_LBL_BBOX,   "valid"),
        (cfg.TEST_LBL_DIR,  cfg.TEST_LBL_BBOX,   "test"),
    ]
    print("=" * 60)
    print("  LABEL PREPROCESSING  (polygon + bbox -> YOLO bbox)")
    print("=" * 60)
    for src_dir, dst_dir, name in splits:
        src, dst = Path(src_dir), Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        src_files = sorted(src.glob("*.txt"))
        dst_files = list(dst.glob("*.txt"))
        if len(dst_files) == len(src_files) > 0:
            print(f"  [{name}] Already done ({len(dst_files)} files). Skipping.")
            continue
        converted = 0; skipped = 0; empty = 0
        for txt in tqdm(src_files, desc=f"  {name}", leave=False):
            out = []
            for line in open(txt).readlines():
                r = _parse_line(line)
                if r:
                    out.append(f"{r[0]} {r[1]:.8f} {r[2]:.8f} {r[3]:.8f} {r[4]:.8f}")
                    converted += 1
                elif line.strip():
                    skipped += 1
            with open(dst / txt.name, "w") as f:
                f.write("\n".join(out))
            if not out:
                empty += 1
        print(f"  [{name}]  {converted} boxes | {skipped} skipped | {empty} empty")
    print("\n Preprocessing complete!\n")


def verify_labels(n=3):
    print("── Label verification ──")
    for name, d in [("train", cfg.TRAIN_LBL_BBOX), ("valid", cfg.VAL_LBL_BBOX)]:
        files  = sorted(Path(d).glob("*.txt"))
        empty  = sum(1 for f in files if not open(f).read().strip())
        sample = random.sample(files, min(n, len(files)))
        print(f"\n  {name}: {len(files)} files, {empty} empty")
        for f in sample:
            lines = open(f).read().strip().split("\n")[:2]
            print(f"    {f.name}: {lines}")
    print()



IMG_EXT = {".jpg",".jpeg",".png"}

class DrugDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir    = img_dir
        self.transforms = transforms
        self.samples    = []
        for fname in sorted(os.listdir(img_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXT:
                continue
            stem = os.path.splitext(fname)[0]
            lbl  = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl):
                self.samples.append((fname, lbl))
        if not self.samples:
            raise RuntimeError(
                f"No pairs found.\n  img: {img_dir}\n  lbl: {lbl_dir}\n"
                "   Run preprocess_labels() first!")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, lbl_path = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(
              os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        boxes, labels = [], []
        for line in open(lbl_path):
            p = line.strip().split()
            if len(p) != 5: continue
            cls_id      = int(p[0])
            cx,cy,w,h   = map(float, p[1:])
            x1 = max(0.,   (cx-w/2)*W)
            y1 = max(0.,   (cy-h/2)*H)
            x2 = min(float(W), (cx+w/2)*W)
            y2 = min(float(H), (cy+h/2)*H)
            if (x2-x1) < 1. or (y2-y1) < 1.: continue
            boxes.append([x1,y1,x2,y2])
            labels.append(cls_id)        

        if self.transforms:
            out    = self.transforms(image=img, bboxes=boxes, labels=labels)
            img    = out["image"]
            boxes  = out["bboxes"]
            labels = out["labels"]

        boxes  = (torch.as_tensor(boxes,  dtype=torch.float32)
                  if boxes  else torch.zeros((0,4), dtype=torch.float32))
        labels = (torch.as_tensor(labels, dtype=torch.int64)
                  if labels else torch.zeros((0,),  dtype=torch.int64))

        return img, {"boxes": boxes, "labels": labels,
                     "image_id": torch.tensor([idx])}


MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_train_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["labels"], min_visibility=0.3))

def get_val_transforms():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["labels"]))


class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        convnext = tv_models.convnext_small(
            weights=tv_models.ConvNeXt_Small_Weights.DEFAULT)
        feats = convnext.features
        self.stem    = feats[0]          
        self.stage1  = feats[1]
        self.down1   = feats[2]          
        self.stage2  = feats[3]
        self.down2   = feats[4]          
        self.stage3  = feats[5]
        self.down3   = feats[6]          
        self.stage4  = feats[7]
        self.out_channels = [96, 192, 384, 768]
        print("   ConvNeXt-Small pretrained backbone loaded")

    def forward(self, x):
        c2 = self.stage1(self.stem(x))           
        c3 = self.stage2(self.down1(c2))         
        c4 = self.stage3(self.down2(c3))         
        c5 = self.stage4(self.down3(c4))          
        return c2, c3, c4, c5

class FPN(nn.Module):
    def __init__(self, in_chs, out_ch=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.out = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_chs])
        self.p6 = nn.Conv2d(in_chs[-1], out_ch, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_ch,     out_ch, 3, stride=2, padding=1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        srcs  = [c3, c4, c5]
        lats  = [l(s) for l, s in zip(self.lat[1:], srcs)]
        for i in range(len(lats)-2, -1, -1):
            lats[i] = lats[i] + F.interpolate(
                lats[i+1], size=lats[i].shape[-2:], mode="nearest")
        outs = [o(l) for o, l in zip(self.out[1:], lats)]
        p6   = self.p6(c5)
        p7   = self.p7(F.relu(p6))
        return outs + [p6, p7]   

class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))
    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=12, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        cls_tower, reg_tower = [], []
        for _ in range(num_convs):
            cls_tower += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(32, in_ch),
                nn.ReLU(inplace=True)]
            reg_tower += [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.GroupNorm(32, in_ch),
                nn.ReLU(inplace=True)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_pred = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_ch, 4,           3, padding=1)
        self.ctr_pred = nn.Conv2d(in_ch, 1,           3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        prior = 0.01
        nn.init.constant_(self.cls_pred.bias,
                          -math.log((1-prior)/prior))

    def forward(self, features):
        cls_outs, reg_outs, ctr_outs = [], [], []
        for i, feat in enumerate(features):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)
            cls_outs.append(self.cls_pred(cls_feat))
            reg_outs.append(
                torch.exp(self.scales[i](self.reg_pred(reg_feat))))
            ctr_outs.append(self.ctr_pred(reg_feat))
        return cls_outs, reg_outs, ctr_outs



def get_points(features, strides, device):
    all_points = []
    for feat, stride in zip(features, strides):
        H, W   = feat.shape[-2:]
        ys     = torch.arange(0, H, device=device) * stride + stride // 2
        xs     = torch.arange(0, W, device=device) * stride + stride // 2
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts    = torch.stack([gx.ravel(), gy.ravel()], dim=1).float()
        all_points.append(pts)
    return all_points      


def fcos_targets(points_list, gt_boxes, gt_labels,
                 strides, scale_ranges, num_classes):
    device     = gt_boxes.device
    all_points = torch.cat(points_list, dim=0)  
    N          = all_points.shape[0]

    level_ids  = torch.cat([
        torch.full((pts.shape[0],), i, dtype=torch.int64, device=device)
        for i, pts in enumerate(points_list)])

    cls_t = torch.full((N,), num_classes, dtype=torch.int64, device=device)
    reg_t = torch.zeros((N, 4), dtype=torch.float32, device=device)
    ctr_t = torch.zeros((N,),   dtype=torch.float32, device=device)

    if gt_boxes.numel() == 0:
        return cls_t, reg_t, ctr_t

    px, py = all_points[:, 0], all_points[:, 1]

    for m in range(len(gt_boxes)):
        x1,y1,x2,y2 = gt_boxes[m]
        label        = gt_labels[m].item()
        lvl_min, lvl_max = 0, len(scale_ranges)-1

        l = px - x1;  t = py - y1
        r = x2 - px;  b = y2 - py

        inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)

        max_reg = torch.stack([l,t,r,b], dim=1).max(dim=1).values
        for lvl_i, (s_min, s_max) in enumerate(scale_ranges):
            in_level = inside & (max_reg >= s_min) & (max_reg < s_max) \
                       & (level_ids == lvl_i)
            if not in_level.any():
                continue
            cls_t[in_level] = label
            reg_t[in_level] = torch.stack(
                [l[in_level], t[in_level],
                 r[in_level], b[in_level]], dim=1)
            lr = torch.stack([l[in_level], r[in_level]], dim=1)
            tb = torch.stack([t[in_level], b[in_level]], dim=1)
            ctr_t[in_level] = torch.sqrt(
                (lr.min(1).values / lr.max(1).values.clamp(1e-6)) *
                (tb.min(1).values / tb.max(1).values.clamp(1e-6)))

    return cls_t, reg_t, ctr_t


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, num_pos=1.0):
    num_classes = logits.shape[1]
    fg_mask   = targets < num_classes
    one_hot   = torch.zeros_like(logits)
    if fg_mask.any():
        one_hot[fg_mask, targets[fg_mask]] = 1.0

    p         = logits.sigmoid()
    ce         = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    p_t        = p * one_hot + (1-p) * (1-one_hot)
    alpha_t    = alpha * one_hot + (1-alpha) * (1-one_hot)
    loss       = alpha_t * (1-p_t)**gamma * ce

    valid_mask = (targets >= 0)
    return loss[valid_mask].sum() / max(num_pos, 1.0)


def iou_loss(pred, target, eps=1e-6):
    if pred.numel() == 0:
        return pred.sum() * 0
    return generalized_box_iou_loss(pred, target, reduction="mean")


def centerness_loss(pred, target, pos_mask):
    if pos_mask.sum() == 0:
        return pred.sum() * 0
    return F.binary_cross_entropy_with_logits(
        pred[pos_mask], target[pos_mask], reduction="mean")


class FCOS(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = ConvNeXtBackbone()
        in_chs           = self.backbone.out_channels   
        self.fpn         = FPN(in_chs, out_ch=256)
        self.head        = FCOSHead(in_ch=256, num_classes=num_classes)
        self.strides     = cfg.FPN_STRIDES
        self.scale_ranges= cfg.SCALE_RANGES

    def forward(self, images, targets=None):
        x        = torch.stack(images)
        feats    = self.fpn(self.backbone(x))
        cls_outs, reg_outs, ctr_outs = self.head(feats)

        if self.training:
            return self._compute_losses(
                feats, cls_outs, reg_outs, ctr_outs, targets)
        else:
            return self._decode(feats, cls_outs, reg_outs, ctr_outs,
                                x.shape[-2:])

    def _compute_losses(self, feats, cls_outs, reg_outs, ctr_outs, targets):
        device      = feats[0].device
        points_list = get_points(feats, self.strides, device)

        total_cls = torch.tensor(0., device=device)
        total_reg = torch.tensor(0., device=device)
        total_ctr = torch.tensor(0., device=device)
        num_imgs  = len(targets)

        for i in range(num_imgs):
            gt_b = targets[i]["boxes"].to(device)
            gt_l = targets[i]["labels"].to(device)

            cls_t, reg_t, ctr_t = fcos_targets(
                points_list, gt_b, gt_l,
                self.strides, self.scale_ranges, self.num_classes)

            cls_flat = torch.cat([
                c[i].permute(1,2,0).reshape(-1, self.num_classes)
                for c in cls_outs], dim=0)
            reg_flat = torch.cat([
                r[i].permute(1,2,0).reshape(-1, 4)
                for r in reg_outs], dim=0)
            ctr_flat = torch.cat([
                c[i].permute(1,2,0).reshape(-1)
                for c in ctr_outs], dim=0)

            pos_mask = cls_t < self.num_classes
            num_pos  = pos_mask.sum().item()

            total_cls += focal_loss(cls_flat, cls_t,
                                    cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA,
                                    max(num_pos, 1))

            if num_pos > 0:
                all_pts = torch.cat(points_list, dim=0)
                pos_pts = all_pts[pos_mask]
                pred_boxes = torch.stack([
                    pos_pts[:,0] - reg_flat[pos_mask][:,0],
                    pos_pts[:,1] - reg_flat[pos_mask][:,1],
                    pos_pts[:,0] + reg_flat[pos_mask][:,2],
                    pos_pts[:,1] + reg_flat[pos_mask][:,3]], dim=1)
                tgt_boxes = torch.stack([
                    pos_pts[:,0] - reg_t[pos_mask][:,0],
                    pos_pts[:,1] - reg_t[pos_mask][:,1],
                    pos_pts[:,0] + reg_t[pos_mask][:,2],
                    pos_pts[:,1] + reg_t[pos_mask][:,3]], dim=1)

                total_reg += iou_loss(pred_boxes, tgt_boxes)
                total_ctr += centerness_loss(ctr_flat, ctr_t, pos_mask)

        n = max(num_imgs, 1)
        return {
            "cls_loss": total_cls / n,
            "reg_loss": total_reg / n,
            "ctr_loss": total_ctr / n,
        }

    def _decode(self, feats, cls_outs, reg_outs, ctr_outs, img_size):
        device      = feats[0].device
        points_list = get_points(feats, self.strides, device)
        H, W        = img_size
        all_boxes, all_scores, all_labels = [], [], []

        for lvl, (pts, cls_o, reg_o, ctr_o) in enumerate(
                zip(points_list, cls_outs, reg_outs, ctr_outs)):
            cls_p = cls_o[0].permute(1,2,0).reshape(-1, self.num_classes).sigmoid()
            reg_p = reg_o[0].permute(1,2,0).reshape(-1, 4)
            ctr_p = ctr_o[0].permute(1,2,0).reshape(-1).sigmoid()

            scores, labels = (cls_p * ctr_p.unsqueeze(1)).max(dim=1)

            keep = scores >= cfg.SCORE_THRESH
            if not keep.any():
                continue

            scores = scores[keep]; labels = labels[keep]
            pts_k  = pts[keep];    reg_k  = reg_p[keep]

            boxes = torch.stack([
                (pts_k[:,0] - reg_k[:,0]).clamp(0, W),
                (pts_k[:,1] - reg_k[:,1]).clamp(0, H),
                (pts_k[:,0] + reg_k[:,2]).clamp(0, W),
                (pts_k[:,1] + reg_k[:,3]).clamp(0, H)], dim=1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_boxes:
            return [{"boxes": torch.empty(0,4), "scores": torch.empty(0),
                     "labels": torch.empty(0, dtype=torch.int64)}]

        boxes  = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        keep   = nms(boxes, scores, cfg.NMS_THRESH)[:cfg.MAX_DETS]
        return [{"boxes": boxes[keep], "scores": scores[keep],
                 "labels": labels[keep]}]


def build_model(num_classes=cfg.NUM_CLASSES):
    return FCOS(num_classes=num_classes)


def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, loader, device, epoch, scaler=None):
    model.train()
    total = 0.0
    loop  = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for imgs, targets in loop:
        imgs    = [x.to(device) for x in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            losses = sum(model(imgs, targets).values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        total += losses.item()
        loop.set_postfix(loss=f"{losses.item():.3f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, epoch):
    model.train()
    total = 0.0
    loop  = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)
    for imgs, targets in loop:
        imgs    = [x.to(device) for x in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        losses  = sum(model(imgs, targets).values())
        total  += losses.item()
        loop.set_postfix(loss=f"{losses.item():.3f}")
    return total / len(loader)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, save_path=None):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.stop       = False

    def step(self, val_loss, model, epoch):
        improved = val_loss < (self.best_loss - self.min_delta)

        if improved:
            self.best_loss  = val_loss
            self.best_epoch = epoch + 1
            self.counter    = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
                print(f"  Best model saved (val={self.best_loss:.4f})")
        else:
            self.counter += 1
            print(f"   No improvement for {self.counter}/{self.patience} epochs "
                  f"(best={self.best_loss:.4f} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                print(f"\n   Early stopping triggered after {self.patience} "
                      f"epochs without improvement.")
                print(f"  Best val loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
                self.stop = True

        return self.stop

def train():
    print(f"Using device: {cfg.DEVICE}\n")

    train_ds = DrugDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_LBL_BBOX,
                           get_train_transforms())
    val_ds   = DrugDataset(cfg.VAL_IMG_DIR,   cfg.VAL_LBL_BBOX,
                           get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = build_model(cfg.NUM_CLASSES).to(cfg.DEVICE)

    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"  Backbone frozen for first {cfg.FREEZE_EPOCHS} epochs.\n")

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(head_params, lr=cfg.LR,
                                    weight_decay=cfg.WEIGHT_DECAY)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    scaler   = torch.amp.GradScaler('cuda') if cfg.USE_AMP and cfg.DEVICE == "cuda" else None
    amp_info = "ON " if scaler else "OFF"
    print(f"  Mixed precision (AMP): {amp_info}\n")

    history  = {"train": [], "val": []}
    best_val = float("inf")

    early_stopper = EarlyStopping(
        patience  = cfg.EARLY_STOP_PATIENCE,
        min_delta = cfg.EARLY_STOP_MIN_DELTA,
        save_path = os.path.join(cfg.SAVE_DIR, "best_fcos.pth")
    )
    print(f"  Early stopping: patience={cfg.EARLY_STOP_PATIENCE}, "
          f"min_delta={cfg.EARLY_STOP_MIN_DELTA}\n")

    for epoch in range(cfg.EPOCHS):
        if epoch == cfg.FREEZE_EPOCHS:
            print("\n  Unfreezing backbone...")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.backbone.parameters()),
                "lr":     cfg.BACKBONE_LR,
            })

        tl = train_one_epoch(model, optimizer, train_loader, cfg.DEVICE, epoch, scaler)
        vl = evaluate(model, val_loader, cfg.DEVICE, epoch)
        scheduler.step()

        history["train"].append(tl)
        history["val"].append(vl)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:02d}/{cfg.EPOCHS}]  "
              f"Train: {tl:.4f}  Val: {vl:.4f}  LR: {lr:.7f}")

        if early_stopper.step(vl, model, epoch):
            break

        if (epoch+1) % cfg.SAVE_EVERY == 0:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       os.path.join(cfg.SAVE_DIR, f"fcos_epoch_{epoch+1}.pth"))
            print(f"  Checkpoint saved: fcos_epoch_{epoch+1}.pth")

    print(f"\n✔ Training finished. Best val loss: "
          f"{early_stopper.best_loss:.4f} at epoch {early_stopper.best_epoch}")
    _plot(history)
    return model


def _plot(h):
    plt.figure(figsize=(8,5))
    plt.plot(h["train"], label="Train Loss")
    plt.plot(h["val"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("FCOS + ConvNeXt — Loss Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.SAVE_DIR, "fcos_loss_curve.png"))
    plt.show()



@torch.no_grad()
def predict(model, img_path, threshold=0.3, device=cfg.DEVICE):
    model.eval()
    img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    t    = A.Compose([A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
                      A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    inp  = t(image=img)["image"].unsqueeze(0).to(device)
    out  = model(inp)[0]
    keep = out["scores"] >= threshold
    boxes  = out["boxes"][keep].cpu().numpy()
    boxes[:, [0,2]] *= orig_w / cfg.IMG_SIZE
    boxes[:, [1,3]] *= orig_h / cfg.IMG_SIZE
    return {"boxes":  boxes,
            "labels": out["labels"][keep].cpu().numpy(),
            "scores": out["scores"][keep].cpu().numpy()}


def visualize(img_path, preds, save_path=None):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.imshow(img)
    cmap = plt.cm.get_cmap("tab20", cfg.NUM_CLASSES)
    for box, label, score in zip(
            preds["boxes"], preds["labels"], preds["scores"]):
        x1,y1,x2,y2 = box
        c = cmap(label)
        ax.add_patch(patches.Rectangle(
            (x1,y1), x2-x1, y2-y1, lw=2, edgecolor=c, facecolor="none"))
        ax.text(x1, y1-5, f"{cfg.CLASS_NAMES[label]}: {score:.2f}",
                color="white", fontsize=9,
                bbox=dict(facecolor=c, alpha=0.7, pad=2, edgecolor="none"))
    ax.axis("off"); plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()


def load_model(ckpt, num_classes=cfg.NUM_CLASSES, device=cfg.DEVICE):
    model = build_model(num_classes).to(device)
    state = torch.load(ckpt, map_location=device)
    if "model" in state: state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model



if __name__ == "__main__":

    
    preprocess_labels()

 
    verify_labels()

  
    MODE      = "train"   
    IMG_PATH  = None
    CKPT_PATH = None
    THRESHOLD = 0.3

    if MODE == "train":
        train()

    elif MODE == "predict":
        assert IMG_PATH  is not None, "Set IMG_PATH"
        assert CKPT_PATH is not None, "Set CKPT_PATH"
        m = load_model(CKPT_PATH)
        p = predict(m, IMG_PATH, THRESHOLD)
        print(f"Detected {len(p['boxes'])} objects:")
        for box, lbl, sc in zip(p["boxes"], p["labels"], p["scores"]):
            print(f"  {cfg.CLASS_NAMES[lbl]:<20s}  "
                  f"score={sc:.3f}  box={box.astype(int).tolist()}")
        visualize(IMG_PATH, p,
                  save_path="/content/working/fcos_prediction.png")
