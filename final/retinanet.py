import torch
import torch.nn as nn
import torchvision
from .convnext_fpn import ConvNeXtBackbone, FPN
from .anchors import AnchorGenerator
from .utils import box_iou, encode_boxes, decode_boxes
from .losses import FocalLoss, SmoothL1

class RetinaHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=12, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def make_tower(out_ch):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(in_channels, out_ch, 3, padding=1)]
            return nn.Sequential(*layers)

        self.cls_tower = make_tower(num_anchors * num_classes)
        self.box_tower = make_tower(num_anchors * 4)

        prior_prob = 0.01
        bias = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_tower[-1].bias, bias)

    def forward(self, feats):
        cls_outs, box_outs = [], []
        for f in feats:
            cls = self.cls_tower(f)
            box = self.box_tower(f)
            cls_outs.append(cls)
            box_outs.append(box)
        return cls_outs, box_outs

class RetinaNet(nn.Module):
    def __init__(self, num_classes=12, backbone_name="convnext_tiny", pretrained=True,
                 fpn_out_channels=256, anchor_sizes=(32,64,128,256,512),
                 anchor_ratios=(0.5,1.0,2.0), anchor_scales=(1.0, 2**(1/3), 2**(2/3))):
        super().__init__()
        self.backbone = ConvNeXtBackbone(backbone_name, pretrained=pretrained)
        self.fpn = FPN(self.backbone.channels, out_channels=fpn_out_channels)

        self.num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.head = RetinaHead(fpn_out_channels, num_classes, self.num_anchors)

        self.anchor_gen = AnchorGenerator(anchor_sizes, anchor_ratios, anchor_scales)
        self.focal = FocalLoss()
        self.smoothl1 = SmoothL1()

        self.num_classes = num_classes

    def forward(self, images, targets=None):
        # images: list[tensor C,H,W]
        device = images[0].device
        batch = torch.stack(images, dim=0)
        _, _, H, W = batch.shape

        feats = self.backbone(batch)          # C2..C5
        pyramid = self.fpn(feats)             # P2..P7
        pyramid = pyramid[1:]                 # 5 levels

        cls_outs, box_outs = self.head(pyramid)

        # flatten per level
        cls_flat = []
        box_flat = []
        anchors = self.anchor_gen(pyramid, (H, W))

        for lvl in range(len(pyramid)):
            cls = cls_outs[lvl]  # (B, A*C, h, w)
            box = box_outs[lvl]  # (B, A*4, h, w)
            B, _, h, w = cls.shape

            cls = cls.permute(0,2,3,1).contiguous().view(B, -1, self.num_classes)
            box = box.permute(0,2,3,1).contiguous().view(B, -1, 4)

            cls_flat.append(cls)
            box_flat.append(box)

        cls_flat = torch.cat(cls_flat, dim=1)  # (B, N, C)
        box_flat = torch.cat(box_flat, dim=1)  # (B, N, 4)
        anchors_flat = torch.cat(anchors, dim=0).to(device)  # (N,4)

        if targets is None:
            return self.infer(cls_flat, box_flat, anchors_flat, (H, W))

        losses = self.compute_losses(cls_flat, box_flat, anchors_flat, targets)
        return losses

    def compute_losses(self, cls_logits, box_deltas, anchors, targets):
        device = cls_logits.device
        B, N, C = cls_logits.shape

        cls_targets = torch.zeros((B, N, C), device=device, dtype=torch.float32)
        box_targets = torch.zeros((B, N, 4), device=device, dtype=torch.float32)
        box_masks = torch.zeros((B, N), device=device, dtype=torch.bool)

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)

            if gt_boxes.numel() == 0:
                continue

            ious = box_iou(anchors, gt_boxes)             # (N, M)
            iou_max, idx = ious.max(dim=1)

            pos = iou_max >= 0.5
            neg = iou_max < 0.4
            ignore = (~pos) & (~neg)

            assigned = idx[pos]
            labels_pos = gt_labels[assigned]  # should be 0..C-1

            cls_targets[b, pos, :] = 0.0
            cls_targets[b, pos, labels_pos] = 1.0

            box_targets[b, pos, :] = encode_boxes(anchors[pos], gt_boxes[assigned])
            box_masks[b, pos] = True

            # For ignore anchors, we set them to -1 and mask out from cls loss
            cls_targets[b, ignore, :] = -1.0

        # focal loss with ignore handling
        cls_mask = cls_targets >= 0
        cls_loss = self.focal(cls_logits[cls_mask].view(-1, C), cls_targets[cls_mask].view(-1, C))

        if box_masks.any():
            box_loss = self.smoothl1(box_deltas[box_masks], box_targets[box_masks])
        else:
            box_loss = torch.tensor(0.0, device=device)

        return {"loss_cls": cls_loss, "loss_box": box_loss, "loss_total": cls_loss + box_loss}

    @torch.no_grad()
    def infer(self, cls_logits, box_deltas, anchors, image_size,
              score_thresh=0.05, nms_thresh=0.5, max_det=300):
        B, N, C = cls_logits.shape
        out = []
        scores_all = torch.sigmoid(cls_logits)

        for b in range(B):
            scores = scores_all[b]  # (N,C)
            deltas = box_deltas[b]  # (N,4)
            boxes = decode_boxes(anchors, deltas)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size[1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size[0])

            final_boxes, final_scores, final_labels = [], [], []
            for c in range(C):
                sc = scores[:, c]
                keep = sc > score_thresh
                if keep.sum() == 0:
                    continue
                bxs = boxes[keep]
                sc2 = sc[keep]
                keep_idx = torchvision.ops.nms(bxs, sc2, nms_thresh)
                bxs = bxs[keep_idx]
                sc2 = sc2[keep_idx]
                lbl = torch.full((len(keep_idx),), c, device=boxes.device, dtype=torch.int64)

                final_boxes.append(bxs)
                final_scores.append(sc2)
                final_labels.append(lbl)

            if len(final_boxes) == 0:
                out.append({"boxes": torch.zeros((0,4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)})
                continue

            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            final_labels = torch.cat(final_labels, dim=0)

            # top-k
            if final_scores.numel() > max_det:
                topk = torch.topk(final_scores, k=max_det)
                idx = topk.indices
                final_boxes = final_boxes[idx]
                final_scores = final_scores[idx]
                final_labels = final_labels[idx]

            out.append({"boxes": final_boxes, "scores": final_scores, "labels": final_labels})
        return out
