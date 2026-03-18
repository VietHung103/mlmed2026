import torch
import math

def generate_anchors(base_size, ratios, scales, device):
    anchors = []
    for r in ratios:
        for s in scales:
            area = (base_size * s) ** 2
            w = math.sqrt(area / r)
            h = w * r
            anchors.append([-w/2, -h/2, w/2, h/2])
    return torch.tensor(anchors, dtype=torch.float32, device=device)  # (A,4) in xyxy offset

def shift_anchors(grid_h, grid_w, stride, base_anchors):
    # centers
    shifts_x = (torch.arange(0, grid_w, device=base_anchors.device) + 0.5) * stride
    shifts_y = (torch.arange(0, grid_h, device=base_anchors.device) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1,4)
    A = base_anchors.shape[0]
    K = shifts.shape[0]
    anchors = base_anchors.reshape(1,A,4) + shifts.reshape(K,1,4)
    return anchors.reshape(K*A, 4)

class AnchorGenerator:
    """
    For levels P3..P7 (5 levels), typical strides: 8,16,32,64,128 when input ~640.
    We'll infer stride from feature map size.
    """
    def __init__(self, sizes, ratios, scales):
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales

    @torch.no_grad()
    def __call__(self, features, image_size):
        # features: list of feature maps (we will pass P3..P7)
        H, W = image_size
        anchors_all = []
        for lvl, feat in enumerate(features):
            _, _, fh, fw = feat.shape
            stride_h = H / fh
            stride_w = W / fw
            stride = (stride_h + stride_w) / 2.0

            base = generate_anchors(self.sizes[lvl], self.ratios, self.scales, feat.device)
            anchors = shift_anchors(fh, fw, stride, base)
            anchors_all.append(anchors)
        return anchors_all  # list of (N_i,4)
