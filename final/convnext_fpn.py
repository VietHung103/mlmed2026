import torch
import torch.nn as nn
import timm

class ConvNeXtBackbone(nn.Module):
    """
    Return feature maps at 4 stages: C2, C3, C4, C5
    """
    def __init__(self, name="convnext_tiny", pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0,1,2,3)  # 4 stages
        )
        self.channels = self.model.feature_info.channels()

    def forward(self, x):
        feats = self.model(x)  # list of 4 tensors
        return feats

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

        # extra P6,P7 from last level
        self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feats):
        # feats: [C2,C3,C4,C5]
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        # top-down
        for i in range(len(lat)-1, 0, -1):
            up = torch.nn.functional.interpolate(lat[i], size=lat[i-1].shape[-2:], mode="nearest")
            lat[i-1] = lat[i-1] + up

        outs = [o(lat_i) for o, lat_i in zip(self.output, lat)]
        p6 = self.p6(outs[-1])
        p7 = self.p7(self.act(p6))
        return outs + [p6, p7]  # P2..P7 (but we'll use P3..P7 typically)
