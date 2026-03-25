import torch
import torch.nn as nn


def _snr_to_tensor(snr, batch_size, device):
    if isinstance(snr, torch.Tensor):
        snr_tensor = snr.to(device=device, dtype=torch.float32)
        if snr_tensor.ndim == 0:
            snr_tensor = snr_tensor.repeat(batch_size)
        elif snr_tensor.numel() == 1:
            snr_tensor = snr_tensor.reshape(1).repeat(batch_size)
        return snr_tensor.reshape(batch_size, 1)
    return torch.full((batch_size, 1), float(snr), device=device, dtype=torch.float32)


class ImportancePredictor(nn.Module):
    """Lightweight learnable importance map predictor.

    Inputs:
    - feature: semantic feature tensor [B, C, H, W]
    - image: optional RGB image tensor [B, 3, H0, W0]
    - snr: scalar/tensor channel quality indicator
    Output:
    - importance map in [0, 1] with shape [B, 1, H, W]
    """

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LazyConv2d(hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.image_proj = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + 1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature, image=None, snr=0.0):
        if feature.ndim != 4:
            raise ValueError(f"ImportancePredictor expects BCHW feature, got shape={feature.shape}")

        b, _, h, w = feature.shape
        device = feature.device
        feat = self.feature_proj(feature)

        if image is None:
            img_feat = torch.zeros((b, feat.shape[1], h, w), device=device, dtype=feat.dtype)
        else:
            if image.shape[-2:] != (h, w):
                image = nn.functional.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
            img_feat = self.image_proj(image)

        snr_map = _snr_to_tensor(snr, b, device).to(dtype=feat.dtype).view(b, 1, 1, 1).expand(b, 1, h, w)
        fused = torch.cat([feat, img_feat, snr_map], dim=1)
        return self.fusion(fused)

