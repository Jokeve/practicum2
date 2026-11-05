"""Untouched baseline U-Net and a separate nnU-Net student template."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


MODEL_CONFIG = {"model": "basic_unet", "model_levels": 3, "convolutions_per_stage": 1, "model_normalization": "none"}
NNUNET_MODEL_CONFIG = {"model": "nnunet_template", "convolutions_per_stage": 2, "model_normalization": "instance"}



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BasicUNet(nn.Module):
    """The small three-level baseline used for the first two runs."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.encoder1 = ConvBlock(4, c)
        self.encoder2 = ConvBlock(c, 2 * c)
        self.bottleneck = ConvBlock(2 * c, 4 * c)
        self.decoder2 = ConvBlock(6 * c, 2 * c)
        self.decoder1 = ConvBlock(3 * c, c)
        self.head = nn.Conv3d(c, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(F.max_pool3d(skip1, 2))
        bottom = self.bottleneck(F.max_pool3d(skip2, 2))
        up2 = F.interpolate(bottom, size=skip2.shape[2:], mode="trilinear", align_corners=False)
        dec2 = self.decoder2(torch.cat([up2, skip2], dim=1))
        up1 = F.interpolate(dec2, size=skip1.shape[2:], mode="trilinear", align_corners=False)
        return self.head(self.decoder1(torch.cat([up1, skip1], dim=1)))


class NNUNetBlock(nn.Module):
    """Step 2: complete the block described on paper pp. 16 and 24."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # STEP 2 — YOUR CODE HERE. The paper uses two convolution blocks at
        # each resolution. Each has a 3 × 3 × 3 convolution, instance
        # normalization, and a leaky ReLU with a negative slope of 0.01.
        # The first convolution changes in_channels to out_channels. Use
        # stride for this convolution so it can halve the image size when
        # the encoder moves to a deeper level. The second keeps the same
        # number of channels and the same image size. See pp. 16 and 24.
        self.layers = nn.Sequential(
            # First convolution block
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), # kernel_size=3 equals 3x3x3
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLu(0.01, inplace=True),

            # Second convolution block
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NNUNet(nn.Module):
    """Student template: use patch size to build a deeper U-Net."""

    def __init__(self, patch_size: tuple[int, int, int], base_channels: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

        # STEP 2 — YOUR CODE HERE. On pp. 18 and 25, the paper makes the
        # network deeper until another downsampling would leave fewer than
        # four voxels along an axis. Work out the number of halvings that
        # this patch allows, and use it instead of the fixed value below.
        self.depth = int(math.floor(math.log2(min(patch_size) / 4)))

        channels = [min(base_channels * 2**level, 320) for level in range(self.depth + 1)]
        # Like BasicUNet, there is an encoder, decoder, and final class head.
        # The lists let the patch size decide how many levels are needed.
        self.encoder = nn.ModuleList()
        for level, out_channels in enumerate(channels):
            in_channels = 4 if level == 0 else channels[level - 1]
            self.encoder.append(NNUNetBlock(in_channels, out_channels, stride=1 if level == 0 else 2))

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(self.depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose3d(channels[level + 1], channels[level], 2, stride=2))
            self.decoder.append(NNUNetBlock(2 * channels[level], channels[level]))
        self.head = nn.Conv3d(channels[0], 4, kernel_size=1)
        # STEP 3 — YOUR CODE HERE. The paper adds a segmentation head at
        # intermediate decoder resolutions to help train deeper layers
        # (pp. 16–17). Add a head for the half-size decoder output here.
        self.aux_head = nn.Conv3d(channels[1], 4, kernel_size=1) if self.depth >= 2 else None

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        # STEP 2 — YOUR CODE HERE. Follow the journey through BasicUNet:
        # the encoder saves features at each image size, and the decoder
        # combines each enlarged feature map with the encoder features from
        # that same size. Here the lists in __init__ contain more levels, so
        # use loops. The paper specifies transposed-convolution upsampling
        # on pp. 16 and 24; BasicUNet shows how to join the skip features.
        # Finish with the four-class head and return one full-size prediction.
        # STEP 3: later, use the half-size decoder features for a second
        # prediction during training. A normal call should still return
        # only the full-size prediction for sliding-window inference.
        
        feats = []
        for block in self.encoder: # Encoder loop 
            feats.append(block(x)) 
        
        x = feats.pop() # Removes and returns the last item (the bottleneck)
        skips = feats[::-1] # Reverses the list to enable decoding
   
        aux = None
        for i, (up, dec, skip) in enumerate(zip(self.up, self.decoder, skips)): # Decoder loop
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:  # only matters for odd patch sizes
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
            if return_aux and self.training and i == self.depth - 2 and self.aux_head is not None:
                # This decoder output has channels[1] channels at half resolution.
                aux = self.aux_head(x)


        out = self.head(x) # Final product
        if return_aux and self.training and aux is not None:
            return out, aux
        return out
        

