import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.transforms import RGB2YCbCr, YCbCr2RGB# tensor -> tensor

class ScaleHyperprior_YUV(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.rgb2yuv = RGB2YCbCr()
        self.yuv2rgb = YCbCr2RGB()

        _N = N // 2
        _M = M // 2

#   LUMA
        self.g_a_luma = nn.Sequential(
            conv(1, _N),
            GDN(_N),
            conv(_N, _N),
            GDN(_N),
            conv(_N, _N),
            GDN(_N),
            conv(_N, _M),
        )

        self.g_s_luma = nn.Sequential(
            deconv(_M, _N),
            GDN(_N, inverse=True),
            deconv(_N, _N),
            GDN(_N, inverse=True),
            deconv(_N, _N),
            GDN(_N, inverse=True),
            deconv(_N, 1),
        )

#   CHROMA
        self.g_a_chroma = nn.Sequential(
            conv(2, _N),
            GDN(_N),
            conv(_N, _N),
            GDN(_N),
            conv(_N, _N),
            GDN(_N),
            conv(_N, _M),
        )

        self.g_s_chroma = nn.Sequential(
            deconv(_M, _N),
            GDN(_N, inverse=True),
            deconv(_N, _N),
            GDN(_N, inverse=True),
            deconv(_N, _N),
            GDN(_N, inverse=True),
            deconv(_N, 2),
        )

#   HYPERPRIOR -> concat luma and chroma
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )


        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        x_yuv = self.rgb2yuv(x).squeeze(0) # shape: [1, 3, w, h]
        x_luma, x_u, x_v = x_yuv.chunk(3, 1) # y, u, v -> [1, 1, w, h]
        x_chroma = torch.cat((x_u, x_v), dim=1) # uv -> [1, 2, w, h]

        y_luma = self.g_a_luma(x_luma) # [1, M/2, w/16, h/16]
        y_chroma = self.g_a_chroma(x_chroma) # [1, M/2, w/16, h/16]

        y = torch.cat((y_luma, y_chroma), dim=1) # [1, M, w/16, h/16]
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat) # [1, M, w/16, h/16]

        y_hat_luma1, y_hat_luma2, y_hat_u, y_hat_v = y_hat.chunk(4, 1) # [1, M/4, w/16, h/16]
        y_hat_luma = torch.cat((y_hat_luma1, y_hat_luma2), dim=1) # [1, M/2, w/16, h/16]
        y_hat_chroma = torch.cat((y_hat_u, y_hat_v), dim=1) # [1, M/2, w/16, h/16]

        x_hat_luma = self.g_s_luma(y_hat_luma)
        x_hat_chroma = self.g_s_chroma(y_hat_chroma)

        x_hat = torch.cat((x_hat_luma, x_hat_chroma), dim=1)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a_luma.0.weight"].size(0) * 2
        M = state_dict["g_a_luma.6.weight"].size(0) * 2
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def compress(self, x):
        x_yuv = self.rgb2yuv(x).squeeze(0)  # shape: [1, 3, w, h]
        x_luma, x_u, x_v = x_yuv.chunk(3, 1)  # y, u, v -> [1, 1, w, h]
        x_chroma = torch.cat((x_u, x_v), dim=1)  # uv -> [1, 2, w, h]

        y_luma = self.g_a_luma(x_luma)  # [1, M/2, w/16, h/16]
        y_chroma = self.g_a_chroma(x_chroma)  # [1, M/2, w/16, h/16]

        y = torch.cat((y_luma, y_chroma), dim=1)  # [1, M, w/16, h/16]
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)

        y_hat_luma1, y_hat_luma2, y_hat_u, y_hat_v = y_hat.chunk(4, 1) # [1, M/4, w/16, h/16]
        y_hat_luma = torch.cat((y_hat_luma1, y_hat_luma2), dim=1) # [1, M/2, w/16, h/16]
        y_hat_chroma = torch.cat((y_hat_u, y_hat_v), dim=1) # [1, M/2, w/16, h/16]

        x_hat_luma = self.g_s_luma(y_hat_luma)
        x_hat_chroma = self.g_s_chroma(y_hat_chroma)

        x_hat = torch.cat((x_hat_luma, x_hat_chroma), dim=1)
        return {"x_hat": x_hat}