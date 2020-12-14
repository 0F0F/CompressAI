import argparse
import struct
import sys
import time
import math
import random
import shutil
import sys

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from torch.utils.data import DataLoader
from torchvision import transforms
import compressai
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
        x_yuv = self.rgb2yuv(x)  # shape: [1, 3, w, h]
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


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return 0, code
    #return model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        "YUV",
        inverse_dict(metric_ids)[metric],
        quality,
    )


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def _encode(image, metric, quality, coder, output):
    compressai.set_entropy_coder(coder)
#def _encode(image, quality, output, metric="mse"):
    enc_start = time.time()
    img = load_image(image)
    start = time.time()
    checkpoint_path = "../params/{}/checkpoint.pth.tar".format(quality)
    state_dict = torch.load(checkpoint_path)["state_dict"]
    #net = ScaleHyperprior_YUV.from_state_dict(torch.load(checkpoint_path)).eval()
    net = ScaleHyperprior_YUV(192, 320)
    net.load_state_dict(state_dict)
    net = net.eval()
    net.update()

    #net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    load_time = time.time() - start

    x = img2torch(img)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    x = pad(x, p)

    with torch.no_grad():
        out = net.compress(x)

    shape = out["shape"]
    header = get_header("YUV", metric, quality)

    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w))
        # write shape and number of encoded latents
        write_uints(f, (shape[0], shape[1], len(out["strings"])))
        for s in out["strings"]:
            write_uints(f, (len(s[0]),))
            write_bytes(f, s[0])

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])
    print(
        f"{bpp:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )


def _decode(inputpath, coder, show, output=None):
    compressai.set_entropy_coder(coder)
#def _decode(inputpath, show, output=None):

    dec_start = time.time()
    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        shape = read_uints(f, 2)
        strings = []
        n_strings = read_uints(f, 1)[0]
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])

    print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
    start = time.time()
    #net = ScaleHyperprior_YUV(192,320)
    checkpoint_path = "../params/{}/checkpoint.pth.tar".format(quality)
    #net = ScaleHyperprior_YUV.from_state_dict(torch.load(checkpoint_path)).eval()
    net = ScaleHyperprior_YUV(192, 320).load_state_dict(torch.load(checkpoint_path)).eval()
    #net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    load_time = time.time() - start

    with torch.no_grad():
        out = net.decompress(strings, shape)

    x_hat = crop(out["x_hat"], original_size)
    img = torch2img(x_hat)
    dec_time = time.time() - dec_start
    print(f"Decoded in {dec_time:.2f}s (model loading: {load_time:.2f}s)")

    if show:
        show_image(img)
    if output is not None:
        img.save(output)



def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image to bit-stream")
    parser.add_argument("image", type=str)
    parser.add_argument(
        "-m",
        "--metric",
        choices=["mse"],
        default="mse",
        help="metric trained against (default: %(default)s",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=list(range(1, 4)),
        type=int,
        default=3,
        help="Quality setting (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args(argv)
    if not args.output:
        args.output = Path(Path(args.image).resolve().name).with_suffix(".bin")

    _encode(args.image, args.metric, args.quality, args.coder, args.output)


def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to imager")
    parser.add_argument("input", type=str)
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-o", "--output", help="Output path")
    args = parser.parse_args(argv)
    _decode(args.input, args.coder, args.show, args.output)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[1:2])
    argv = argv[2:]
    torch.set_num_threads(1)  # just to be sure
    if args.command == "encode":
        encode(argv)
    elif args.command == "decode":
        decode(argv)


if __name__ == "__main__":
    main(sys.argv)
