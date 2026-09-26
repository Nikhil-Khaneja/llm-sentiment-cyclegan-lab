"""CycleGAN networks, written from scratch (no pretrained weights anywhere).

Generator: ResNet encoder -> residual blocks -> decoder (Johnson et al. / CycleGAN paper).
Discriminator: 70x70 PatchGAN - classifies overlapping 70x70 patches as real/fake.
"""
import random

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class ResnetGenerator(nn.Module):
    """c7s1-ngf, d(2ngf), d(4ngf), R(4ngf) x n_blocks, u(2ngf), u(ngf), c7s1-3, tanh."""

    def __init__(self, ngf=64, n_blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsample x4
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsample x4
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """C64-C128-C256-C512 (4x4 convs, LeakyReLU 0.2, no norm on the first layer) -> 1-channel map.
    Each output cell sees a 70x70 receptive field of the input."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.InstanceNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ImagePool:
    """History buffer of generated images (Shrivastava et al.): the discriminator sees a mix of
    current and past fakes, which damps generator/discriminator oscillation."""

    def __init__(self, size):
        self.size, self.images = size, []

    def query(self, batch):
        if self.size == 0:
            return batch
        out = []
        for img in batch.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() < 0.5:
                j = random.randrange(self.size)
                out.append(self.images[j].clone())
                self.images[j] = img
            else:
                out.append(img)
        return torch.cat(out)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
