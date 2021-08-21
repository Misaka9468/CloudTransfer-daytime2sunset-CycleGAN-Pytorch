import torch.nn as nn
import torch.nn.functional as F


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        # use reflectionPad2d instead of Conv2d(padding=1)
        # use InstanceNorm2d instead of BatchNorm2d
        # the value of padding are determined by the size of kernel to make sure H_in == H_out
        ConvBlock = [nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channel, in_channel, 3),
                     nn.InstanceNorm2d(in_channel),
                     nn.ReLU(inplace=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channel, in_channel, 3),
                     nn.InstanceNorm2d(in_channel)]

        self.conv_block = nn.Sequential(*ConvBlock)

    def forward(self, x):
        return x + self.conv_block(x)


# Generator
class Generator(nn.Module):
    def __init__(self, in_channel, out_channel, num_resblock=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channel, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]
        # Encode: image -> feature
        # use Conv2d to extract the features of image
        # channel: 64->128->256
        # size: k -> k/2 -> k/4
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True)]

        model += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(inplace=True)]

        # Transform: add ResBlocks
        for _ in range(num_resblock):
            model += [ResBlock(in_channel=256)]
        # Decode: feature -> image
        # use ConvTranspose2d
        # channels: 256->128->64
        # size: k/4 -> k/2 -> k
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(inplace=True)]
        # Output
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, out_channel, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()
        # 4 Conv
        # channels: in_channel->64->128->256->512
        # size: k-> k/2 -> k/4 -> k/8 -> k/16
        model = [nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, kernel_size=4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # Classification
        # k/16 * k/16 size with 1 channel
        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # input (batch,channel,H,W)
        # kernel size is H*W
        # use view to flatten. to (batchsize,1)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
