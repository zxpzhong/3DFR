# full assembly of the sub-parts to form the complete net
import torch.nn.functional as F
from .unet_parts import *
# from .mobilenet2 import *
def debug(str):
    if False:
        print(str)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)

        self.up0 = up(1024, 256, bilinear=False)
        self.up1 = up(512, 128,bilinear = False)
        self.up2 = up(256, 64,bilinear = False)
        self.up3 = up(128, 32,bilinear = False)
        self.up4 = up(64, 16,bilinear = False)
        self.outc1 = outconv(16, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        debug('x1 shape is {}'.format(x1.shape))
        x2 = self.down1(x1)
        debug('x2 shape is {}'.format(x2.shape))
        x3 = self.down2(x2)
        debug('x3 shape is {}'.format(x3.shape))
        x4 = self.down3(x3)
        debug('x4 shape is {}'.format(x4.shape))
        x5 = self.down4(x4)
        debug('x5 shape is {}'.format(x5.shape))
        x6 = self.down5(x5)
        debug('x6 shape is {}'.format(x6.shape))

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc1(x)
        return x