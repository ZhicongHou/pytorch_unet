from unet.unet_component import *

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.start = DoubleConv(in_channels=n_channels, out_channels=64)
    self.down1 = DownUnit(in_channels=64, out_channels=128)
    self.down2 = DownUnit(in_channels=128, out_channels=256)
    self.down3 = DownUnit(in_channels=256, out_channels=512)
    self.down4 = DownUnit(in_channels=512, out_channels=1024)
    self.up1 = UpUnit(in_channels=1024, out_channels=512)
    self.up2 = UpUnit(in_channels=512, out_channels=256)
    self.up3 = UpUnit(in_channels=256, out_channels=128)
    self.up4 = UpUnit(in_channels=128, out_channels=64)
    self.out = OutUnit(in_channels=64,out_channels=n_classes)
  def forward(self, x):
    x1 = self.start(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x6 = self.up1(x4,x5)
    x7 = self.up2(x3,x6)
    x8 = self.up3(x2,x7)
    x9 = self.up4(x1,x8)
    x_out = self.out(x9)
    return x_out