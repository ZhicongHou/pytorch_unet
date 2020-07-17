import torch
import torch.nn as nn

class OneConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(OneConv, self).__init__()
    self.one_conv = nn.Sequential(
      # 这样设置可以使得input_size = output_size
      nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU() #inplace=True什么意思
    )
  def forward(self, x):
    return self.one_conv(x)


class DoubleConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(DoubleConv, self).__init__()
    self.double_conv = nn.Sequential(
      OneConv(in_channels,out_channels),
      OneConv(out_channels,out_channels)
    )
  def forward(self, x):
    return self.double_conv(x)

class DownUnit(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(DownUnit, self).__init__()
    self.down_unit = nn.Sequential(
      nn.MaxPool2d(2,2),
      DoubleConv(in_channels,out_channels)
    )
  def forward(self, x):
    return self.down_unit(x)

class UpUnit(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(UpUnit, self).__init__()
    self.tran_conv = nn.ConvTranspose2d(in_channels,in_channels//2, kernel_size=2, stride=2)
    self.double_conv = DoubleConv(in_channels,out_channels)
  def forward(self, lx, rx):
    rx = self.tran_conv(rx)
    x = torch.cat([lx,rx],dim=1)
    return self.double_conv(x)

class OutUnit(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(OutUnit, self).__init__()
    self.out_unit = nn.Sequential(
      nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
      nn.BatchNorm2d(out_channels),
      nn.Sigmoid()
    )
  def forward(self,x):
    return self.out_unit(x)


