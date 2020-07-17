import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch import optim
from dataset.dcm_dataset import DcmDataset
from unet.unet_model import UNet
from test import test

def validate(net,device,criterion,val_loader):
  loss = 0
  itr = 0
  net.eval()
  with torch.no_grad():
    for batch in val_loader:
      images = batch['dcm']
      masks = batch['mask']
      images = images.to(device=device, dtype=torch.float32)
      mask_type = torch.float32 if net.n_classes == 1 else torch.long
      masks = masks.to(device=device, dtype=mask_type)

      masks_pred = net(images)
      loss_dict = criterion(masks_pred, masks)
      # 统计损失值之和，并梯度下降
      loss += loss_dict.item()
      itr += 1
  return loss/itr


def train(net, device, data_dir,
          epochs=10,
          batch_size=16,
          lr=0.001,
          val_percent=0.2
          ):
    dataset = DcmDataset(data_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.BCELoss()

    train_list = []
    valid_list = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        itr = 0
        for batch in train_loader:
            images = batch['dcm']
            masks = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            masks = masks.to(device=device, dtype=mask_type)

            masks_pred = net(images)
            # 两个输入的数据类型要一样，因为sigmid'是float型，所以masks也要float型
            loss = criterion(masks_pred, masks)
            epoch_loss += loss.item()
            itr += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
        epoch_loss = epoch_loss / itr
        val_loss = validate(net, device, criterion, val_loader)  # validate
        train_list.append(epoch_loss)
        valid_list.append(val_loss)
        print('Epoch:%d, train_loss:%.3f, val_loss:%.3f' % (epoch, epoch_loss, val_loss))

    range_x = range(10, len(train_list))
    plt.plot(range_x, train_list[10:], 'r', label='Training loss')
    plt.plot(range_x, valid_list[10:], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()






if __name__ == '__main__':

    net = UNet(1, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train(net, device, '/content/drive/My Drive/dcm_mini',
          epochs=20,
          batch_size=16,
          lr=0.001,
          val_percent=0.2
    )
    test(net, device, '/content/drive/My Drive/dcm_mini')

