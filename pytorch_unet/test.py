import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dcm_dataset import DcmDataset


def test(net,device,data_dir,
      batch_size = 16
  ):

  dataset = DcmDataset(data_dir)[54:]
  test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

  batch = next(iter(test_loader))
  images = batch['dcm']
  masks = batch['mask']
  images.to(device,dtype=torch.float32)
  masks.to(device,dtype=torch.float32)

  for index in range(images.shape[0]):
    plt.figure(figsize=(6,6),dpi=80)
    plt.subplot(131)
    plt.imshow(images[index,0,:,:].cpu(),cmap='gray')
    plt.subplot(132)
    plt.imshow(masks[index,0,:,:].cpu(),cmap='gray')

    net.eval()
    with torch.no_grad():
      plt.subplot(133)
      prediction = net(images.to(device,dtype=torch.float32))
      prediction = (prediction > 0.5).float()
      plt.imshow(prediction[index,0,:,:].cpu(),cmap='gray')
    plt.show()


