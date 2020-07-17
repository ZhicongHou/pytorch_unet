
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
from PIL import Image


class DcmDataset(Dataset):
    def __init__(self ,root):
        self.root = root
        self.transforms = self.get_transform()
        self.data_list = []
        '''
        但经过测试，放到内存上要比放磁盘上慢
        为什么放在内存上更慢，不应该是比从磁盘上读取更快才对嘛？为什么更慢呢？
    
        又经过测试，似乎是colab分配的两个GPU的速度不一样，交换GPU训练后，放到内存上要快很多！！
        '''
        dcm_list, mask_list = self.getLits(root)
        for index in range(len(dcm_list)):
            dcm_img = sitk.ReadImage(dcm_list[index])
            dcm = sitk.GetArrayFromImage(dcm_img)[0]
            # 注意：这里用PIL、cv2去读为255，用plt去读为1. 所以如果用PIL、cv2去读的话，要/255
            mask = Image.open(mask_list[index])
            dcm = np.array(dcm)[223:479 ,123:379] # 裁剪关键部分
            mask = np.array(mask)[223:479 ,123:379]
            dcm, mask = self.process(dcm ,mask)
            self.data_list.append({'dcm' :dcm, 'mask' :mask})

    '''
    经过测试，每次取一个数据，都会通过这个函数，即dataset[index]也会经过这个函数
    因此，数据增强可以在该函数写
    '''
    def __getitem__(self, index):
        # print(random.random())
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def getLits(self ,root_dir):
        _dcm_list = []
        _mask_list = []
        for patient in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir ,patient)
            for dir in os.listdir(patient_path):
                dir_path = os.path.join(patient_path ,dir)
                for img in sorted(os.listdir(dir_path)):
                    img_path = os.path.join(dir_path ,img)
                    if(img.endswith('.dcm')):
                        _dcm_list.append(img_path)
                    elif(img.endswith('.png')):
                        _mask_list.append(img_path)
        _dcm_list = sorted(_dcm_list)
        _mask_list = sorted(_mask_list)
        return _dcm_list, _mask_list

    def process(self ,dcm ,mask):
        dcm[dcm <-50 ] =-50
        dcm[dcm >150 ] =150
        dcm = dcm +50  # 因此dcm的取值范围为[0,200/255]
        dcm = dcm.astype(np.uint8)
        mask = mask.astype(np.uint8)
        dcm = self.transforms(dcm)
        mask = self.transforms(mask)
        return dcm ,mask

    def get_transform(self ,train=False):
        transforms = []
        transforms.append(T.ToPILImage())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToTensor())
        return T.Compose(transforms)  # 返回一个函数，但函数的参数和返回值是什么？