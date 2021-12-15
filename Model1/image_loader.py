# 数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

#不管传进来的什么图，直接先切成100*100然后随机旋转
img_transform = transforms.Compose([
    transforms.RandomCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

label_transform = transforms.Compose([
    transforms.ToTensor()
])

#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self,root):
    # 所有图片的绝对路径
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=img_transform
        #噪声图像会被替换成空白占位符,图像出错就返回占位符
        self.pad_data = torch.zeros((3,100,100))
        self.pad_i = torch.as_tensor(int(0))

    def __getitem__(self, index):
        img_path = self.imgs[index]
        #部分图片有可能是4通道的，最典型的例子就是png，改成3通道
        pil_img = Image.open(img_path).convert('RGB')
        #从名字当中分离出label
        i = img_path.split('\\')[1].split('.')[0].split('_')[0]
        i  = torch.as_tensor(int(i))
        if self.transforms:
            try:
                data = self.transforms(pil_img)
            except:
                return self.pad_data,self.pad_i
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data, i

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataSet = FlameSet('./Group_Icon')
    trainloader = torch.utils.data.DataLoader(dataSet, batch_size=2, shuffle=True)
    for i_batch,batch_data in enumerate(trainloader):
        print(i_batch)#打印batch编号
        print(batch_data[0].size())#打印该batch里面图片的大小
        print(batch_data[1][0].size())
