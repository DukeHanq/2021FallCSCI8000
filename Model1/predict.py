from torchvision import transforms
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

if __name__ == '__main__':
    image_path = './0_0.png'

    group_dic = {
        '0' : 'Accessorie',
        '1' : 'Armor',
        '2' : 'Food',
        '3' : 'Material',
        '4' : 'Shield',
        '5' : 'Special_Item',
        '6' : 'Supply',
        '7' : 'Weapon'
    }
    #我们是在cuda里面训练的，所以加载时也记得把这个模型转为cuda模式
    net = torch.load('resnet18.pkl').cuda()
    #先利用image接口读取图片，然后转为RGB(.jpg不用转，但是我们这里读取的是png)
    pic = Image.open(image_path).convert('RGB')
    #需要预测的数据同理,也要转为cuda
    pic_tensor = img_transform(pic).unsqueeze(0).cuda()
    #预测输出结果,输出结果转为cpu模式后转numpy
    result_list = net(pic_tensor).cpu().detach().numpy()
    print(group_dic[str(np.argmax(result_list))])