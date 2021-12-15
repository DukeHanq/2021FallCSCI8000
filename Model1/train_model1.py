import os
from PIL import Image
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from Model1_ResNet18 import ResidualBlock,ResNet,ResNet18

#set hyperparameter
EPOCH = 1
pre_epoch = 0
train_BATCH_SIZE = 8
test_BATCH_SIZE = 4
LR = 0.01

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

#windows里面训练的时候一定要加这一行，不然会报错
if __name__ == '__main__':
    #check gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #把resnet18加载到我们指定的设备上，这里是显卡
    net = ResNet18().to(device)
    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    #name format:0_123.png(0 is group)
    dataSet = FlameSet('./Group_Icon')

    trainloader = torch.utils.data.DataLoader(dataSet, batch_size=train_BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataSet, batch_size=test_BATCH_SIZE, shuffle=False, num_workers=2)

    #train
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        #循环从训练集里面取数据，每次加载一个batch
        for i, data in enumerate(trainloader, 0):
            #prepare dataset
            length = len(trainloader)
            #一个data是input和label的结合,取出来之后加载到gpu当中去
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            #获取输出
            outputs = net(inputs)
            #获取损失
            loss = criterion(outputs, labels)
            #利用损失反向传播
            loss.backward()
            #优化
            optimizer.step()
            
            #print ac & loss in each batch
            #求出总损失，然后除以已经训练过的图像，就是平均损失了
            sum_loss += loss.item()
            #找出output当中的最大值，作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            #计数
            total += labels.size(0)
            #检测是否正确，否则不增加correct，correct/total为正确率
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            
        #get the ac with testdataset in each epoch
        #遍历测试集检查正确率
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Test\'s ac is: %.3f%%' % (100 * correct / total))

    print('Train has finished, total epoch is %d' % EPOCH)
    torch.save(net, './resnet18.pkl')
    print('The net has been saved! Check ' + './resnet18.pkl')