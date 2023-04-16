import torch
import torchvision.datasets
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import math
from torch.optim import Adam
from torch.autograd import Variable
import pathlib




class MyDataset(Dataset):
    def __init__(self,path):
        transformer = transforms.Compose([
            transforms.Resize((150,150)),
            transforms.ToTensor()
        ])
        self.data = torchvision.datasets.ImageFolder(path,transformer)
        self.n_samples = len(self.data)
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return self.n_samples

path = r'C:\Users\Gebruiker\Downloads\archive\seg_train\seg_train'
mydata = MyDataset(path)

batch_size = 256
mydataloader = DataLoader(dataset=mydata.data,batch_size=batch_size,shuffle=True)
dataiter = iter(mydataloader)
data = dataiter.next()
feature,label = data
number_of_classes = 6
print(label)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.Neural_net = nn.Sequential(

            #in 5 3 150 150
            nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1),
            #out 5 12 150 150

            nn.BatchNorm2d(num_features=12),
            # Shape= (256,12,150,150)
            nn.ReLU(),
            # Shape= (256,12,150,150)

            nn.MaxPool2d(kernel_size=2),
            # Reduce the image size be factor 2
            # Shape= (256,12,75,75)

            nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1),
            # Shape= (256,20,75,75)
            nn.ReLU(),
            # Shape= (256,20,75,75)

            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Shape= (256,32,75,75)
            nn.BatchNorm2d(num_features=32),
            # Shape= (256,32,75,75)
            nn.ReLU(),
            # Shape= (256,32,75,75)
        )
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=number_of_classes)


    def forward(self,x):
        output = self.Neural_net(x)
        output = output.view(-1, 32 * 75 * 75)
        output = self.fc(output)
        return output

model = ConvNet()
learning_rate = 0.001

#loss = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),learning_rate,weight_decay=0.0001)

num_epochs = 1
total_samples = len(mydata)
n_iterations = math.ceil(total_samples/batch_size)

for epoch in range(num_epochs):
    for i, current_batch in enumerate(mydataloader):
        inputs, labels = current_batch
        y_pred = model(inputs)

        L = loss( y_pred,labels)
        # # calculate the gradient
        L.backward()  # dL/dw or dw
        #
        # ##update the weights
        optimizer.step()
        optimizer.zero_grad()
        print('training')
        break


path = r'C:\Users\Gebruiker\Downloads\archive\seg_test\seg_test'
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

def prediction(img_path,transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)

    output = model(input)
    index= output.data.numpy().argmax()

    pred = classes[index]
    return pred

import glob
root=pathlib.Path(path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
pred_path=r'C:\Users\Gebruiker\Downloads\archive\seg_pred\seg_pred'
images_path=glob.glob(pred_path+'/*.jpg')
pred_dict={}
for i in images_path:
    pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)
print(pred_dict)


