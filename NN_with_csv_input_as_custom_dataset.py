import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import math

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.Lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.Lin(x)



class MyDataset(Dataset):
    def __init__(self,path):
        xy = np.loadtxt(path,delimiter=";",dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,0])
        self.y = torch.from_numpy(xy[:,1])
        self.n_samples = xy.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples


path = r'C:\Users\Gebruiker\Documents\Book1.csv'

mydata = MyDataset(path)
# print(mydata.x)
# print(mydata.y)
# print(mydata.n_samples)

batch_size = 5

mydataloader = DataLoader(dataset=mydata,batch_size=batch_size,shuffle=True)
# dataiter = iter(mydataloader)
# data = dataiter.next()
# feature,label = data
# print(feature,label)



model = LinearRegression(batch_size,batch_size)
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


num_epochs = 70
total_samples = len(mydata)
n_iterations = math.ceil(total_samples/batch_size)


for epoch in range(num_epochs):
    for i, current_batch in enumerate(mydataloader):
        inputs,labels = current_batch

        y_pred = model(inputs)
        # calculate loss
        L = loss(labels, y_pred)

        # calculate the gradient
        L.backward()  # dL/dw or dw

        ##update the weights
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}')

x_test = torch.tensor([20,30,40,50,60],dtype=torch.float32)
y_test = model(x_test)
print(f'results {x_test} corresponds to {y_test}')
