import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#generate some data
X_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32)) #convert numpy doubles to torch tensors with float type
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape

#model
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

#loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X) #forward pass

    loss = criterion(y_pred,y) #calculate the current loss
    loss.backward() #apply backward propagation

    optimizer.step() #update the weights

    optimizer.zero_grad() #refresh/empty the gradients

    if (epoch+1)%10==0:
        print(f'epoch:{epoch}, loss={loss.item()}')

ytest = model(X).detach() #grad calc attrib is false (we made a copy of the original trained model
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,ytest,'b')
plt.show()









