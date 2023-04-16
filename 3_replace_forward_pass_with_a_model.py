import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)

Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


n_samples,n_features = X.shape
print(n_samples,n_features) #4 sample, 1 feature (each sample is 1D)
input_size = n_features
output_size = n_features

model= nn.Linear(input_size,output_size)

learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(1000):
    print('epoch:',{epoch})
    y_pred = model(X)

    #calculate loss
    L = loss(Y,y_pred)

    #calculate the gradient
    L.backward() #dL/dw or dw

    ##update the weights
    optimizer.step()
    optimizer.zero_grad()

    print('weight:',{w})
    print('loss:',{L})

x_test = torch.tensor([5],dtype=torch.float32)
y_test = model(x_test)
print('y(5)=',{y_test})