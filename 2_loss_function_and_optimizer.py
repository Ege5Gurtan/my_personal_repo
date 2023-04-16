import torch
import torch.nn as nn

X = torch.tensor([1,2,3,4],dtype=torch.float32)

Y = torch.tensor([2,4,6,8],dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


def forward(x):
    return x*w

learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w],lr=learning_rate)

for epoch in range(100):
    print('epoch:',{epoch})
    y_pred = forward(X)

    #calculate loss
    L = loss(Y,y_pred)

    #calculate the gradient
    L.backward() #dL/dw or dw

    ##update the weights
    optimizer.step()
    optimizer.zero_grad()

    print('weight:',{w})
    print('loss:',{L})

