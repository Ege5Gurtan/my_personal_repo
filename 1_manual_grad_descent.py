import numpy as np

X = np.array([1,2,3,4],dtype=np.float32)

Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

def forward(x):
    return x*w

def loss(y,y_predicted):
    return ((y-y_predicted)**2).mean()

def grad_of_loss_wrt_weight(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

lr = 0.01
for epoch in range(10):
    print('epoch:',{epoch})
    y_pred = forward(X)
    L = loss(Y,y_pred)
    dw = grad_of_loss_wrt_weight(X,Y,y_pred)
    w = w - dw*lr
    print('weight:',{w})
    print('loss:',{L})

