import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size=64,shuffle=True)

dataiter = iter(data_loader)
images,labels = next(dataiter)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12, 3), ##bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12), ##bottleneck
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
print('model before training')
#print(model)
torch.save(model.state_dict(),'./model_params_before.pth') #save model parameters
torch.save(model,'./model_before.pt')


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3,weight_decay=1e-5)

# ###training
num_epochs = 4
outputs = []
for epoch in range(num_epochs):
    for (img,_) in data_loader:

        img = img.reshape(-1,28*28)
        recon = model(img) #reconstructed image (image after passing through autoencoder)

        loss = criterion(recon,img)

        optimizer.zero_grad() #ensures that gradients don't accumulate and take larger steps than intended
        loss.backward() #compute gradients of the loss function using the chain rule
        optimizer.step() #update the parameters of the model in place

    print(f'Epoch:{epoch+1}, Loss:{loss.item()}:.4f')
    outputs.append((epoch,img,recon))


print('model after training')
torch.save(model.state_dict(),'./model_params_after.pth') #save model parameters
torch.save(model,'./model_after.pt')

for k in range(0,num_epochs,4):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()

    for i,item in enumerate(imgs):
        if i>=9: break
        plt.subplot(2,9,i+1)
        item = item.reshape(-1,28,28)
        plt.imshow(item[0])

    for i,item in enumerate(recon):
        if i>= 9: break
        plt.subplot(2,9,9+i+1)
        item = item.reshape(-1,28,28)
        plt.imshow(item[0])