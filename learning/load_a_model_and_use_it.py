import torch
from torchvision import datasets,transforms
import torch.nn as nn
import matplotlib.pyplot as plt

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

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size=64,shuffle=True)

dataiter = iter(data_loader)
images,labels = next(dataiter)
image1 = images[0][0]

model = torch.load('./model_after.pt')
model.eval()

img = image1.reshape(-1,28*28)
recon = model(img)
recon = recon.detach().numpy()
recon = recon[0]
recon = recon.reshape(-1, 28, 28)
plt.figure()
plt.title('reconstructed')
plt.imshow(recon[0])

plt.figure()
plt.title('original')
plt.imshow(image1)

plt.show()
