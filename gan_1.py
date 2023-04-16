import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# input_dim = 25
# output_dim =15


def get_generator_block(input_dim,output_dim):
    gen_block = nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
    )
    return gen_block

def calculate_gen_loss(noise,gen_obj,disc_obj,criterion_obj):
    gen_fake = gen_obj(noise)
    disc_fake = disc_obj(gen_fake)
    gen_loss = criterion_obj(disc_fake,torch.ones_like(disc_fake))
    return gen_loss

def get_discriminator_block(input_dim,output_dim):
    disc_block = nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.LeakyReLU(0.2)
    )
    return disc_block

def calculate_disc_loss(noise,gen_obj,disc_obj,criterion_obj,real_im_batch):
    gen_fake = gen_obj(noise).detach()
    disc_fake = disc_obj(gen_fake)
    disc_real = disc_obj(real_im_batch)
    lossD_real = criterion_obj(disc_real,torch.ones_like(disc_real))
    lossD_fake = criterion_obj(disc_fake,torch.zeros_like(disc_real))
    disc_loss = (lossD_real+lossD_fake)/2
    return disc_loss

class Generator(nn.Module):

    def __init__(self,z_dim=10,im_dim=784,hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim,hidden_dim),
            get_generator_block(hidden_dim,hidden_dim*2),
            get_generator_block(hidden_dim*2,hidden_dim*4),
            get_generator_block(hidden_dim*4,hidden_dim*8),
            nn.Linear(hidden_dim*8,im_dim),
            nn.Sigmoid()
        )
    def forward(self,noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen

class Discriminator(nn.Module):
    def __init__(self,im_dim=784,hidden_dim=128):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim,hidden_dim*4),
            get_discriminator_block(hidden_dim * 4,hidden_dim*2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim,1)
        )
    def forward(self,image):
        return self.disc(image)
    def get_disc(self):
        return self.disc




n_samples = 1000
z_dim =64
noise = torch.randn(n_samples,z_dim,device='cpu')
gen = Generator(z_dim).to('cpu')
disc = Discriminator().to('cpu')

dataloader = DataLoader(
    MNIST('.',download=False,transform=transforms.ToTensor()),
    batch_size= 128,
    shuffle=True
)

gen_loss = calculate_gen_loss(noise,gen,disc,nn.BCEWithLogitsLoss())

n_images = n_samples
real = torch.ones(512,784)
disc_loss = calculate_disc_loss(noise,gen,disc,nn.BCEWithLogitsLoss(),real)

print(
    'done'
)



