from torchvision import datasets,transforms

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
all_images = mnist_data._load_data()
print(all_images[0][0].shape) #shape of 1 image
print(all_images[0].shape) #shape of the whole image collection

images = all_images[0]
labels = all_images[1]