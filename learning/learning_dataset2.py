import matplotlib.pyplot as plt
from torchvision import datasets,transforms

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
all_images = mnist_data._load_data()

images = all_images[0]
labels = all_images[1]

plt.imshow(images[0])
plt.show()