import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
all_images = mnist_data._load_data()

images = all_images[0]
labels = all_images[1]

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,batch_size = 64, shuffle=True)
for current_data in data_loader:
    current_labels = current_data[1]
    current_image_sets = current_data[0]
    number_of_current_images_in_current_image_set = len(current_image_sets)
    label_of_the_first_image = current_labels[0]
    first_image_in_image_set = current_data[0][0][0]
    plt.imshow(first_image_in_image_set)
    plt.show()

