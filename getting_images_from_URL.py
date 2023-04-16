training_data = []

#% matplotlib
#inline
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import requests
import numpy as np
from io import BytesIO
from IPython.display import display, HTML

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_CHANNELS = 3

images = [
    "https://data.heatonresearch.com/images/jupyter/Brookings.jpeg",
    #"https://data.heatonresearch.com/images/jupyter/WashU_Graham_Chapel.jpeg",
    #"https://data.heatonresearch.com/images/jupyter/SeigleHall.jpeg",
"https://i.guim.co.uk/img/media/fe1e34da640c5c56ed16f76ce6f994fa9343d09d/0_174_3408_2046/master/3408.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=0d3f33fb6aa6e0154b7713a00454c83d"

]


def make_square(img):
    cols, rows = img.size

    if rows > cols:
        pad = (rows - cols) / 2
        img = img.crop((pad, 0, cols, cols))
    else:
        pad = (cols - rows) / 2
        img = img.crop((0, pad, rows, rows))

    return img


for url in images:
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    img = make_square(img)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    plt.imshow(np.asarray(img))
    plt.show()
    training_data.append(np.asarray(img))