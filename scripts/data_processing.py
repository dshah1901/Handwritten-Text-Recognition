import torch
from torchvision import transforms, datasets
from PIL import Image, ImageFilter
import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

# Downalods and loads the dataset and divides them into training_loader and test_loader
def get_data(para):
    # MNIST Dataset
    batch_size = 64
    train_dataset = datasets.MNIST(root='mnist_data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='mnist_data/',
                                train=False,
                                transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    if (para == 1):
        return train_loader
    elif (para == 2):
        return test_loader
    elif (para == 3 ):
        return train_dataset
    elif (para == 4):
        return test_dataset
    else: 
        return train_loader, test_loader



#    Title: Converting .png image to MNIST dataset format
#    Author: Naveen Kumar Dasari
#    Date: May 3 '19 
#    Availability: https://stackoverflow.com/questions/35842274/convert-own-image-to-mnists-image

def prepare_image(path: str):
    """
    Converting image to MNIST dataset format
    """

    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    pixels = list(new_image.getdata())  # get pixel values
    pixels_normalized = [(255 - x) * 1.0 / 255.0 for x in pixels]

    # Need adequate shape
    adequate_shape = np.reshape(pixels_normalized, (1, 28, 28))
    output = torch.FloatTensor(adequate_shape).unsqueeze(0)
    return output

