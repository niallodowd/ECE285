
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import torch
from torch import nn, optim
import torch.utils.data
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from zipfile import ZipFile
from io import BytesIO

# Image manipulation.
import PIL.Image
from IPython.display import display


def load_image(filename):
    image = PIL.Image.open(filename) # open colour image
    return (image)

def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.
    # Convert the pixel-values to the range between 0.0 and 1.0
#     image = np.clip(image/255.0, 0.0, 1.0) 
    plt.imshow(image, interpolation='lanczos')
    # Plot using matplotlib.
    plt.imshow(image)
    plt.show()
    
def normalize_images(x):
    ''' 
    This function normalizes input numpy ndarray x to range [-1,1]. 
    This could be done directly by using a linear function.
    param x :  input matrix
    rtype: float64 
    '''
    return 2*x.astype(np.float64)/255. -1

    
num_epoch = 50
howmany2load=2500
tessize = 50
batch_size = 10
B = batch_size
archive = ZipFile("./img_align_celeba.zip", 'r')
x_train=np.empty([howmany2load,32,32,3])
x_train_orig=np.empty([howmany2load,32,32,3])

for i in range(1,howmany2load): 
    image = load_image(filename=BytesIO(archive.read(archive.namelist()[i])))
    image = image.resize((32,32), PIL.Image.ANTIALIAS)
    x_train_orig[i,:,:]=np.array(image)
    


# In[ ]:


x_train = normalize_images(x_train_orig)

x_test=x_train[0:tessize,:,:,:]
x_train=x_train[tessize:-1,:,:,:]
print (x_test.shape)
print (x_train.shape)

noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 



x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)


# In[ ]:


plot_image (x_train[0])
plot_image (x_train_noisy[0])
np.max(x_train[0])

