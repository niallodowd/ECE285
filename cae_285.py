## Convolutional AE trained on CelebA dataset
import PIL
import skimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from keras.models import load_model
from skimage.color import rgb2gray
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

# Reading the dataset of images as an array of pixel values
from skimage.io import imread_collection
seq = imread_collection("img_align_celeba/*.jpg", conserve_memory=True)

# Looking at one of the images
img = seq[7]
imgplot = plt.imshow(img)

# Shape of an image
print('Shape of an image in our dataset: {}'.format(img.shape))


#Loading the dataset
images = np.zeros((50000, 56, 56, 1))
for i in range(50000):
    image = skimage.transform.resize(seq[i], (56, 56, 3))
    images[i,:,:,0]= rgb2gray(image)


# Looking at one of the grayscale image
io.imshow(images[7, :, :, 0])


# Introducing the network
input_img = Input(shape=(56, 56, 1))


# Encoder
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x) 

# Decoder
x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

# Instantiate the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Normalize
x_train = images[:30000]
x_test = images[30000:35000]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 56, 56, 1))
x_test = np.reshape(x_test, (len(x_test), 56, 56, 1))


# Train 
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=50,
                shuffle=True,
                validation_data=(x_test, x_test))

# Save weights
autoencoder.save_weights('cae_weights.h5')

# Predict
decoded_imgs = autoencoder.predict(x_test)


# Display results
n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i].reshape(56,56))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(56, 56))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

