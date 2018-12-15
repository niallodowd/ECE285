
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
d = generator.cpu()((torch.Tensor(x_ctest_noisy)*2-1)*100)
#d = generator.cpu()(torch.Tensor(x_ctest_noisy))
a = d.data.numpy()
a = np.moveaxis(a,[0,1,2,3],[0,3,1,2])
a.shape
generator = generator.cuda()
import skimage
from skimage.color import rgb2gray
n = 10
plt.figure(figsize=(15, 5))
for i in range(1,n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(rgb2gray(x_test[i,:,:,:]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(rgb2gray(x_test_noisy[i,:,:,:]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(3, n, i+2*n)
    plt.imshow(rgb2gray(a[i,:,:,:]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

