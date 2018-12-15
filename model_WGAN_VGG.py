
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg11
import torch.autograd as ag


# In[2]:


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg11_model = vgg11(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg11_model.features.children())[:35])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

feature_extractor = FeatureExtractor()


# In[3]:


class Generator_CNN(nn.Module):
    def __init__(self):
        super(Generator_CNN, self).__init__()
        self.g_conv_n32s1_f = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1_6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n1s1 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = self.relu(self.g_conv_n32s1_f(img))
        out = self.relu(self.g_conv_n32s1_1(out))
        out = self.relu(self.g_conv_n32s1_2(out))
        out = self.relu(self.g_conv_n32s1_3(out))
        out = self.relu(self.g_conv_n32s1_4(out))
        out = self.relu(self.g_conv_n32s1_5(out))
        out = self.relu(self.g_conv_n32s1_6(out))
        out = self.relu(self.g_conv_n1s1(out)) 
        return out
    
generator = Generator_CNN()
print (type(generator))


# In[4]:


class Discriminator_CNN(nn.Module):
#     (nn.Module):
    def __init__(self, input_size=3):
        super(Discriminator_CNN, self).__init__()  # n-f/s +1, # 56*56*3   #32x32x3
        
        self.conv1 = nn.Conv2d(3,64,3,stride=1, groups=1) # 54*54*3    #30*30*3
        self.conv2 = nn.Conv2d(64,64,3,stride=2) # 26*26*64   #14*14*64         
        self.conv3 = nn.Conv2d(64,128,3,stride=1) #24*24*128   #12*12*128
        self.conv4 = nn.Conv2d(128,128,3,stride=2) #11*11*128  #5*5*128
        self.conv5 = nn.Conv2d(128,256,3,stride=1) #5*5*256    #3*3*256
        self.leaky = nn.LeakyReLU()
        self.fc1 = nn.Linear(2304,1024)
        self.fc2 = nn.Linear(1024,1)

        
    def forward(self, img):
        out = self.conv1(img)
        out = self.leaky(out)
        out = self.conv2(out)
        out = self.leaky(out)
        out = self.conv3(out)
        out = self.leaky(out)
        out = self.conv4(out)
        out = self.leaky(out)
        out = self.conv5(out)
        out = self.leaky(out)
        out = out.view(-1,self.num_flat_features(out))
        out = self.fc1(out)
        out = self.leaky(out)
        out = self.fc2(out)
        out = self.leaky(out)
        return out
    
    # Determine the number of features in a batch of tensors
    def num_flat_features ( self , x ):
        size = x.size()[1:]
        return np.prod(size)
    
dis = Discriminator_CNN()
print(dis)
print (dis.parameters)


# In[5]:


generator.cuda()
discriminator.cuda()
feature_extractor.cuda()

criterion_GAN = nn.MSELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5,0.9))

one = torch.Tensor([1])
mone = one * -1
if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()


# total_step = len(train_loader)
total_step = 10
type(generator)


# In[ ]:


criterion_perceptual = nn.L1Loss()
discriminator = Discriminator_CNN(input_size=55)
feature_extractor = FeatureExtractor()

