
# coding: utf-8

# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg11
import torch.autograd as ag


# In[ ]:


def calc_gradient_penalty(discriminator, real_data, fake_data, lambda_):
    alpha = torch.rand(real_data.shape[2],1)
    alpha = alpha.cuda() if torch.cuda.is_available() else alpha #earya
    interpolates = (alpha * real_data + ((1-alpha) * fake_data))
    if torch.cuda.is_available():      
        interpolates = interpolates.cuda()
    print (type(interpolates))
    interpolates = ag.Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)
    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


# In[ ]:


LEARNING_RATE = 1e-5
LAMBDA_ = 10
d_min = -1024.0
d_max = 3072.0


# In[ ]:


x_ctrain = np.moveaxis(x_train, [0,1,2,3], [0,2,3,1])
x_ctest = np.moveaxis(x_test, [0,1,2,3], [0,2,3,1])
x_ctrain_noisy = np.moveaxis(x_train_noisy, [0,1,2,3], [0,2,3,1])
x_ctest_noisy = np.moveaxis(x_test_noisy, [0,1,2,3], [0,2,3,1])
del(x_train)
del(x_train_noisy)
x_ctest_noisy = ((x_ctest_noisy)*2-1)
x_ctrain_noisy = ((x_ctrain_noisy)*2-1)
x_ctest = ((x_ctest)*2-1)
x_ctrain = ((x_ctrain)*2-1)


for epoch in range(num_epoch):
      
    input_img0 = x_ctrain_noisy 
    target_img0 = x_ctrain 
    input_img1 = ag.Variable(torch.Tensor(input_img0), requires_grad=True)
    input_img2 = torch.Tensor(input_img0)
    input_img3 = torch.Tensor(input_img0).cuda()
    input_img = input_img1.type(torch.Tensor).cuda()
    target_img1 = torch.Tensor(target_img0)
    target_img = target_img1.type(torch.Tensor).cuda()

        # Train D
    D_ITER = 1
    optimizer_D.zero_grad()
    for i in range(D_ITER):
        
        N = howmany2load - (tessize+1)
        for l in range(0, (N+B-1)//B):
            idx =(np.arange(B*l, min(B*(l+1), N)))
            discriminator.zero_grad()
            temp = Variable(target_img[idx,:,:,:],requires_grad=False)
            d_real_decision = discriminator(temp)   ##? batch?? feed x_train
            del(temp)
            d_real_error = -torch.mean(d_real_decision)
            d_real_error.backward(retain_graph=True)

            # Train D on fake
            d_fake_data = generator(input_img[idx,:,:,:]).detach() ## ??? g(X) x_train_noisy ??
            d_fake_decision = discriminator(d_fake_data)
            d_fake_error = torch.mean(d_fake_decision)
            d_fake_error.backward()   ## ?????
    
        
        
        
        N = howmany2load - (tessize+1)
        for l in range(0, (N+B-1)//B):
            idx =(np.arange(B*l, min(B*(l+1), N)))
#             print(idx)
            real_data = target_img[idx,:,:,:]
            fake_data = input_img3[idx,:,:,:]
            lambda_ = 10
            alpha = torch.rand(real_data.shape[2],1)
            alpha = alpha.cuda() if torch.cuda.is_available() else alpha
            interpolates = (alpha * real_data + ((1-alpha) * fake_data))
            if torch.cuda.is_available():
                interpolates = interpolates.cuda()
            interpolates = ag.Variable(interpolates, requires_grad=True)
            disc_interpolates = discriminator(interpolates)
            gradients = ag.grad(outputs=disc_interpolates.float(), inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
            d_error = d_fake_error - d_real_error + gradient_penalty
    
    
        optimizer_D.step()
        
#     print ("Discriminator training complete")
    optimizer_G.zero_grad()
        # Train G
    generator.zero_grad()
    g_fake_data = generator(input_img[idx,:,:,:])     ## noisy img
    dg_fake_decision = discriminator(g_fake_data)
    g_error = -torch.mean(dg_fake_decision)

    fake_data_dup = d_fake_data
    real_data_dup = Variable(target_img[idx,:,:,:], requires_grad=False)
    fake_features = feature_extractor(fake_data_dup)
    real_features = (feature_extractor(real_data_dup))#, requires_grad=False)
    perceptual_error = criterion_perceptual(fake_features, real_features.detach())
    
    
    g_perceptual_error = g_error + (0.1 * perceptual_error)   ## LAMBDA 1
    g_perceptual_error.backward()
    optimizer_G.step()
   
    
        
        


# In[ ]:


torch.save(generator.state_dict(), 'wgan')

