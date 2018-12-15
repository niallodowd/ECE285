
# coding: utf-8

# In[ ]:


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


# In[ ]:



dim_orig = image_size * image_size
x_train = np.reshape(x_train, [-1, dim_orig])
x_test = np.reshape(x_test, [-1, dim_orig])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[ ]:


x = Input(batch_shape=(batch_size, dim_orig))
h = Dense(dim_interm, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


# In[ ]:


decoder_h = Dense(dim_interm, activation='relu')
decoder_mean = Dense(dim_orig, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# In[ ]:


vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean, name='vae_mlp')

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

