Created a variational (VAE) and a generative adversarials network (GAN) to generate images from the MNIST dataset using PyTorch.


A variational autoencoder is a model that learns a probabilistic representation of data. In simple terms, it learns how to map data to a latent distribution constrained to be close to a normal distribution, and then how to map samples from that distribution back to the original data. We can thus generate images by sampling new points from the normal distribution. 

A generative adversarial network (GAN) is a model that learns to generate images through competition between two networks. In simple terms, one network (the generator) learns to create fake images, while another network (the discriminator) learns to distinguish real images from fake images. Through this adversarial process, the generator improves until it can produce images that closely resemble (ideally indistinguishable from) the original dataset.

GAN image generation example:


<img width="242" height="242" alt="image" src="https://github.com/user-attachments/assets/a692e79f-d865-4efa-bad7-700f2e38b935" />

