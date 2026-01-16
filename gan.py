from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

# Followed the linked tutorial for the GAN implementation
#  https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        # Define the generator network (format used from PyTorch GAN tutorial)
        self.main = nn.Sequential(
            nn.Linear(latent_size, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.main(z)
        

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Initialize the BCE loss function
bce_loss = nn.BCELoss(reduction='sum')

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    # Set training mode for the generator and discriminator
    generator.train()
    discriminator.train()
    
    total_discriminator_loss = 0
    total_generator_loss = 0

    for i, data in enumerate(train_loader, 0):
        ### Update Discriminator network ###
        ## Train with real images 
        #Set gradients to zero
        discriminator.zero_grad()

        # Format the batch
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)

        # Format the label
        labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real images through discriminator
        real_output = discriminator(real_images).view(-1)

        # Compute the discriminator loss on real images
        # Want discriminator to classify real images as real (so use the real labels)
        real_loss = bce_loss(real_output, labels)

        # Calculate the gradients for the real loss (backpropagation)
        real_loss.backward()

        ## Train with fake images
        # Generate batch random vectgors sampled from the standard normal distribution
        noise = torch.randn(batch_size, latent_size).to(device)

        # Generate fake image batch G
        fake_images = generator(noise)

        # Format the label
        labels.fill_(fake_label)

        # Forward pass fake images through discriminator
        fake_output = discriminator(fake_images.detach()).view(-1)
        # Compute the discriminator loss on fake images
        # Want discriminator to classify fake images as fake (so use the fake labels)
        fake_loss = bce_loss(fake_output, labels)
        # Calculate the gradients for the fake loss (backpropagation)
        fake_loss.backward()

        # Compute the total discriminator loss
        discriminator_loss = real_loss + fake_loss
        total_discriminator_loss += discriminator_loss.item()

        # Update the discriminator
        discriminator_optimizer.step()


        ### Update Generator Network ###
        generator.zero_grad()

        # Use real lables for the generator
        labels.fill_(real_label)

        # Perform a forward pass through the discriminator with the fake images (we just updated the discriminator)
        fake_output = discriminator(fake_images).view(-1)

        # Calculate the generator loss based on this output
        # Want generator to fool the discriminator, so we use the fake output
        # but the real labels
        generator_loss = bce_loss(fake_output, labels)

        # Calculate the gradients for the generator loss (backpropagation)
        generator_loss.backward()

        # Update the total generator loss
        total_generator_loss += generator_loss.item()

        # Update the generator
        generator_optimizer.step()

    # Compute the average generator and discriminator loss for the epoch
    avg_generator_loss = total_generator_loss / len(train_loader.dataset)
    avg_discriminator_loss = total_discriminator_loss / len(train_loader.dataset)

    return avg_generator_loss, avg_discriminator_loss

# Test is the saem as train, but without the gradient computation 
def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)

    # Set evaluation mode for the generator and discriminator
    generator.eval()
    discriminator.eval()
    
    total_generator_loss = 0
    total_discriminator_loss = 0
    
    # Disable the gradient computation (optimizing no longer needed)
    with torch.no_grad():
        # Iterate over the test dataset
        for i, data in enumerate(test_loader, 0):
            # Format the batch
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)

            # Format the labels 
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            # Forward pass real images through discriminator
            real_output = discriminator(real_images).view(-1)
            # Compute the discriminator loss on real images
            real_loss = bce_loss(real_output, labels)
            
            # Generate batch random vectgors sampled from the standard normal distribution
            noise = torch.randn(batch_size, latent_size).to(device)
            # Input into the generator to get fake images
            fake_images = generator(noise)

            # Format the labels for the fake images
            labels.fill_(fake_label)
            # Forward pass fake images through discriminator
            fake_output = discriminator(fake_images).view(-1)
            # Compute the discriminator loss on fake images
            fake_loss = bce_loss(fake_output, labels)

            # Total discriminator loss is the sum of the real and fake losses
            discriminator_loss = real_loss + fake_loss
            total_discriminator_loss += discriminator_loss.item()

            # Format the labels for the generator
            labels.fill_(real_label)    
            # Forward pass fake images through discriminator
            fake_output_for_g = discriminator(fake_images).view(-1)
            # Want generator to fool the discriminator, so we use the fake output
            # but the real labels
            generator_loss = bce_loss(fake_output_for_g, labels)
            # Update the total generator loss
            total_generator_loss += generator_loss.item()
            
    avg_generator_loss = total_generator_loss / len(test_loader.dataset)
    avg_discriminator_loss = total_discriminator_loss / len(test_loader.dataset)

    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
