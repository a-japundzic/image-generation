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

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# This was causing me an issue so I commented it out
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder functions
        # Hidden (h)
        self.he_fnc_lin = nn.Linear(784, 400)
        self.he_fnc_relu = nn.ReLU()

        # Output (o)
        self.oe_fnc_lin1 = nn.Linear(400, latent_size)
        self.oe_fnc_lin2 = nn.Linear(400, latent_size)

        # Decoder functions
        # Hidden (h)
        self.hd_fnc_lin = nn.Linear(latent_size, 400)
        self.hd_fnc_relu = nn.ReLU()

        # Output (o)
        self.od_fnc_lin = nn.Linear(400, 784)
        self.od_fnc_sigmoid = nn.Sigmoid()


    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)

        # Single hidden layer with 400 nodes using ReLU activations
        h = self.he_fnc_relu(self.he_fnc_lin(x))

        # Two linear output layers (no activations)
        means = self.oe_fnc_lin1(h)
        log_variances = self.oe_fnc_lin2(h)

        # Output for reparameterization
        return means, log_variances

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        
        # Find the standard (needed for input to reparameterization trick)
        # Assuming that log_variance is log(sigma^2)
        std = torch.exp(0.5 * log_variances)

        # Draw a random sample from the standard normal distribution
        G = torch.randn_like(std)

        # Use formual shown in lecture notes to get Z
        return means + std * G

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        # Single hidden layer with 400 nodes using ReLU activations
        h = self.hd_fnc_relu(self.hd_fnc_lin(z))

        # Output using Sigmoid activation
        x_tilde = self.od_fnc_sigmoid(self.od_fnc_lin(h))

        return x_tilde

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)

        # Encode
        means, log_variances = self.encode(x)

        # Reparameterize
        z = self.reparameterize(means, log_variances)

        # Decode
        reconstructed_x = self.decode(z)

        return reconstructed_x, means, log_variances


def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)

    # Reconstruction loss
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_variances - means**2 - torch.exp(log_variances))

    # VAE loss
    loss = reconstruction_loss + kl_divergence

    return loss, reconstruction_loss


def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    
    # Set model to train mode
    model.train()

    # Track the total loss and reconstruction loss for the epoch
    total_train_loss = 0
    total_train_reconstruction_loss = 0

    # Iterate over the training dataset
    for images, _ in train_loader:
        # Move images to the device
        images = images.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Flatten x to match reconstructed output size (batch_size, 784)
        x = images.view(-1, 784)

        # Forward pass
        reconstructed_x, means, log_variances = model(x)

        # Compute the loss and its gradients (backpropagation)
        loss, reconstruction_loss = vae_loss_function(reconstructed_x, x, means, log_variances)
        loss.backward()

        # Adjust the learning rate
        optimizer.step()

        # Update the total loss and reconstruction loss
        total_train_loss += loss.item()
        total_train_reconstruction_loss += reconstruction_loss.item()

    # Compute the average loss and reconstruction loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    avg_train_reconstruction_loss = total_train_reconstruction_loss / len(train_loader.dataset)

    return avg_train_loss, avg_train_reconstruction_loss

def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    
    # Set model to evaluation mode
    model.eval()

    # Track the total loss and reconstruction loss for the epoch
    total_test_loss = 0
    total_test_reconstruction_loss = 0

    # Disable the gradient computation (optimizing no longer needed)
    with torch.no_grad():
        # Iterate over the test dataset
        for images, _ in test_loader:
            # Move images to the device
            images = images.to(device)

            # Flatten images to match reconstructed output size (batch_size, 784)
            x = images.view(-1, 784)

            # Forward pass
            reconstructed_x, means, log_variances = model(x)

            # Compute the loss (no gradients needed)
            loss, reconstruction_loss = vae_loss_function(reconstructed_x, x, means, log_variances)

            # Update the total loss and reconstruction loss
            total_test_loss += loss.item()
            total_test_reconstruction_loss += reconstruction_loss.item()

    # Compute the average loss and reconstruction loss for the epoch
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    avg_test_reconstruction_loss = total_test_reconstruction_loss / len(test_loader.dataset)

    return avg_test_loss, avg_test_reconstruction_loss

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
