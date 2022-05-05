from typing import Optional, List
import torch
from torch import nn


class PixelEncoder(nn.Module):

    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, obs_dim, activation='ReLU'):
        """
            The Pixel encoder takes input of dimension [3. 64, 64]. This ensures 
            that  the output of the final convolution will be of shape [256, 2, 2], which 
            flattens to a width dimension of 1024. We can also account for images
            of different dimensions, and then use a linea layer to embed these features
            into the proper dimension.

            If a different embedding size is chosen, then we can simply add a final
            linear layer which converts the resulting 1024 tensor to the desired dimension.
        """

        super().__init__()

        self.embedding_size = embedding_size
        self._obs_dim = obs_dim
        kernel = 4
        stride = 2

        self.conv1 = nn.Conv2d(self._obs_dim[0], 32, kernel, stride=stride)
        self.conv2 = nn.Conv2d(32, 64, kernel, stride=stride)
        self.conv3 = nn.Conv2d(64, 128, kernel, stride=stride)
        self.conv4 = nn.Conv2d(128, 256, kernel, stride=stride)
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4]

        w, h = self._obs_dim[1], self._obs_dim[2] 
        for _ in range(4):
            w = ((w - (kernel-1) - 1) // stride) + 1
            h = ((h - (kernel-1) - 1) // stride) + 1
        self.flatten_width = 256 * w * h

        self.fc = (
            nn.Identity() if embedding_size == self.flatten_width 
            else nn.Linear(self.flatten_width, embedding_size)
        )

        self.act_fn = getattr(nn, activation)()

    def forward(self, observation):
        x = observation
        for conv in self.conv_layers:
            x = self.act_fn(conv(x))
        x = x.view(-1, self.flatten_width)
        return self.fc(x)  

class PixelObservationModel(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size, activation='ReLU'):
        """
            The observation model acts as a decoder network to predict pixel level values from the 
            current model values. This makes it slightly different from a normal AutoEncoder or VAE
            as it does not directly take in values that come from the encoder model. Rather, the encoded
            values are indirrectly passed in through the latent state vectors and the belief values (hidden state)
            of out recurrent neural network. 
        """
        super().__init__()
        # inital embedding layer.
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.embedding_size = embedding_size

        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.conv_layers = [self.conv1, self.conv2, self.conv3]

        self.act_fn = getattr(nn, activation)()

    def forward(self, belief, state):
        # Linear embedding layer -> no activation function.
        x = self.fc1(torch.cat([belief, state], dim=1)) 
        x = x.view(-1, self.embedding_size, 1, 1)

        for conv in self.conv_layers:
            x = self.act_fn(conv(x))

        return self.conv4(x)
