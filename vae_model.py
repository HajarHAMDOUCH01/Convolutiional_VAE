import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm

class Unflatten(nn.Module):
    def __init__(self, channels=256, height=16, width=16):
        super(Unflatten, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
    
    def forward(self, input):
        return input.view(input.size(0), self.channels, self.height, self.width)

class ConvolutionnalVAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=32, input_size=128):
        super(ConvolutionnalVAE, self).__init__()
        self.h_dim = 256 * 16 * 16  # 65536
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128->64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space layers
        self.l_mu = nn.Linear(self.h_dim, z_dim)
        self.l_logvar = nn.Linear(self.h_dim, z_dim)  
        
        self.dec_projection = nn.Linear(z_dim, self.h_dim)  
        
        # Decoder
        self.decoder = nn.Sequential(
            Unflatten(channels=256, height=16, width=16),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64->128
        )
    
    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        epsilon = torch.randn_like(mu)  
        z = mu + std * epsilon
        return z
    
    def bottleneck(self, h):
        mu = self.l_mu(h)
        log_var = self.l_logvar(h)  
        z = self.reparametrize(mu, log_var)
        return z, mu, log_var
    
    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var
    
    def decode(self, z):
        z = self.dec_projection(z)  
        z = self.decoder(z)
        z = torch.clamp(z, min=-10, max=10)
        z = torch.sigmoid(z)
        return z
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, mu, logvar