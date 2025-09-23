import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToPILImage

import sys 

sys.path.append("/content/Convolutiional_VAE")
from vae_model import ConvolutionnalVAE

# Load your checkpoint
checkpoint = torch.load('/content/vae_checkpoint_epoch_100.pth', map_location="cpu")  # Your 100/300 checkpoint

# Create model instance with same parameters
z_dim = 256  # Use whatever z_dim you trained with
model = ConvolutionnalVAE(z_dim=z_dim, input_size=256)  # Adjust based on your training
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Important for generation!

# Generate faces immediately!
with torch.no_grad():
    # Method 1: Sample from random noise
    z_random = torch.randn(1, z_dim)  # Batch size 1, latent dim
    generated_face = model.decode(z_random)
    
    # Method 2: Sample from learned distribution (better quality)
    z_normal = torch.randn(1, z_dim) * 0.5  # Adjust std for better results
    generated_face_normal = model.decode(z_normal)

to_pil = ToPILImage()
image = to_pil(generated_face.squeeze(0))  # Remove batch dim and convert

# Save with PIL
image.save('generated_face_pil.png')