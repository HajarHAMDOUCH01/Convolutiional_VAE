import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision import transforms

import sys 

sys.path.append("/content/Convolutiional_VAE")
from vae_model import ConvolutionnalVAE

checkpoint = torch.load('/content/vae_checkpoint_epoch_100.pth', map_location="cpu")  # Your 100/300 checkpoint

# Create model instance with same parameters
z_dim = 256  # Use whatever z_dim you trained with
model = ConvolutionnalVAE(z_dim=z_dim, input_size=256)  # Adjust based on your training
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  
image_face1 = 
image_face2 = 

def generate_new_face(z_dim, model):
    with torch.no_grad():

        z_random = torch.randn(1, z_dim)  
        generated_face = model.decode(z_random)
        
        z_normal = torch.randn(1, z_dim) * 0.5  
        generated_face_normal = model.decode(z_normal)

    to_pil = ToPILImage()
    image = to_pil(generated_face.squeeze(0))  

    # Save with PIL
    image.save('generated_face_pil.png')

def interpolate_2_faces(model, face1, face2, alpha=0.5):
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((256*256)),
            transforms.ToTensor()
        ])
        face1_tensor = transform(face1)
        face2_tensor = transform(face2)
        z1 = model.encode(face1_tensor)
        z2 = model.encode(face2_tensor)

        z_seed = z1 * alpha + z2 * (1-alpha)

        mixed_face = model.decode(z_seed)

    to_pil = ToPILImage()
    image = to_pil(mixed_face.squeeze(0))  

    # Save with PIL
    image.save('mixed_face.png')

if __name__=="__main__":
    interpolate_2_faces(model, image_face1, image_face2, alpha=0.5)
