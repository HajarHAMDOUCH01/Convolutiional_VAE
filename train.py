import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import sys 
sys.path.append("/content/Convolutionnal_VAE")

from vae_model import ConvolutionnalVAE
from dataset import FacesDataset

root = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + beta * KLD, BCE, KLD

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    z_dim = 32
    batch_size = 4
    lr = 1e-3
    num_epochs = 100
    beta = 1.0  
    
    root = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
    transform = get_transforms()
    faces_dataset = FacesDataset(root=root, transform=transform)
    dataloader = DataLoader(faces_dataset, batch_size=batch_size, shuffle=True)
    
    model = ConvolutionnalVAE(image_channels=3, z_dim=z_dim, input_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    epoch_losses = []
    epoch_bce_losses = []
    epoch_kld_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_bce = 0
        total_kld = 0
        num_samples = 0
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(real_images)
            
            loss, bce_loss, kld_loss = vae_loss(recon_imgs, real_images, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss
            total_bce += bce_loss
            total_kld += kld_loss
            num_samples += batch_size_actual
            
            loop.set_postfix({
                "Loss": f"{loss/batch_size_actual:.2f}",
                "BCE": f"{bce_loss/batch_size_actual:.2f}", 
                "KLD": f"{kld_loss/batch_size_actual:.2f}"
            })
        
        avg_loss = total_loss / num_samples
        avg_bce = total_bce / num_samples
        avg_kld = total_kld / num_samples
        
        epoch_losses.append(avg_loss)
        epoch_bce_losses.append(avg_bce)
        epoch_kld_losses.append(avg_kld)
        
        print(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}, "
              f"BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'vae_checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("Training complete! Saving final model...")
    torch.save(model.state_dict(), 'vae_final_model.pth')
    torch.save(model, 'vae_final_model.bin')

    
    # training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epoch_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Total Training Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epoch_bce_losses, label='BCE Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss (BCE)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_kld_losses, label='KLD Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, epoch_losses

if __name__=="__main__":
  train_vae()