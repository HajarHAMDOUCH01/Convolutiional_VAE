import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc

import sys 
sys.path.append("/content/Convolutiional_VAE")

from vae_model import ConvolutionnalVAE
from dataset import FacesDataset
from losses import perceptual_loss_cvae, cvae_loss, cvae_total_loss

root = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clear_memory()
    
    z_dim = 256
    batch_size = 32
    lr = 5e-4
    num_epochs = 300
    beta = 0.5
    
    root = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
    transform = get_transforms()
    faces_dataset = FacesDataset(root=root, transform=transform)
    dataloader = DataLoader(faces_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True
                            )
    
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
        num_batchs = 0
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device, non_blocking=True)
            batch_size_actual = real_images.size(0)
            
            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(real_images)            
            loss, kld_loss, bce_loss = cvae_total_loss(recon_imgs, real_images, mu, logvar, beta, mae_weight=1.0, percep_weight=0.5)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.detach().item()
            total_bce += bce_loss.detach().item()
            total_kld += kld_loss.detach().item()
            num_batchs += 1
            
            loop.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "BCE": f"{bce_loss.item():.4f}", 
                "KLD": f"{kld_loss.item():.4f}",
                "Beta": f"{beta:.2f}"
            })

            del recon_imgs, mu, logvar, loss, bce_loss, kld_loss

            if batch_idx % 50 == 0:
                clear_memory()
        
        avg_loss = total_loss / num_batchs
        avg_bce = total_bce / num_batchs
        avg_kld = total_kld / num_batchs
        
        epoch_losses.append(avg_loss)
        epoch_bce_losses.append(avg_bce)
        epoch_kld_losses.append(avg_kld)
        
        print(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}, "
              f"BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}")
        
        if (epoch + 1) % 20 == 0:
          beta = min(beta + 0.1 , 1.0)

        # cleaning memory after each epoch
        clear_memory()
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'/content/drive/MyDrive/vae_checkpoint_epoch_{epoch+1}.pth')
            torch.save(model, f'/content/drive/MyDrive/vae_model_epoch_{epoch+1}.bin')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    del checkpoint
    clear_memory()
    
    print("Training complete! Saving final model...")
    torch.save(model.state_dict(), '/content/drive/MyDrive/vae_final_model.pth')
    torch.save(model, '/content/drive/MyDrive/vae_final_model.bin')

    plot_training_curves(epoch_losses, epoch_bce_losses, epoch_kld_losses)

    return model, epoch_losses

def plot_training_curves(epoch_losses, epoch_bce_losses, epoch_kld_losses):
    
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
    plt.savefig('/content/drive/MyDrive/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    

if __name__=="__main__":
  model , losses = train_vae()
