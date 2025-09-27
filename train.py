import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import gc
import os

import sys 
sys.path.append("/content/Convolutiional_VAE")

from vae_model import ConvolutionnalVAE
from dataset import FacesDataset
from losses import perceptual_loss_cvae, cvae_loss, cvae_total_loss, VGG19
from training_config import training_config


checkpoint_path = "/content/drive/MyDrive/vae_checkpoint_epoch_80.pth"
dataset_path = training_config["dataset_path"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device : {device}")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((training_config["image_input_size"], training_config["image_input_size"])),
        transforms.ToTensor(),
    ])

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

model = ConvolutionnalVAE(image_channels=3, z_dim=training_config["z_dim"], input_size=training_config["image_input_size"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

model.train()

def load_model_from_checkpoint(checkpoint_path, model, optimizer):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    if 'beta' in checkpoint:
        beta = checkpoint['beta']
    else:
        beta = 0.8
        # beta = min(training_config["beta"] + (checkpoint['epoch'] // 10) * 0.1, 1.0)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Will resume training from epoch {start_epoch}")
    print(f"Current beta: {beta}")

    return model, optimizer, start_epoch, beta

def train_vae(model, optimizer, dataset_path, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vgg19_model = VGG19()
    vgg19_model.to(device)
    vgg19_model.eval()

    clear_memory()
    
    z_dim = training_config["z_dim"]
    batch_size = training_config["batch_size"]
    lr = training_config["lr"]
    num_epochs = training_config["num_epochs"]
    beta = training_config["beta"]
    
    start_epoch = 0
    epoch_losses = []
    epoch_bce_losses = []
    epoch_kld_losses = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, beta = load_model_from_checkpoint(checkpoint_path, model, optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        if checkpoint_path:
            print(f"Checkpoint file {checkpoint_path} not found. Starting fresh training.")
        print("Starting training from scratch")

    transform = get_transforms()
    faces_dataset = FacesDataset(root=dataset_path, transform=transform)
    dataloader = DataLoader(faces_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True
                            )
    
    for epoch in range(start_epoch, num_epochs):
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

            if epoch+1 <= 10:
                loss, kld_loss, bce_loss = cvae_total_loss(vgg19_model, recon_imgs, real_images, mu, logvar, beta, mae_weight=1.0, percep_weight=0.5, use_percep=False)
            else:
                loss, kld_loss, bce_loss = cvae_total_loss(vgg19_model, recon_imgs, real_images, mu, logvar, beta, mae_weight=1.0, percep_weight=0.5, use_percep=True)
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
        
        if (epoch + 1) % 10 == 0:
            beta = min(beta + 0.1, 1.0)

        # cleaning memory after each epoch
        clear_memory()
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'beta': beta
            }
            torch.save(checkpoint, f'{training_config["save_dir"]}/vae_checkpoint_epoch_{epoch+1}.pth')
            torch.save(model, f'{training_config["save_dir"]}/vae_model_epoch_{epoch+1}.bin')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    clear_memory()
    
    print("Training complete! Saving final model...")
    torch.save(model.state_dict(), f'{training_config["save_dir"]}/vae_final_model.pth')
    torch.save(model, f'{training_config["save_dir"]}/vae_final_model.bin')

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
    plt.savefig(f'{training_config["save_dir"]}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    model, losses = train_vae(model, optimizer, dataset_path, checkpoint_path)