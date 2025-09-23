import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image

class VAEEvaluator:
    """
    Comprehensive evaluation metrics for VAE models
    Baby step enhancement - easy to implement, big impact!
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        # Load pretrained Inception for FID calculation
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        
    def reconstruction_metrics(self, real_images, num_samples=100):
        """
        Calculate reconstruction quality metrics
        """
        print(f"📊 Calculating reconstruction metrics on {num_samples} samples...")
        
        # Select random subset
        indices = torch.randperm(len(real_images))[:num_samples]
        sample_images = real_images[indices].to(self.device)
        
        metrics = {}
        mse_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for i in range(0, len(sample_images), 8):  # Process in small batches
                batch = sample_images[i:i+8]
                recon_batch, _, _ = self.model(batch)
                
                # MSE (Mean Squared Error)
                mse = F.mse_loss(recon_batch, batch, reduction='none').mean(dim=[1,2,3])
                mse_scores.extend(mse.cpu().numpy())
                
                # Simple SSIM approximation (you can use proper SSIM library)
                ssim_batch = self.calculate_simple_ssim(batch, recon_batch)
                ssim_scores.extend(ssim_batch)
        
        metrics['reconstruction_mse'] = {
            'mean': np.mean(mse_scores),
            'std': np.std(mse_scores),
            'min': np.min(mse_scores),
            'max': np.max(mse_scores)
        }
        
        metrics['reconstruction_ssim'] = {
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores)
        }
        
        return metrics
    
    def calculate_simple_ssim(self, img1, img2, window_size=11):
        """
        Simplified SSIM calculation
        """
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean(dim=[1,2,3]).cpu().numpy()
    
    def latent_space_metrics(self, real_images, num_samples=200):
        """
        Analyze latent space properties
        """
        print(f"🧠 Analyzing latent space properties on {num_samples} samples...")
        
        indices = torch.randperm(len(real_images))[:num_samples]
        sample_images = real_images[indices].to(self.device)
        
        latent_vectors = []
        mu_vectors = []
        logvar_vectors = []
        
        with torch.no_grad():
            for i in range(0, len(sample_images), 16):
                batch = sample_images[i:i+16]
                z, mu, logvar = self.model.encode(batch)
                
                latent_vectors.append(z.cpu())
                mu_vectors.append(mu.cpu())
                logvar_vectors.append(logvar.cpu())
        
        # Concatenate all vectors
        all_z = torch.cat(latent_vectors, dim=0)
        all_mu = torch.cat(mu_vectors, dim=0)
        all_logvar = torch.cat(logvar_vectors, dim=0)
        
        metrics = {}
        
        # Latent space statistics
        metrics['latent_stats'] = {
            'mean_norm': torch.norm(all_z, dim=1).mean().item(),
            'std_norm': torch.norm(all_z, dim=1).std().item(),
            'mean_activation': all_z.mean().item(),
            'std_activation': all_z.std().item()
        }
        
        # Posterior statistics
        metrics['posterior_stats'] = {
            'mu_mean': all_mu.mean().item(),
            'mu_std': all_mu.std().item(),
            'logvar_mean': all_logvar.mean().item(),
            'logvar_std': all_logvar.std().item()
        }
        
        # Latent space diversity (average pairwise distance)
        distances = torch.pdist(all_z).mean().item()
        metrics['latent_diversity'] = distances
        
        return metrics, all_z
    
    def generation_quality_metrics(self, num_generated=100, z_dim=256):
        """
        Evaluate generation quality
        """
        print(f"🎨 Evaluating generation quality with {num_generated} samples...")
        
        generated_images = []
        
        with torch.no_grad():
            for i in range(0, num_generated, 16):
                batch_size = min(16, num_generated - i)
                z = torch.randn(batch_size, z_dim, device=self.device)
                gen_batch = self.model.decode(z)
                generated_images.append(gen_batch.cpu())
        
        all_generated = torch.cat(generated_images, dim=0)
        
        metrics = {}
        
        # Pixel statistics
        metrics['pixel_stats'] = {
            'mean_pixel_value': all_generated.mean().item(),
            'std_pixel_value': all_generated.std().item(),
            'min_pixel_value': all_generated.min().item(),
            'max_pixel_value': all_generated.max().item()
        }
        
        # Color channel analysis
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = all_generated[:, i]
            metrics[f'{channel}_channel'] = {
                'mean': channel_data.mean().item(),
                'std': channel_data.std().item()
            }
        
        return metrics, all_generated
    
    def interpolation_smoothness(self, num_pairs=10, num_steps=20, z_dim=256):
        """
        Measure interpolation smoothness
        """
        print(f"🌈 Testing interpolation smoothness with {num_pairs} pairs...")
        
        smoothness_scores = []
        
        with torch.no_grad():
            for _ in range(num_pairs):
                # Sample two random points
                z1 = torch.randn(1, z_dim, device=self.device)
                z2 = torch.randn(1, z_dim, device=self.device)
                
                # Generate interpolation
                alphas = torch.linspace(0, 1, num_steps, device=self.device)
                interpolated_images = []
                
                for alpha in alphas:
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    img = self.model.decode(z_interp)
                    interpolated_images.append(img)
                
                # Calculate smoothness (average pixel difference between consecutive frames)
                smoothness = 0
                for i in range(len(interpolated_images) - 1):
                    diff = F.mse_loss(interpolated_images[i], interpolated_images[i+1])
                    smoothness += diff.item()
                
                smoothness_scores.append(smoothness / (num_steps - 1))
        
        return {
            'interpolation_smoothness': {
                'mean': np.mean(smoothness_scores),
                'std': np.std(smoothness_scores)
            }
        }
    
    def comprehensive_evaluation(self, real_images):
        """
        Run all evaluation metrics
        """
        print("🚀 Starting comprehensive VAE evaluation...")
        print("=" * 60)
        
        all_metrics = {}
        
        # 1. Reconstruction metrics
        all_metrics.update(self.reconstruction_metrics(real_images))
        
        # 2. Latent space analysis
        latent_metrics, latent_vectors = self.latent_space_metrics(real_images)
        all_metrics.update(latent_metrics)
        
        # 3. Generation quality
        gen_metrics, generated_images = self.generation_quality_metrics(z_dim=self.model.l_mu.out_features)
        all_metrics.update(gen_metrics)
        
        # 4. Interpolation smoothness
        all_metrics.update(self.interpolation_smoothness(z_dim=self.model.l_mu.out_features))
        
        return all_metrics, {
            'latent_vectors': latent_vectors,
            'generated_images': generated_images
        }

def print_evaluation_results(metrics):
    """
    Pretty print evaluation results
    """
    print("\n" + "="*60)
    print("📊 VAE EVALUATION RESULTS")
    print("="*60)
    
    print("\n🔧 RECONSTRUCTION QUALITY:")
    recon_mse = metrics['reconstruction_mse']
    print(f"  MSE: {recon_mse['mean']:.6f} ± {recon_mse['std']:.6f}")
    print(f"  Range: [{recon_mse['min']:.6f}, {recon_mse['max']:.6f}]")
    
    recon_ssim = metrics['reconstruction_ssim']
    print(f"  SSIM: {recon_ssim['mean']:.4f} ± {recon_ssim['std']:.4f}")
    
    print("\n🧠 LATENT SPACE ANALYSIS:")
    latent_stats = metrics['latent_stats']
    print(f"  Latent norm: {latent_stats['mean_norm']:.4f} ± {latent_stats['std_norm']:.4f}")
    print(f"  Activation: {latent_stats['mean_activation']:.4f} ± {latent_stats['std_activation']:.4f}")
    print(f"  Diversity: {metrics['latent_diversity']:.4f}")
    
    posterior_stats = metrics['posterior_stats']
    print(f"  μ (mean): {posterior_stats['mu_mean']:.4f} ± {posterior_stats['mu_std']:.4f}")
    print(f"  log σ² (logvar): {posterior_stats['logvar_mean']:.4f} ± {posterior_stats['logvar_std']:.4f}")
    
    print("\n🎨 GENERATION QUALITY:")
    pixel_stats = metrics['pixel_stats']
    print(f"  Pixel range: [{pixel_stats['min_pixel_value']:.4f}, {pixel_stats['max_pixel_value']:.4f}]")
    print(f"  Pixel mean: {pixel_stats['mean_pixel_value']:.4f} ± {pixel_stats['std_pixel_value']:.4f}")
    
    for channel in ['red', 'green', 'blue']:
        channel_stats = metrics[f'{channel}_channel']
        print(f"  {channel.capitalize()}: {channel_stats['mean']:.4f} ± {channel_stats['std']:.4f}")
    
    print("\n🌈 INTERPOLATION QUALITY:")
    interp_smooth = metrics['interpolation_smoothness']
    print(f"  Smoothness: {interp_smooth['mean']:.6f} ± {interp_smooth['std']:.6f}")
    
    print("\n" + "="*60)

# Usage example
def evaluate_trained_vae(model_path, test_images_path=None):
    """
    Quick evaluation of trained VAE
    """
    from vae_model import ConvolutionnalVAE
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    z_dim = 256  # Update based on your model
    model = ConvolutionnalVAE(image_channels=3, z_dim=z_dim, input_size=256).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = VAEEvaluator(model, device)
    
    # For now, generate some dummy test images (replace with real test set)
    dummy_images = torch.randn(500, 3, 256, 256)  # Replace with actual test images
    
    # Run evaluation
    metrics, extras = evaluator.comprehensive_evaluation(dummy_images)
    
    # Print results
    print_evaluation_results(metrics)
    
    return metrics, extras

if __name__ == "__main__":
    # Example usage
    metrics, extras = evaluate_trained_vae('vae_final_model.pth')