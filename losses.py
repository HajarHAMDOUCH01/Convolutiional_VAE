import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cvae_loss(recon_x, x):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    return BCE

def KLdivergence_loss(mu, log_var):
    # KL Divergence loss
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD


VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

class VGG19(nn.Module):
    """VGG19 features for perceptual loss"""
    def __init__(self):
        super(VGG19, self).__init__()

        from torchvision.models import vgg19, VGG19_Weights
        vgg_features = vgg19(weights='DEFAULT').features

        self.vgg_model_weights = VGG19_Weights
        
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # conv1_2
        self.slice1 = nn.Sequential(*[vgg_features[x] for x in range(0, 3)])  
        # conv2_2
        self.slice2 = nn.Sequential(*[vgg_features[x] for x in range(3, 8)])  
        # conv_3_2
        for x in range(8, 13):
            self.slice3.add_module(str(x), vgg_features[x])
        # conv_4_2
        for x in range(13, 21):
            self.slice4.add_module(str(x), vgg_features[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x : image 
        h_conv_1_2 = self.slice1(x)
        h_conv_2_2 = self.slice2(h_conv_1_2)
        h_conv_3_2 = self.slice3(h_conv_2_2)
        h_conv_4_2 = self.slice4(h_conv_3_2)

        return [h_conv_1_2, h_conv_2_2, h_conv_3_2, h_conv_4_2]
    
def perceptual_loss_cvae(vgg19_model, recon_x, x):
    """
    Fixed perceptual loss with proper gradient flow
    """
    loss_layers_indices = [0, 1, 2, 3]
    loss_layers_weighting = [0.01, 0.02, 0.5, 0.5]
    total_reconstruction_loss = 0.0

    # FIXED: Remove torch.no_grad() to allow gradients to flow
    # Compute features for original images (can be done without gradients)
    with torch.no_grad():
        x_features = vgg19_model.forward(x)
    
    # Compute features for reconstructed images (MUST have gradients)
    recon_x_features = vgg19_model.forward(recon_x)

    for layer in loss_layers_indices:
        _, c, h, w = x_features[layer].shape
        
        # FIXED: Proper device handling and normalization
        x_layer_features_normalized = x_features[layer] / (c * h * w)
        recon_x_layer_features_normalized = recon_x_features[layer] / (c * h * w)
        
        # Apply layer weighting
        layer_weight = loss_layers_weighting[layer]
        layer_loss = layer_weight * F.mse_loss(
            recon_x_layer_features_normalized, 
            x_layer_features_normalized, 
            reduction='mean'
        )
        total_reconstruction_loss += layer_loss
    
    return total_reconstruction_loss

def cvae_total_loss(vgg19_model, recon_x, x, mu, log_var, beta, mae_weight=1.0, percep_weight=0.5, use_percep=False):
    """
    Total VAE loss with optional perceptual loss
    """
    mae_loss = mae_weight * cvae_loss(recon_x, x)
    KLD_loss = KLdivergence_loss(mu, log_var)

    if use_percep:
        percep_loss = percep_weight * perceptual_loss_cvae(vgg19_model, recon_x, x)
        recon_loss = mae_loss + percep_loss
    else:
        recon_loss = mae_loss

    total_loss = recon_loss + beta * KLD_loss
    
    return total_loss, beta * KLD_loss, recon_loss