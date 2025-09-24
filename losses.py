import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cvae_loss(recon_x, x, mu, log_var, beta=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    
    # KL Divergence loss
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + beta * KLD, BCE, KLD


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
    """VGG19 features for perceptual losses"""
    def __init__(self):
        super(VGG19, self).__init__()

        from torchvision.models import vgg19, VGG19_Weights
        vgg_features = vgg19(weights='DEFAULT').features

        self.vgg_model_weights = VGG19_Weights
        
        # Extraction of specific layers for content and style losses
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        # relu1_2 
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        # relu_2_2 
        for x in range(4,9):
            self.slice2.add_module(str(x), vgg_features[x])
        # relu3_3
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_features[x])
        # relu4_3 
        for x in range(16, 25):
            self.slice4.add_module(str(x), vgg_features[x])
            
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x : image 
        
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
    
def perceptual_loss_cvae(recon_x, x, mu, log_var, beta=1.0):
    total_reconstruction_loss = 0.0
    vgg19_model = VGG19()
    vgg19_model.to(device)
    vgg19_model.eval()

    with torch.no_grad():
        x_features = vgg19_model.forward(x)
        recon_x_features = vgg19_model.forward(recon_x)

    for layer in range(4):
        x_features[layer].to(device)
        recon_x_features[layer].to(device)
        layer_loss = F.mse_loss(x_features[layer], recon_x_features[layer], reduction='mean')
        total_reconstruction_loss += layer_loss
    
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return total_reconstruction_loss + beta * KLD, total_reconstruction_loss , KLD
    
    