#4 losses:
#   1) adversarial losses
#   2) cycle losses
#   3) reconstruction loss
#   4) regularization loss
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import MSELoss, L1Loss, KLDivLoss


#receives
def adversarialLoss(discriminator_out, isReal):
    if isReal:
        target = torch.ones_like(discriminator_out)
    else:
        target = torch.zeros_like(discriminator_out)
    criterion = MSELoss()
    loss = criterion(discriminator_out, target)
    return loss

def cycleLoss(image, reconstructed):
    criterion = L1Loss()
    loss = criterion(image, reconstructed)
    return loss

#VAE errora
def reconstructionLoss(generated_image, target_image):
    criterion = nn.L1Loss()
    return criterion(generated_image, target_image)

def KLDLoss(mu, logvar):
    loss = -0.5 * torch.mean( 1 + logvar - mu.pow(2) - torch.exp(logvar) )
    return loss