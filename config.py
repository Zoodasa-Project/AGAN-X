import torch

CONFIG = {
    # Model parameters
    'image_size': 256,
    'patch_size': 16,
    'num_classes': 1000,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    
    # Training parameters
    'batch_size': 16,
    'num_epochs': 1000,
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    
    # Diffusion parameters
    'num_diffusion_steps': 1000,
    
    # Data parameters
    'data_dir': './data/anime_faces',
    
    # Device
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Loss weights
    'perceptual_weight': 1.0,
    'adversarial_weight': 0.1,
    'tv_weight': 1e-4,
}
