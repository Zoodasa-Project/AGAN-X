import torch
import torch.optim as optim
from models.upscaler import AdvancedAnimeUpscaler
from utils.data_loader import get_dataloader
from utils.loss_functions import PerceptualLoss, AdversarialLoss, total_variation_loss
from utils.training_utils import generate_noise_levels, noise_images, denoise_images, meta_learning_update, generate_adversarial_examples
from config import CONFIG

def train():
    model = AdvancedAnimeUpscaler(
        CONFIG['image_size'], CONFIG['patch_size'], CONFIG['num_classes'],
        CONFIG['dim'], CONFIG['depth'], CONFIG['heads'], CONFIG['mlp_dim']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    
    dataloader = get_dataloader(CONFIG['data_dir'], CONFIG['batch_size'], CONFIG['image_size'])
    
    perceptual_loss = PerceptualLoss().to(CONFIG['device'])
    adversarial_loss = AdversarialLoss().to(CONFIG['device'])
    
    for epoch in range(CONFIG['num_epochs']):
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Generate noisy images
            t = generate_noise_levels(CONFIG['batch_size'], CONFIG['num_diffusion_steps'], CONFIG['device'])
            noisy_images = noise_images(batch, t, CONFIG['num_diffusion_steps'])
            
            # Denoise images
            denoised_images = denoise_images(model, noisy_images, t, CONFIG['num_diffusion_steps'])
            
            # Calculate losses
            p_loss = perceptual_loss(denoised_images, batch)
            a_loss = adversarial_loss(model.discriminator(batch), model.discriminator(denoised_images))
            tv_loss = total_variation_loss(denoised_images)
            
            total_loss = CONFIG['perceptual_weight'] * p_loss + \
                         CONFIG['adversarial_weight'] * a_loss + \
                         CONFIG['tv_weight'] * tv_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Meta-learning update
            meta_model = meta_learning_update(model, batch)
            
            # Adversarial training
            adv_batch = generate_adversarial_examples(model, batch)
            adv_loss = perceptual_loss(model(adv_batch), batch)
            adv_loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {total_loss.item():.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
