import torch
from torchvision import transforms
from PIL import Image
from models.upscaler import AdvancedAnimeUpscaler
from config import CONFIG

def load_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, filename):
    tensor = tensor.squeeze().cpu().float().numpy()
    tensor = (tensor + 1) / 2.0 * 255.0
    tensor = tensor.clip(0, 255).astype("uint8")
    tensor = tensor.transpose(1, 2, 0)
    image = Image.fromarray(tensor)
    image.save(filename)

def upscale_image(model, image_path, output_path):
    model.eval()
    with torch.no_grad():
        input_image = load_image(image_path, CONFIG['image_size']).to(CONFIG['device'])
        t = torch.zeros(1, dtype=torch.long, device=CONFIG['device'])
        upscaled_image = model(input_image, t)
        save_image(upscaled_image, output_path)

if __name__ == "__main__":
    model = AdvancedAnimeUpscaler(
        CONFIG['image_size'], CONFIG['patch_size'], CONFIG['num_classes'],
        CONFIG['dim'], CONFIG['depth'], CONFIG['heads'], CONFIG['mlp_dim']
    ).to(CONFIG['device'])
    
    model.load_state_dict(torch.load("checkpoint_epoch_1000.pth"))
    
    input_image_path = "input_image.jpg"
    output_image_path = "upscaled_image.png"
    
    upscale_image(model, input_image_path, output_image_path)
    print(f"Upscaled image saved to {output_image_path}")
