import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    """텐서를 PIL 이미지로 변환합니다."""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def save_image(tensor, filename):
    """텐서를 이미지 파일로 저장합니다."""
    image = tensor_to_image(tensor)
    image.save(filename)

def plot_comparison(low_res, high_res, generated):
    """저해상도, 고해상도, 생성된 이미지를 비교 플롯합니다."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(tensor_to_image(low_res))
    axs[0].set_title('Low Resolution')
    axs[1].imshow(tensor_to_image(high_res))
    axs[1].set_title('High Resolution')
    axs[2].imshow(tensor_to_image(generated))
    axs[2].set_title('Generated')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def calculate_psnr(img1, img2):
    """두 이미지 간의 PSNR(Peak Signal-to-Noise Ratio)을 계산합니다."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()
