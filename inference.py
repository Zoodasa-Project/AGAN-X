import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.generator import Generator
from utils.image_processing import save_image, plot_comparison
from config import DEVICE, GENERATOR_PATH, UPSCALE_FACTOR

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((img.size[1] // UPSCALE_FACTOR, img.size[0] // UPSCALE_FACTOR)),  # 입력 이미지 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)

def postprocess_image(tensor):
    # [-1, 1] 범위에서 [0, 1] 범위로 변환
    tensor = (tensor + 1) / 2.0
    # 값을 0과 1 사이로 클리핑
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def upscale_image(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor.to(DEVICE))
    return postprocess_image(output.cpu().squeeze(0))

def main(input_path, output_path):
    # 모델 로드
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    model.eval()

    # 이미지 로드 및 전처리
    img = load_image(input_path)
    img_tensor = preprocess_image(img)

    # 업스케일링 수행
    upscaled_img = upscale_image(model, img_tensor)

    # 결과 저장
    save_image(upscaled_img, output_path)

    # 원본과 업스케일링된 이미지 비교 플롯
    plot_comparison(img_tensor.squeeze(0), transforms.ToTensor()(img), upscaled_img)

    print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale an image using the trained model.")
    parser.add_argument("input_path", type=str, help="Path to the input low resolution image.")
    parser.add_argument("output_path", type=str, help="Path to save the upscaled image.")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
