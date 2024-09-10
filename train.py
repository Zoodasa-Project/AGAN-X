import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import AnimeDataset, get_data_loaders
from utils.image_processing import save_image, plot_comparison, calculate_psnr
from config import *
import time
import os

def create_directories():
    directories = ['samples', 'models', 'data/raw', 'data/processed']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory '{directory}' created or already exists.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {str(e)}")

# 학습 함수 시작 전에 호출
create_directories()

def train():
    # 데이터 로더 초기화
    train_loader = get_data_loaders()

    # 모델 초기화
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Loss 함수 정의
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Optimizer 정의
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # VGG19 특징 추출기 초기화 (내용 손실 계산용)
    vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE)
    for param in vgg.parameters():
        param.requires_grad = False

    # 훈련 루프
    for epoch in range(NUM_EPOCHS):
        for i, (low_res, high_res) in enumerate(train_loader):
            batch_size = low_res.size(0)
            high_res_label = torch.ones((batch_size, 1)).to(DEVICE)
            low_res_label = torch.zeros((batch_size, 1)).to(DEVICE)

            # 고해상도 및 저해상도 이미지를 장치로 이동
            high_res = high_res.to(DEVICE)
            low_res = low_res.to(DEVICE)

            #########################
            # 생성자 훈련
            #########################
            optimizer_G.zero_grad()

            # 생성된 고해상도 이미지
            generated = generator(low_res)

            # 판별자의 출력
            disc_output = discriminator(generated)

            # 생성자 손실 계산
            content_loss = content_criterion(generated, high_res)
            adversarial_loss = adversarial_criterion(disc_output, high_res_label)
            perceptual_loss = content_criterion(vgg(generated), vgg(high_res))
            gen_loss = content_loss + 1e-3 * adversarial_loss + 0.006 * perceptual_loss

            gen_loss.backward()
            optimizer_G.step()

            #########################
            # 판별자 훈련
            #########################
            optimizer_D.zero_grad()

            # 실제 고해상도 이미지에 대한 판별자 출력
            real_output = discriminator(high_res)
            real_loss = adversarial_criterion(real_output, high_res_label)

            # 생성된 고해상도 이미지에 대한 판별자 출력
            fake_output = discriminator(generated.detach())
            fake_loss = adversarial_criterion(fake_output, low_res_label)

            # 판별자 손실 계산
            disc_loss = (real_loss + fake_loss) / 2

            disc_loss.backward()
            optimizer_D.step()

            # 진행 상황 출력
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                      f"Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

        # 에폭 종료 시 샘플 이미지 저장
        save_image(generated[0], f"samples/generated_epoch_{epoch+1}.png")
        save_image(high_res[0], f"samples/real_epoch_{epoch+1}.png")

        # PSNR 계산
        psnr = calculate_psnr(generated[0], high_res[0])
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], PSNR: {psnr:.2f}")

    # 최종 모델 저장
    torch.save(generator.state_dict(), GENERATOR_PATH)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_PATH)

if __name__ == "__main__":
    train()
