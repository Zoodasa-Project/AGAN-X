import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, BATCH_SIZE, UPSCALE_FACTOR, DEVICE
from torchvision import transforms

augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
])


class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            image = augmentation_transforms(image)  # 데이터 증강 적용
        
        # 고해상도 이미지
        target = image
        # 저해상도 이미지 생성
        input = transforms.Resize((image.shape[1] // UPSCALE_FACTOR, image.shape[2] // UPSCALE_FACTOR))(image)
        
        return input, target

def get_data_loaders():
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 이미지 크기 통일
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 데이터셋 생성
    train_dataset = AnimeDataset(root_dir=RAW_DATA_DIR, transform=transform)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    return train_loader

# 데이터 증강을 위한 함수
def augment_data(image):
    augment_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    return augment_transforms(image)
