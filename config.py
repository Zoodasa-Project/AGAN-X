import os

# 데이터 관련 설정
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 모델 관련 설정
MODEL_DIR = 'models'
GENERATOR_PATH = os.path.join(MODEL_DIR, 'generator.pth')
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'discriminator.pth')

# 학습 관련 설정
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
UPSCALE_FACTOR = 4

# 디바이스 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
