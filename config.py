import torch
import os

# 디바이스 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터 관련 설정
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 모델 관련 설정
MODEL_DIR = 'models'
GENERATOR_PATH = os.path.join(MODEL_DIR, 'generator.pth')
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'discriminator.pth')

# 학습 관련 설정
LEARNING_RATE = 0.0001  # 학습률 감소
BATCH_SIZE = 16  # 배치 크기 감소 (메모리 문제가 없다면 증가 고려)
NUM_EPOCHS = 500  # 에폭 수 증가
UPSCALE_FACTOR = 4

# 평가 관련 설정
EVAL_INTERVAL = 5  # 5 에폭마다 평가

print(f"Using device: {DEVICE}")
