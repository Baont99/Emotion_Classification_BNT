import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.Emotion_model import EmotionModel
from config.Emotion_cfg import EmotionDataConfig, ModelConfig

N_CLASSES = 8
SAVE_PATH = "models/weights/emotion_weights.pt"

data_transform = transforms.Compose([
   transforms.Resize((EmotionDataConfig.IMG_SIZE, EmotionDataConfig.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(EmotionDataConfig.NORMALIZE_MEAN,
                                         EmotionDataConfig.NORMALIZE_STD)
])


# Đường dẫn đến thư mục chứa dữ liệu (mỗi cảm xúc là một thư mục con)
base_dir = 'phanloaicamxuc'

# Tạo một ImageFolder chứa toàn bộ dữ liệu
dataset = datasets.ImageFolder(root=base_dir, transform=data_transform)

# Đếm tổng số lượng mẫu
total_size = len(dataset)

# Tính toán số mẫu cho từng tập (train, val, test 6-2-2)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Chia ngẫu nhiên dataset thành 3 tập: train, val, test
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Tạo DataLoader cho từng tập
train_loader = DataLoader(train_dataset, EmotionDataConfig.TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, EmotionDataConfig.VAL_BATCH_SIZE, shuffle=False)

model = EmotionModel(N_CLASSES)
model.fit(train_loader, 
          val_loader, 
          learning_rate = ModelConfig.LEARNING_RATE,
          weight_decay= ModelConfig.WEIGHT_DECAY,
          num_epochs=ModelConfig.NUM_EPOCHS)

torch.save(model.state_dict(), SAVE_PATH)