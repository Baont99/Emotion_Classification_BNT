import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class EmotionDataConfig: 
    N_CLASSES = 8
    IMG_SIZE = 224
    ID8DLABEL = {0: 'Surprise', 1: 'Sad', 2:'Neutral', 3:'Happy', 4:'Fear', 5:'Disgust', 6:'Contempt', 7:'Anger' } 
    LABEL8ID = {'Surprise': 0, 'Sad': 1,'Neutral' :2, 'Happy' :3, 'Fear':4,'Disgust':5, 'Contempt':6, 'Anger' :7 } 
    Emotion_labels = ['Surprise', 'Sad', ] 
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32

class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = 'resnet18'
    MODEL_WEIGHT = ROOT_DIR / 'models' / 'weights' / 'emotion_weights.pt' 
    DEVICE = 'cpu'
    LEARNING_RATE = 1e-3  # Learning rate for model training
    WEIGHT_DECAY = 1e-5  # Weight decay for regularization
    NUM_EPOCHS = 10  # Number of training epochs