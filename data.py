# imports
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


# directories
TRAIN_DIR = 'C:/Users/Towsif/Desktop/Files/BdSL/BdSL_digits/split/train'
VAL_DIR = 'C:/Users/Towsif/Desktop/Files/BdSL/BdSL_digits/split/val'
TEST_DIR = 'C:/Users/Towsif/Desktop/Files/BdSL/BdSL_digits/split/test'


train_data = ImageFolder(TRAIN_DIR, transform=ToTensor())
val_data = ImageFolder(VAL_DIR, transform=ToTensor())
test_data = ImageFolder(TEST_DIR, transform=ToTensor())

batch_size = 64

# dataloaders
train_dl = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_data, batch_size*2, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_data, batch_size*2, pin_memory=True)
