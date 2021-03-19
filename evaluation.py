# imports
import torch
from data import*

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = 'enitre_model.pth'
model = torch.load(PATH)
model.eval()

with torch.no_grad():
    n_correct_test = 0
    n_samples_test = 0
    for images, lables in test_dl:
        images = images.to(device)
        lables = lables.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # 1 is the lables
        n_samples_test += lables.shape[0]  # 0 is samples per batch
        n_correct_test += (predictions == lables).sum().item()

    test_acc = 100 * n_correct_test / n_samples_test
    #print(f'accuracy = {acc}')
    n_correct_val = 0
    n_samples_val = 0
    for images, lables in val_dl:
        images = images.to(device)
        lables = lables.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # 1 is the lables
        n_samples_val += lables.shape[0]  # 0 is samples per batch
        n_correct_val += (predictions == lables).sum().item()

    val_acc = 100 * n_correct_val / n_samples_val

    n_correct_train = 0
    n_samples_train = 0
    for images, lables in train_dl:
        images = images.to(device)
        lables = lables.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # 1 is the lables
        n_samples_train += lables.shape[0]  # 0 is samples per batch
        n_correct_train += (predictions == lables).sum().item()

    train_acc = 100 * n_correct_train / n_samples_train

    print(
        f'Test accuracy is {test_acc:.2f} % - validation accuracy is {val_acc:.2f} % - train accuracy is {train_acc:.2f} %')
