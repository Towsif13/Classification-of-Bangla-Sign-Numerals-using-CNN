# imports
import torch
import torch.nn as nn
from data import*
from model import ModelNet
import os
import matplotlib.pyplot as plt


# hyperparameters
num_epochs = 20
learning_rate = 0.01

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model initialization
model = ModelNet()
model.to(device)

#loss and criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
n_total_steps = len(train_dl)
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for i, (images, lables) in enumerate(train_dl):
        images = images.to(device)
        lables = lables.to(device)

        optimizer.zero_grad()

        # forward
        outputs = model(images)
        loss = criterion(outputs, lables)

        # backward

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        if(i+1) % 20 == 0:
            print(
                f'=> epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f} ')

    # validation loop
    model.eval()
    with torch.no_grad():
        for i, (images, lables) in enumerate(val_dl):
            images = images.to(device)
            lables = lables.to(device)

            outputs = model(images)
            loss = criterion(outputs, lables)

            val_loss += loss.item() * images.size(0)

    epoch_val_loss = val_loss / len(val_dl)

    epoch_train_loss = train_loss / len(train_dl)
    print(
        f'Epoch {epoch+1} => Train loss: {epoch_train_loss:.4f} => Val loss: {epoch_val_loss:.4f}')
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    plt.plot(train_losses, 'r')
    plt.plot(val_losses, 'b')
    plt.savefig('loss_curve.png')


PATH = 'enitre_model.pth'
torch.save(model, PATH)
