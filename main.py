#import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import Dataset
from torchvision import transforms
from models.CustomModel import CustomNet
from models.ResnetModel import CustomResnet
from models.ENetModel import CustomENet
from sklearn.model_selection import train_test_split as splits
import pandas as pd

full = pd.read_csv('train.csv')
mals = full[(full['target'] == 1)]
bens = full[(full['target'] == 0)]
trmals, temals = splits(mals, train_size=.75)
trbens, tebens = splits(bens, train_size=.75)
del full
train_data = trmals.append(trbens)
val_data = temals.append(tebens)
test_data = pd.read_csv('test.csv')

train_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

test_transforms = transforms.Compose([
                        transforms.ToTensor()
])

train_set = Dataset('data/train/', train_data, transform = train_transforms)
val_set = Dataset('data/train/', val_data, transform = train_transforms)
test_set = Dataset('data/test/', test_data, transform = test_transforms, train=False)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

model = CustomENet()
model.load_state_dict(torch.load('./models/trained/CustomENet.model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Using', device + '...')

epochs = 31
lr = .0001
weights = [.005, .995]
weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=.1)

model = model.to(device)
bestAcc = 0.0
with open('pr_enet.txt', 'r') as f:
    bestAcc = float(f.readline())

print('Current best accuracy:', bestAcc)
for epoch in range(epochs):

    correct = 0
    total = 0
    running_loss = 0
    model.train()

    for i, batch in enumerate(tqdm(train_loader)):
        imgs = batch['data'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        _, pred = torch.max(outputs,1)
        correct += (pred == labels).sum().item()

    print('Epoch:', epoch, 'Loss:', running_loss / total, 'Training Accuracy:', correct / total)
    scheduler.step()
    if(epoch % 5 == 0):
        print('Validation in Epoch:', epoch)
        correct = 0
        total = 0
        running_loss = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                imgs = batch['data'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                _, pred = torch.max(outputs,1)
                correct += (pred == labels).sum().item()
        print('Validation for Epoch:', epoch, 'Loss:', running_loss / total, 'Validation Accuracy:', correct/total)
        if(correct/total > bestAcc):
            bestAcc = correct/total
            torch.save(model.state_dict(), './models/trained/CustomENet.model')

with open('pr_enet.txt','w') as f:
    f.write(str(bestAcc))
