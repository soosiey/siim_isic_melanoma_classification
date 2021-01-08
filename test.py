import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
#import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import Dataset
from torchvision import transforms
#from models.CustomModel import CustomNet
#from models.ResnetModel import CustomResnet
from models.ENetModel import CustomENet
#from sklearn.model_selection import train_test_split as splits
import pandas as pd
import torch.nn.functional as F

test_data = pd.read_csv('test.csv')


test_transforms = transforms.Compose([
                        transforms.ToTensor()
])

test_set = Dataset('data/test/', test_data, transform = test_transforms, train=False)

test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

model = CustomENet()
model.load_state_dict(torch.load('./models/trained/CustomENet.model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
outputpd = pd.DataFrame(columns=['image_name','target'])
print('Using', device + '...')


model = model.to(device)
model.eval()


for i in tqdm(range(len(test_data))):
    img_name = test_data.iloc[i,0]
    img = np.array(Image.open('data/test/' + img_name + '.png'))
    img = test_transforms(img).to(device)
    img = img.view(-1,3,224,224)
    outputs = model(img)
    pred = F.softmax(outputs, dim = 1)
    pred = pred[:,1]
    outputpd = outputpd.append({'image_name':img_name, 'target':pred.item()}, ignore_index=True)
print(outputpd.head())
outputpd.to_csv('submission.csv', index=False)
