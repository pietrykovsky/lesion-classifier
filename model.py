import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

INPUT_SIZE = 224

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Sequential(
                      nn.Linear(512, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, 1),
                      )
model = model.cuda()