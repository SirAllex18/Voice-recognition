import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(72000, 64)
        self.fc2 = nn.Linear(64, num_speakers)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Flatten before the FC layers
        x = x.view(x.size(0), -1)
        #print("Flatten shape:", x.shape)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
