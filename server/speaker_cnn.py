import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1,   16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16,  32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32,  64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2,2)
        
   
        self.fc1 = nn.Linear(10752, 256)
        self.fc2 = nn.Linear(256, num_speakers)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.pool(x)                    
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.pool(x)                     
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.pool(x)                     
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))  
        x = self.pool(x)                     
        # Flatten
        x = x.view(x.size(0), -1)           
        
        # Fully connected layers
        x = F.relu(self.fc1(x))            
        x = self.fc2(x)                    
        
        return x
