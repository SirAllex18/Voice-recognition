import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = None 
        self.fc2   = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x_flat = x.view(x.size(0), -1)
        
        if self.fc1 is None:
            in_features = x_flat.size(1)
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, num_speakers).to(x.device)

        x = torch.relu(self.fc1(x_flat))
        x = self.fc2(x)
        return x

num_speakers = len(np.unique(y_train))
model = SpeakerCNN(num_speakers=num_speakers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader)
test_acc = 100.0 * correct / total
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

torch.save(model.state_dict(), "speaker_cnn_model.pth")
print("Model saved to speaker_cnn_model.pth")
