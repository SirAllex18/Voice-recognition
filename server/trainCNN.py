import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from speaker_cnn import SpeakerCNN
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


X_train_part = np.load("X_train_part.npy")
y_train_part = np.load("y_train_part.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


X_train_tensor = torch.tensor(X_train_part, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_part, dtype=torch.long)

X_val_tensor   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)


X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)
test_dataset   = TensorDataset(X_test_tensor, y_test_tensor)
test_loader    = DataLoader(test_dataset, batch_size=32, shuffle=False)


num_speakers = len(np.unique(y_train_part))
print("Number of speakers in (train_part) dataset:", num_speakers)
model = SpeakerCNN(num_speakers=num_speakers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)

EPOCHS = 15
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
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

    model.eval()
    val_loss = 0.0
    val_steps = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            loss_val = criterion(val_outputs, val_labels)

            val_loss += loss_val.item()
            val_steps += 1

            _, pred_val = torch.max(val_outputs, 1)
            correct_val += (pred_val == val_labels).sum().item()
            total_val   += val_labels.size(0)

    val_loss = val_loss / val_steps
    val_acc = 100.0 * correct_val / total_val
    print(f"          Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")


    scheduler.step(val_loss)

model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        outputs_test = model(test_inputs)
        loss_test = criterion(outputs_test, test_labels)

        test_loss += loss_test.item()
        _, test_pred = torch.max(outputs_test, 1)
        correct_test += (test_pred == test_labels).sum().item()
        total_test   += test_labels.size(0)

test_loss /= len(test_loader)
test_acc  = 100.0 * correct_test / total_test
print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

torch.save(model.state_dict(), "speaker_cnn_model.pth")
print("Model saved to speaker_cnn_model.pth")
