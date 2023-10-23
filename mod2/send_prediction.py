import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Define the same CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 channels, 4x4 image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output has 10 classes for MNIST

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

# Load the saved model
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Not using my dataset
# For simplicity, we'll use a random tensor as input
input_data = torch.randn(1, 1, 28, 28)  # Replace with your image data

# Get the class prediction
with torch.no_grad():
    output = model(input_data)
    _, predicted_class = output.max(1)
    prediction_result = predicted_class.item()

# Connect to the named pipe created by the C program
FIFO_NAME = "my_pipe"

if not os.path.exists(FIFO_NAME):
    print("Named pipe does not exist. Make sure the C program is running.")
    exit(1)

# Send the prediction result to the C process
with open(FIFO_NAME, "w") as pipe:
    pipe.write(str(prediction_result) + "\n")
    pipe.flush()

print(f"Inference result sent to C process: {prediction_result}")
