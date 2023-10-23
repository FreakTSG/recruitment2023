import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional
import torchvision.models as models


training_set: torch.utils.data.Dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
validation_set: torch.utils.data.Dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())



from bobnet import BobNet
from cnn import CNN;

model1 = BobNet()
model2=CNN()

# batch size
MLP_BATCH_SIZE=64

# learning rate
MLP_LEARNING_RATE=0.001

# momentum
MLP_MOMENTUM=0.9

# training epochs to run
MLP_EPOCHS=10

# create the training loader
mlp_training_loader = DataLoader(training_set, batch_size=MLP_BATCH_SIZE, shuffle=True) 

# create the validation loader
mlp_validation_loader = DataLoader(validation_set, batch_size=MLP_BATCH_SIZE, shuffle=True)

mlp_loss_fn = torch.nn.CrossEntropyLoss()

#mlp_optimizer = torch.optim.SGD(model1.parameters(), lr=MLP_LEARNING_RATE, momentum=MLP_MOMENTUM)
mlp_optimizer = torch.optim.SGD(model2.parameters(), lr=MLP_LEARNING_RATE, momentum=MLP_MOMENTUM)

import utils

# how many batches between logs
LOGGING_INTERVAL=100

#utils.train_model(model1, MLP_EPOCHS, mlp_optimizer, mlp_loss_fn, mlp_training_loader, mlp_validation_loader, LOGGING_INTERVAL)
utils.train_model(model2, MLP_EPOCHS, mlp_optimizer, mlp_loss_fn, mlp_training_loader, mlp_validation_loader, LOGGING_INTERVAL)
torch.save(model2.state_dict(),'model_weights.pth')
torch.save(model2, 'model.pth')