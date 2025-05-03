""" This project is a study of the health of oysters, specifically
 the infestation of mud blisters from burrowing worms, to build an
 image recognition model that can accurately predict how much surface 
 area of an oyster is infected with the parasites.
"""

# (0) load up the modules ...

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam

import os
import matplotlib.pyplot as plt
import numpy as np
from src.resnet_test import test_model

# (1)
# Compose is a helper that lets you chain together several image-processing steps into one callable
# Every time an image is loaded, PyTorch will run it through your full chain of preprocessing steps automatically

# resizes, transforms to Tensor vectors, [flips, rotates,] and normalizes the images 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),

    # Those [0.485, 0.456, 0.406] means and [0.229, 0.224, 0.225] std-devs are simply 
    # the per-channel average and standard deviation of the ImageNet training set, 
    # which the authors of torchvision pre-computed once and published for you to re-use.
    # Normalize the image with mean and standard deviation for each channel (R, G, B)
    # The mean and standard deviation values are precomputed for the ImageNet dataset
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# same thing as above but for the validation dataset
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# finds and gets the directory path of where the python file is
script_dir = os.path.dirname(os.path.abspath(__file__))

# Augmented dataset paths
train_data_dir = os.path.join(script_dir, '../dataset/train')
val_data_dir = os.path.join(script_dir, '../dataset/validation')
test_data_dir = os.path.join(script_dir, '../dataset/test')

# Create datasets from the image directories and transforms them
train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=val_transforms)
# test_dataset ... (set below with testing code)

# Create data loaders, but shuffles the images for train so it is not looking at the 
# severity levels in order, batch_size is the number of images it takes when passing 
# through the neural network

# group images into “mini-batches” of 32 examples apiece;
# modern hardware (especially GPUs) works much faster when you feed it 
# many examples at once rather than one at a time.
#
# also, batch for Gradient Stability (training only): 
# During training, you average the loss over each batch to get 
# a more stable gradient step. In testing/evaluation, you don’t 
# compute gradients, but you still usually evaluate in batches so 
# you don’t have to load the entire dataset into memory at once.

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test images loaded below

# Load ResNet50 pre-trained model
model = models.resnet50(pretrained=True)

# model = models.resnet50(pretrained=False)  # Load without pre-trained weights
# model.load_state_dict(torch.load('resnet50_mud_blisters.pth'))

# Freeze all layers except the last layer, prevents from changing the parameters of the
# original resnet model
for param in model.parameters():
    param.requires_grad = False

# gets all the different possible classes of information from previous layers and constructs 
# and reduces it to the last layer of five classses (severity 0-4)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)


# Define loss and optimizer

# Cross-entropy measures how far the predicted probabilities are from the true class. In other words,
# how wrong is it
criterion = nn.CrossEntropyLoss() 

# An optimizer in machine learning is an algorithm that adjusts the weights (parameters) of a model to minimize 
# the loss function during training, it is used for learning purposes, how fast it should learn
optimizer = Adam(model.fc.parameters(), lr=0.001) 

# see if there is a GPU, if not use the CPU and puts the model into the specified hardware device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# intializing function to train the model (this is when the model learns)
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1):

    # epochs is the number of cycles it learns
    for epoch in range(num_epochs):
        model.train()

        # variables to keep track of stats
        running_loss = 0.0
        correct = 0
        total = 0
        
        # goes through images per patch from the train dataset
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass to get model output
            outputs = model(images)

            # computation for the loss of predicted value to the actual value
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training loss and accuracy (more stats)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # more stats of how the model is doing and how it is adjusting itself after each batch learned
        epoch_loss = running_loss / len(train_loader.dataset)

        # accuracy is based off predictions of training
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # time to test the model after learning
        validate_model(model, val_loader)

# Validation function to see how well the model does when it is learning
def validate_model(model, val_loader):
    # have the model evaluate itself thorugh testing
    model.eval()
    correct = 0
    total = 0
    
    # disabling learning and adjusting since the model is being tested, no learning
    with torch.no_grad():
        # iterates through images through the validation dataset
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # get the model's answers from the images and see if it is correct
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')

# -------------------------------------------------------
# Train the model using our train function
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

# set the model to evaluation mode 
model.eval()

# Function to display image, prediction, and label
def show_images_predictions(dataloader):
    image_count = 0  # Counter to keep track of the number of images processed
    
    print("cwd =", os.getcwd())
    print("exists aug_correct_severity_predic?", os.path.isdir("aug_correct_severity_predic"))
    print("exists aug_incorrect_severity_predic?", os.path.isdir("aug_incorrect_severity_predic"))

    
    
    for images, labels in dataloader:
        # Have the model make the prediction
        with torch.no_grad():
            output = model(images.to(device))
        _, predicted_classes = output.max(1)

        # Display the image and its predicted and actual labels
        for i in range(images.size(0)):
            #print("*** image #", i)
            # Denormalize the image
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.imshow(img)  # Convert tensor to image

            # checking to see if the model prediction and the actual label match so the plot can be saved in the right directory
            # for model analysis

            if predicted_classes[i].item() == labels[i].item():
                plt.title(f'Predicted: {predicted_classes[i].item()}, Actual: {labels[i].item()}', color='green')
                plt.savefig(f'./aug_correct_severity_predic/oyster_{image_count}.jpg')
                plt.close()
            else:
                plt.title(f'Predicted: {predicted_classes[i].item()}, Actual: {labels[i].item()}', color='red')
                plt.savefig(f'./aug_incorrect_severity_predic/oyster_{image_count}.jpg')
                plt.close()

            image_count += 1  # Increment the counter

            #plt.show()

show_images_predictions(val_loader)


# Save the model
torch.save(model.state_dict(), 'trainedModel/resnet50_mud_blisters.pth')

# --------------------------------------------------
# TESTING ...
# transformations for the images so it can be put into the model
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# the directory path of where the test directory
test_data_dir = os.path.join(script_dir, '../dataset/test')

# Load the test data
test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transforms) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

# loading the saved model
model = models.resnet50()

# gets the original Res-Net50 model and changes the output features to 5 classes
# to symbolize the 5 severity levels of mudblister oysters
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

# loading the saved model from above that has the saved learning weights
model.load_state_dict(torch.load("trainedModel/resnet50_mud_blisters.pth", weights_only=True))

# set the model to evaluation mode for testing
model.eval()

# Test the model 
test_model(model, test_loader)