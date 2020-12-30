import time

import numpy as np
import torch
from torch.optim import lr_scheduler


def train(epoch, num_epochs, optimizer, trainloader, unet, device, criterion, scheduler=None):
    start = time.time()
    unet.train()  # Change model to 'train' mode
    running_loss = 0
    accuracy = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = unet(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if(scheduler):
            scheduler.step()
        accuracy = calculate_dice_coeff(accuracy, labels, outputs, device)
        running_loss += loss.item()
    train_acc = accuracy / len(trainloader)
    print('Training: Epoch [%d/%d] Loss: %.4f, Dice Coefficient: %.4f, Time (s): %d' % (
        epoch + 1, num_epochs, running_loss/len(trainloader), train_acc, time.time() - start))
    return unet, running_loss/len(trainloader)


def test(unet, epoch, num_epochs, testloader, device, criterion):
    unet.eval()
    start = time.time()
    val_acc, val_loss, _ = runValidation(criterion, device, testloader, unet)
    time_elapsed = time.time() - start
    print('Testing: Epoch [%d/%d], Val Loss: %.4f, Dice Coefficient: %.4f, Time(s): %d' % (
        epoch+1, num_epochs, val_loss, val_acc, time_elapsed))
    return val_acc


def runValidation(criterion, device, testloader, unet, store=False):
    accuracy = 0
    running_loss = 0
    outputs_store = []
    labels_store = []
    img_store = []
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = unet(images)
        if store:
            outputs_store.append(outputs)
            labels_store.append(labels)
            img_store.append(images)
        loss = criterion(outputs, labels)
        accuracy = calculate_dice_coeff(accuracy, labels, outputs, device)
        running_loss += loss.item()
    val_loss = running_loss / len(testloader)
    val_acc = accuracy / len(testloader)
    stores = (outputs_store, labels_store, img_store)
    return val_acc, val_loss, stores


def calculate_dice_coeff(accuracy, labels, outputs, device):
    if device == torch.device("cuda:0"):
        outputs = np.array(outputs.cpu().data)
        labels = np.array(labels.cpu().data)
    else:
        outputs = np.array(outputs.data)
        labels = np.array(labels.data)
    outputs[outputs>=0.5] = 1.0
    outputs[outputs<0.5] = 0.0
    labels[labels>=0.5] = 1.0
    labels[labels<0.5] = 0.0
    equals = (outputs == labels).sum()
    accuracy += 2*equals/(outputs.size + labels.size)
    return accuracy


def mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion):
    optimizer = torch.optim.Adam(unet.parameters(), lr=learn_rate)
    best_loss = 1e10
    scheduler = lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.1)
    best_model = None
    for epoch in range(num_epochs):
        unet, epoch_loss= train(epoch, num_epochs, optimizer, trainloader, unet, device, criterion, scheduler=scheduler)
        test(unet, epoch, num_epochs, testloader, device, criterion)
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_model = unet
    return best_model