import os
import torch
import torchvision
import torch.nn as nn
import time
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataLoader import imagesDataSet, makeDataSet
from src.model import UNet
from src.trainModels import runValidation, mainLoop
from src.visualize import plot


def train_q1(criterion):
    ########################
    # Question1.1
    ########################
    train_set = imagesDataSet(train_img_data, train_mask_data, device)
    val_set = imagesDataSet(test_img_data, test_mask_data, device)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    unet = UNet(64, num_class).to(device)
    unet = mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion)
    saveModel(unet, "q1_models", "model_q1.1.pt")

def train_q2(criterion):
    ########################
    # Question1.2
    ########################
    train_img_data_aug = makeDataSet(train_images_path, augmentation=True)
    train_mask_data_aug = makeDataSet(train_mask_path, augmentation=True)
    test_img_data_aug = makeDataSet(test_images_path)
    test_mask_data_aug = makeDataSet(test_mask_path)
    train_set = imagesDataSet(train_img_data_aug, train_mask_data_aug, device)
    val_set = imagesDataSet(test_img_data_aug, test_mask_data_aug, device)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    unet = UNet(64, num_class).to(device)
    unet = mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion)
    saveModel(unet, "q1_models", "model_q1.2.pt")

def train_q3(criterion):
    ########################
    # Question1.3
    ########################
    # Resize images and convert to grayscale to fit my Unet
    input_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # Fetch data from VOC2012
    train_set = torchvision.datasets.VOCSegmentation(root='./data', download=True, transform=input_transform,
                                                     target_transform=target_transform)
    test_set = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', download=True,
                                                    transform=input_transform, target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True, num_workers=2)
    unet = UNet(64, num_class).to(device)
    unet = mainLoop(unet, 50, learn_rate, trainloader, testloader, device, criterion)
    # train our own cat data set
    train_set = imagesDataSet(train_img_data, train_mask_data, device)
    val_set = imagesDataSet(test_img_data, test_mask_data, device)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    unet = mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion)
    saveModel(unet, "q1_models", "model_q1.3.pt")

def saveModel(unet, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(unet.state_dict(), folder + '/' + filename)

######################################################################
# Run Model
######################################################################

def loadModel(model_name):
    # load modals
    model = UNet(64, num_class)
    model.load_state_dict(torch.load('q1_models/' + model_name))
    model = model.to(device)
    return model

def runTest(unet, criterion, need_plot=False):
    val_set = imagesDataSet(test_img_data, test_mask_data, device)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    start = time.time()
    val_acc, val_loss, stores = runValidation(criterion, device, testloader, unet, store=need_plot)
    time_elapsed = time.time() - start
    print('Finish running test: Val Loss: %.4f, Dice Coefficient: %.4f, Time(s): %d' % (
        val_loss, val_acc, time_elapsed))
    return stores

def runQ1(need_plot, need_visualize, criterion):
    unet = loadModel("model_q1.1.pt")
    outputs_store, labels_store, img_store = runTest(unet, criterion, need_plot=need_plot)
    if need_plot:
        plot(outputs_store, labels_store, img_store, "question1.1_plot", device, need_visualize=need_visualize)

def runQ2(need_plot, need_visualize, criterion):
    unet = loadModel("model_q1.2.pt")
    outputs_store, labels_store, img_store = runTest(unet, criterion, need_plot=need_plot)
    if need_plot:
        plot(outputs_store, labels_store, img_store, "question1.2_plot", device, need_visualize=need_visualize)

def runQ3(need_plot, need_visualize, criterion):
    unet = loadModel("model_q1.3.pt")
    outputs_store, labels_store, img_store = runTest(unet, criterion, need_plot=need_plot)
    if need_plot:
        plot(outputs_store, labels_store, img_store, "question1.3_plot", device, need_visualize=need_visualize)

if __name__ == '__main__':
    train_images_path = './cat_data/Train/input/*.jpg'
    train_mask_path = './cat_data/Train/mask/*.jpg'

    test_images_path = './cat_data/Test/input/*.jpg'
    test_mask_path = './cat_data/Test/mask/*.jpg'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_epochs = 40
    batch_size = 10
    learn_rate = 0.0005
    num_class = 1

    train_img_data = makeDataSet(train_images_path)
    train_mask_data = makeDataSet(train_mask_path)

    test_img_data = makeDataSet(test_images_path)
    test_mask_data = makeDataSet(test_mask_path)

    # Toggle these this boolean to enable training mode and model test mode
    startTrain = False
    # Enable output the image
    plot_cat = True
    # Enable segmentation in original image
    plot_cat_segmentation = True

    criterion = nn.MSELoss()
    # criterion = DiceLoss()
    if startTrain:
        train_q1(criterion)
        train_q2(criterion)
        # train_q3(criterion)
    else:
        # In model test mode, I assume you have the corresponding model in ./q1_models directory
        runQ1(plot_cat, plot_cat_segmentation, criterion)
        runQ2(plot_cat, plot_cat_segmentation, criterion)
        # runQ3(plot_cat, plot_cat_segmentation, criterion)