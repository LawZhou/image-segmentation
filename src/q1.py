import cv2 as cv
import os
import glob
import numpy as np
import torch
import torchvision
import torch.nn as nn
import time
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms

######################################################################
# Helper functions
######################################################################
def resizeImg(image):
    '''Resize an image to 128*128'''
    return cv.resize(image, (128, 128))


def makeDataSet(path, augmentation = None):
    images = []
    for file in glob.glob(path):
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = resizeImg(img)
        if augmentation:
            # original image
            images.append(np.expand_dims(img, axis=0))
            # Horizontal Flip
            horizontal_flip = cv.flip(img, 1)
            horizontal_flip = np.expand_dims(horizontal_flip, axis=0)
            images.append(horizontal_flip)
            # rotated image
            rotation = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            rotation = np.expand_dims(rotation, axis=0)
            images.append(rotation)
            # flip image horizontally and vertically
            flip = cv.flip(img, -1)
            flip = np.expand_dims(flip, axis=0)
            images.append(flip)
            # increase contrast of the image
            contrast = cv.equalizeHist(img)
            contrast = np.expand_dims(contrast, axis=0)
            images.append(contrast)

        else:
            img = np.expand_dims(img, axis=0)
            images.append(img)

    return np.array(images)/255.0

def plot(outputs_store, labels_store, img_store, filename, device, x_dim=200, y_dim=200, need_visualize=False):
    if (len(outputs_store) == 0):
        return None
    else:
        output = plotImageFromData(device, outputs_store, x_dim, y_dim)
        labels = plotImageFromData(device, labels_store, x_dim, y_dim)
        if need_visualize:
            images = plotImageFromData(device, img_store, x_dim, y_dim, orig_image=True, need_visualize=need_visualize,
                                       mask_store=outputs_store)
        else:
            images = plotImageFromData(device, img_store, x_dim, y_dim, orig_image=True, need_visualize=need_visualize)
        result = np.hstack((images, labels, output))
        cv.imwrite(filename+".png", result)


def plotImageFromData(device, outputs_store, x_dim, y_dim, orig_image=False, need_visualize=False, mask_store=None):
    plots = check_device(device, outputs_store, 0)
    mask = None
    if visualize and mask_store is not None:
        mask = check_device(device, mask_store, 0)
    plots = extracImagesFromBatch(plots, x_dim, y_dim, orig_image=orig_image, need_visualize=need_visualize, mask=mask)
    for i in range(1, len(outputs_store)):
        tensor = check_device(device, outputs_store, i)

        if visualize and mask_store is not None:
            mask = check_device(device, mask_store, i)

        one_batch = extracImagesFromBatch(tensor, x_dim, y_dim, orig_image=orig_image, need_visualize=need_visualize, mask=mask)
        plots = np.vstack((plots, one_batch))
    plots = (plots*255.0).astype(np.uint8)
    return plots


def check_device(device, outputs_store, i):
    if device == torch.device("cuda:0"):
        out = outputs_store[i].cpu()
    else:
        out = outputs_store[i]
    plots = np.array(out.data)
    return plots


def extracImagesFromBatch(batch, x_dim, y_dim, orig_image=False, need_visualize=False, mask=None):
    batch = np.squeeze(batch, axis=1)
    if not orig_image:
        batch[batch>=0.5] = 1.0
        batch[batch<0.5] = 0.0
    result = cv.resize(batch[0], (x_dim, y_dim))
    if need_visualize and mask is not None and orig_image:
        mask = np.squeeze(mask, axis=1)
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        first_mask = cv.resize(mask[0], (x_dim, y_dim))
        result = visualize(result, first_mask)
    for i in range(1, batch.shape[0]):
        img =  cv.resize(batch[i], (x_dim, y_dim))
        if need_visualize and mask is not None and orig_image:
            mask_i = cv.resize(mask[i], (x_dim, y_dim))
            img = visualize(img, mask_i)
        result = np.vstack((result, img))
    return result

def visualize(image, mask):

    mask = (mask * 255.0).astype(np.uint8)
    mask = cv.Canny(mask, 100, 200)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    image = cv.drawContours(image, contours, -1, 1, 3)
    return image

######################################################################
# DATA LOADER
######################################################################

class imagesDataSet(Dataset):
    def __init__(self, input_images, output, device, transform=None):
        self.input_images = torch.from_numpy(input_images).float()
        self.output = torch.from_numpy(output).float()
        self.transform=transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        if self.transform:
            image = self.transform(image)

        return [image, output]

######################################################################
# Train
######################################################################
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
        accuracy = calculate_dice_coeff(accuracy, labels, outputs)
        running_loss += loss.item()
    train_acc = accuracy / len(trainloader)
    print('Training: Epoch [%d/%d] Loss: %.4f, Dice Coefficient: %.4f, Time (s): %d' % (
        epoch + 1, num_epochs, running_loss/len(trainloader), train_acc, time.time() - start))
    return unet, running_loss/len(trainloader)

######################################################################
# Test
######################################################################
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
        accuracy = calculate_dice_coeff(accuracy, labels, outputs)
        running_loss += loss.item()
    val_loss = running_loss / len(testloader)
    val_acc = accuracy / len(testloader)
    stores = (outputs_store, labels_store, img_store)
    return val_acc, val_loss, stores


def calculate_dice_coeff(accuracy, labels, outputs):
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

######################################################################
# MODEL
######################################################################

def seqLayersDown(in_channel, out_channel, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

def seqLayersUp(in_channel, out_channel, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Upsample(scale_factor=2),
    )

class UNet(nn.Module):
    def __init__(self, num_filter, num_class):
        super(UNet, self).__init__()
        kernel = 3
        padding = kernel//2
        self.downconv1 = seqLayersDown(1, num_filter, kernel, 1)
        self.downconv2 = seqLayersDown(num_filter, num_filter*2, kernel, padding)
        self.downconv3 = seqLayersDown(num_filter*2, num_filter*4, kernel, padding)
        self.downconv4 = seqLayersDown(num_filter*4, num_filter*8, kernel, padding)

        self.rfconv = nn.Sequential(
                        nn.Conv2d(num_filter*8, num_filter*8, kernel_size=kernel, padding=padding),
                        nn.BatchNorm2d(num_filter*8),
                        nn.ReLU()
                     )

        self.upconv4 = seqLayersUp(num_filter*8 + num_filter*8, num_filter*8, kernel, padding)
        self.upconv3 = seqLayersUp(num_filter*8 + num_filter*4, num_filter*4, kernel, padding)
        self.upconv2 = seqLayersUp(num_filter*4 + num_filter*2, num_filter*2, kernel, padding)
        self.upconv1 = seqLayersUp(num_filter*2 + num_filter, 3, kernel, padding)

        self.finalconv = nn.Conv2d(1+3, num_class,  kernel_size=kernel, padding=padding)


    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.downconv3(self.out2)
        self.out4 = self.downconv4(self.out3)
        self.rfOut = self.rfconv(self.out4)
        self.out5 = self.upconv4(torch.cat((self.rfOut, self.out4), dim = 1))
        self.out6 = self.upconv3(torch.cat((self.out5, self.out3), dim = 1))
        self.out7 = self.upconv2(torch.cat((self.out6, self.out2), dim = 1))
        self.out8 = self.upconv1(torch.cat((self.out7, self.out1), dim = 1))
        self.out_final = self.finalconv(torch.cat((self.out8, x), dim=1))
        return self.out_final


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predict, target):
        loss = torch.mean((predict - target) ** 2)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        predict = predict.contiguous()
        target = target.contiguous()
        intersect = (predict * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersect + 1.) / (predict.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + 1.)))
        return loss.mean()
######################################################################
# MAIN LOOP
######################################################################

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


def train_q1():
    ########################
    # Question1.1
    ########################
    train_set = imagesDataSet(train_img_data, train_mask_data, device)
    val_set = imagesDataSet(test_img_data, test_mask_data, device)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    unet = UNet(64, num_class).to(device)
    # Toggle using MSE loss or Dice loss here:
    criterion = MSELoss()
    # criterion = DiceLoss()
    unet = mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion)
    saveModel(unet, "q1_models", "model_q1.1.pt")

def train_q2():
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
    criterion = MSELoss()
    unet = mainLoop(unet, num_epochs, learn_rate, trainloader, testloader, device, criterion)
    saveModel(unet, "q1_models", "model_q1.2.pt")

def train_q3():
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
    criterion = MSELoss()
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

def runQ1(need_plot, need_visualize):
    unet = loadModel("model_q1.1.pt")
    criterion = MSELoss()
    outputs_store, labels_store, img_store = runTest(unet, criterion, need_plot=need_plot)
    if need_plot:
        plot(outputs_store, labels_store, img_store, "question1.1_plot", device, need_visualize=need_visualize)

def runQ2(need_plot, need_visualize):
    unet = loadModel("model_q1.2.pt")
    criterion = MSELoss()
    outputs_store, labels_store, img_store = runTest(unet, criterion, need_plot=need_plot)
    if need_plot:
        plot(outputs_store, labels_store, img_store, "question1.2_plot", device, need_visualize=need_visualize)

def runQ3(need_plot, need_visualize):
    unet = loadModel("model_q1.3.pt")
    criterion = MSELoss()
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
    startTrain = True
    # Enable output the image
    plot_cat = True
    # Enable segmentation in original image
    plot_cat_segmentation = True
    if startTrain:
        train_q1()
        train_q2()
        train_q3()
    else:
        # In model test mode, I assume you have the corresponding model in ./q1_models directory
        runQ1(plot_cat, plot_cat_segmentation)
        runQ2(plot_cat, plot_cat_segmentation)
        runQ3(plot_cat, plot_cat_segmentation)