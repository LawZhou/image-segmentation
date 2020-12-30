from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import os
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
import gc


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]

def draw_circle_color(img, row, col, rad, color = "white"):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    if color == "white":
        img[rr[valid], cc[valid], 0] = val[valid]
        img[rr[valid], cc[valid], 1] = val[valid]
        img[rr[valid], cc[valid], 2] = val[valid]
    else:
        img[rr[valid], cc[valid], 0] = val[valid]
        img[rr[valid], cc[valid], 1] = 0
        img[rr[valid], cc[valid], 2] = 0

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img, models, device):
    model_row, model_col, model_rad = models

    img = img.reshape(1, 1, 200, 200)
    img = torch.from_numpy(img).float()
    img = img.to(device)
    output_row_tensor = model_row(img)
    output_col_tensor = model_col(img)
    output_rad_tensor = model_rad(img)

    if device == torch.device("cuda:0"):
        output_row = output_row_tensor.cpu().data[0]
        output_col = output_col_tensor.cpu().data[0]
        output_rad = output_rad_tensor.cpu().data[0]

    else:
        output_row = output_row_tensor.data[0]
        output_col = output_col_tensor.data[0]
        output_rad = output_rad_tensor.data[0]

    return output_row, output_col, output_rad



def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
            shape0.intersection(shape1).area /
            shape0.union(shape1).area
    )

def iou_loss(outputs, targets, device):
    if device == torch.device("cuda:0"):
        outputs = outputs.cpu().data
        targets = targets.cpu().data
    else:
        outputs = outputs.data
        targets = targets.data
    iou_total = 0
    for i in range(outputs.shape[0]):
        o = outputs[i]
        t = targets[i]

        o_param = (o[0], o[1], o[2])
        t_param = (t[0], t[1], t[2])

        iou_total += iou(o_param, t_param)

    mean = iou_total/outputs.shape[0]
    loss = torch.tensor([mean], requires_grad=True)

    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize(params, detected):
    row0, col0, rad0 = params
    row1, col1, rad1 = detected
    img = np.zeros((200, 200, 3), dtype=np.float)
    # draw original circle
    draw_circle_color(img, int(row0), int(col0), int(rad0))
    # draw predicted circle
    draw_circle_color(img, int(row1), int(col1), int(rad1), color="red")
    img = (img*255).astype(np.uint8)
    print('IOU for this prediction: %.4f'%(iou(detected, params)))
    io.imshow(img)
    plt.show()
######################################################################
# DATA LOADER
######################################################################

class CircleDataSet(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        param, img = noisy_circle(200, 50, 2)
        return img, param


######################################################################
# Model
######################################################################

class CNN(nn.Module):
    def __init__(self, num_filters, image_size):
        super(CNN, self).__init__()
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2), )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2), )
        self.downconv3 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters * 4,  kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.MaxPool2d(2), )
        self.downconv4 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.MaxPool2d(2), )

        self.finalconv = nn.Linear(num_filters*8*(image_size//16)*(image_size//16), 1)

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.downconv3(self.out2)
        self.out4 = self.downconv4(self.out3)
        out = self.out4.view(self.out4.size(0), -1)
        self.out_final = self.finalconv(out)
        return self.out_final

######################################################################
# MAIN LOOP
######################################################################

def mainLoop(num_epochs, learn_rate, trainloader, testloader, device):
    # Initialize model per parameter
    model_row = CNN(16, 200).to(device)
    model_col = CNN(16, 200).to(device)
    model_rad = CNN(16, 200).to(device)
    print("The model takes %d parameters" % (count_parameters(model_row)))
    # LOSS FUNCTION
    criterion = nn.MSELoss()
    # Initialize optimizer
    opt_row = torch.optim.Adam(model_row.parameters(), lr=learn_rate)
    opt_col = torch.optim.Adam(model_col.parameters(), lr=learn_rate)
    opt_rad = torch.optim.Adam(model_rad.parameters(), lr=learn_rate)
    # initialize best loss
    best_loss = 1e10
    for epoch in range(num_epochs):
        models, epoch_loss = train(epoch, num_epochs, (opt_row, opt_col, opt_rad), trainloader,
                                  (model_row, model_col, model_rad), device, criterion, iou_loss_fn=False)
        test(models, epoch, num_epochs, testloader, device, criterion)
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_models = models
    return best_models


def train(epoch, num_epochs, optimizers, trainloader, models, device, criterion, iou_loss_fn=False):
    o_row, o_col, o_rad = optimizers
    m_row, m_col, m_rad = models
    start = time.time()
    running_loss = 0

    for images, labels in trainloader:
        m_row.train()
        m_col.train()
        m_rad.train()
        labels = torch.stack(labels, 1).float()
        images = images.to(device)
        labels = labels.to(device)

        input = images.unsqueeze(1).float()
        output_row = m_row(input)
        output_col = m_col(input)
        output_rad = m_rad(input)

        o_row.zero_grad()
        o_col.zero_grad()
        o_rad.zero_grad()
        if not iou_loss_fn:
            loss_row = criterion(output_row[:,0], labels[:,0])
            loss_col = criterion(output_col[:,0], labels[:,1])
            loss_rad = criterion(output_rad[:,0], labels[:,2])
            loss_row.backward()
            loss_col.backward()
            loss_rad.backward()
            running_loss += loss_row.item() + loss_col.item() + loss_rad.item()
        else:
            outputs = torch.stack((output_row, output_col, output_rad), dim=1)
            loss = iou_loss(outputs, labels, device)
            loss.backward()
            running_loss += loss
        o_row.step()
        o_col.step()
        o_rad.step()

        gc.collect()

    print('Epoch [%d/%d] Loss: %.4f, Time (s): %d' % (
        epoch + 1, num_epochs, running_loss / len(trainloader), time.time() - start))
    return models, running_loss / len(trainloader)


def test(models, epoch, num_epochs, testloader, device, criterion):
    m_row, m_col, m_rad = models
    m_row.eval()
    m_col.eval()
    m_rad.eval()

    iou_total = 0
    running_loss = 0
    start = time.time()
    labels_store = []
    outputs_store = []
    for images, labels in testloader:
        labels = torch.stack(labels, 1).float()
        images = images.to(device)
        labels = labels.to(device)

        input = images.unsqueeze(1).float()

        output_row_tensor = m_row(input)
        output_col_tensor = m_col(input)
        output_rad_tensor = m_rad(input)
        if device == torch.device("cuda:0"):
            output_row = output_row_tensor.cpu().data[0]
            output_col = output_col_tensor.cpu().data[0]
            output_rad = output_rad_tensor.cpu().data[0]
            labels_param = labels.cpu().data[0]
        else:
            output_row = output_row_tensor.data[0]
            output_col = output_col_tensor.data[0]
            output_rad = output_rad_tensor.data[0]
            labels_param = labels.data[0]

        output_param = (output_row, output_col, output_rad)

        loss_row = criterion(output_row_tensor[:,0], labels[:,0])
        loss_col = criterion(output_col_tensor[:,0], labels[:,1])
        loss_rad = criterion(output_rad_tensor[:,0], labels[:,2])

        running_loss += loss_row.item() + loss_col.item() + loss_rad.item()

        iou_total += iou(output_param, labels_param)

    iou_avg = iou_total / len(testloader)
    val_loss = running_loss / len(testloader)
    time_elapsed = time.time() - start
    print('Epoch [%d/%d], IOU avg: %.4f, Val Loss: %.4f,Time(s): %d' % (
        epoch + 1, num_epochs, iou_avg, val_loss, time_elapsed))
    return outputs_store, labels_store

def startTraining(device):
    num_epochs = 20
    batch_size = 50
    learn_rate = 0.01

    train_set = CircleDataSet(50000)
    val_set = CircleDataSet(500)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(val_set, batch_size=1, shuffle=True)

    model_row, model_col, model_rad = mainLoop(num_epochs, learn_rate, trainloader, testloader, device)

    if not os.path.exists('q2_models'):
        os.makedirs('q2_models')
    torch.save(model_row.state_dict(), 'q2_models/model_row.pt')
    torch.save(model_col.state_dict(), 'q2_models/model_col.pt')
    torch.save(model_rad.state_dict(), 'q2_models/model_rad.pt')


def loadModels(device):
    # load modals
    model_row = CNN(16, 200)
    model_col = CNN(16, 200)
    model_rad = CNN(16, 200)
    model_row.load_state_dict(torch.load('q2_models/model_row.pt'))
    model_col.load_state_dict(torch.load('q2_models/model_col.pt'))
    model_rad.load_state_dict(torch.load('q2_models/model_rad.pt'))
    model_row = model_row.to(device)
    model_col = model_col.to(device)
    model_rad = model_rad.to(device)
    return (model_row, model_col, model_rad)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Toggle training mode and model test mode using this boolean
    train_model = False
    # Enable circle plotting by this boolean
    plot_circle = False
    if train_model:
        startTraining(device)
    else:
        models = loadModels(device)
        results = []
        for _ in range(1000):
            params, img = noisy_circle(200, 50, 2)
            detected = find_circle(img, models, device)
            if plot_circle:
                visualize(params, detected)
            results.append(iou(params, detected))
        results = np.array(results)
        print("IOU mean: %.4f" %(results.mean()))
        print("IOU > 0.7 mean: %.4f" %((results > 0.7).mean()))
