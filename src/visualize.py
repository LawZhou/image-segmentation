import cv2 as cv
import numpy as np
import torch


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