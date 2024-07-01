import random
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np

def filter_masks(masks):
    """
    check if any mask contain another mask and remove the smaller one frm the bigger one.
    return the filtered masks.
    :param masks (list): list of masks output from sam.
    """
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i == j:
                continue
            # check if mask i contain mask j by anding them
            anded = np.logical_and(masks[i]["segmentation"], masks[j]["segmentation"])
            if np.all(anded == masks[j]["segmentation"]):
                # subtract mask j from mask i
                masks[i]["segmentation"] = np.logical_xor(masks[i]["segmentation"], masks[j]["segmentation"])
                masks[i]["area"] = np.sum(masks[i]["segmentation"])

    # remove empty masks
    masks = [mask for mask in masks if np.sum(mask["segmentation"]) > 0]
    return masks

def show_anns(anns):
    """
    Display the masks on the image.
    :param anns (list): list of masks output from sam.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def increase_bbox_by_margin(bbox, margin):
    """
    Increases the size of a bounding box by the given margin.

    :param bbox: The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param margin: The margin to increase the bounding box size by in pixels.
    :return: The increased bounding box coordinates as a tuple of (x, y, w, h).
    """
    x, y, w, h = bbox
    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2
    # Check if x is negative
    if x < 0:
        w += x
        x = 0

    # Check if y is negative
    if y < 0:
        h += y
        y = 0
    return (x, y, w, h)

def draw_all_bounding_boxs(image, masks, iou_threshold=0.89, color=(0, 255, 0), thickness=2, bbox_margin=0):
    """
    Draws and crop bounding boxs on the given image using the provided XYWH format bbox.

    :param image: The input image as a numpy array.
    :param masks: all masks sam output.
    :param bbox: The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param color: The color of the bounding box as a tuple of (R, G, B).
    :param thickness: The thickness of the bounding box edges in pixels.
    :param bbox_margin: increase the bbox by this value of pixel in all direction.
    :return: The image with the bounding box drawn on it.
    """
    bbox_image = image.copy()
    for mask in masks:
        if mask["predicted_iou"] > iou_threshold:
            m = mask["segmentation"]
            # Add the margin to the bounding box
            x, y, w, h = increase_bbox_by_margin(mask["bbox"], bbox_margin)
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), color, thickness)
    return bbox_image


def crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=0):
    """
    Draws and crop bounding boxs on the given image using the provided XYWH format bbox.

    :param image: The input image as a numpy array.
    :param masks: all masks sam output.
    :param bbox: The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param color: The color of the bounding box as a tuple of (R, G, B).
    :param thickness: The thickness of the bounding box edges in pixels.
    :param bbox_margin: increase the bbox by this value of pixel in all direction.
    :return: The image with the bounding box drawn on it.
    """
    images = []
    for mask in masks:
        if block_background:
            crop = crop_image(image, mask)
        else:
            crop = crop_bbox(image, mask["bbox"], bbox_margin)
        crop = cv2.resize(crop, (512, 512))
        images.append(crop)
    return images


def crop_image(image, mask):
    """
    Crops the input image to the mask.
    :param image (np.array): The input image as a numpy array.
    :param mask (dict): The mask as a dictionary with the "segmentation" and "bbox" keys.
    :return: The cropped image.
    """
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    x, y, w, h = int(x), int(y), int(w), int(h)
    crop = masked[y : y + h, x : x + w, :]
    return crop


def crop_bbox(image, bbox, bbox_margin=0):
    """
    Crops the input image to the bounding box.
    :param image (np.array): The input image as a numpy array.
    :param bbox (tuple): The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param bbox_margin (int): The margin to increase the bounding box size by in pixels.
    :return: The cropped image.
    """
    # Add the margin to the bounding box
    x, y, w, h = increase_bbox_by_margin(bbox, bbox_margin)
    # make x, y, w, h positive and not bigger than the image size and int
    x, y, w, h = int(x), int(y), int(w), int(h)
    crop = image[y : y + h, x : x + w]
    return crop


def plot_cropped_images(images, file_name="", path="output_img/", ssh=False):
    """
    Plots a list of images in a grid using Matplotlib.

    :param images: A list of images as NumPy arrays.
    """
    random_plot_name = random.randint(0, 100000)
    n_images = len(images)
    rows = int(n_images**0.5)
    cols = int(n_images / rows) + int(n_images % rows > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < n_images:
            ax.imshow(images[i])
        ax.axis("off")
    if ssh:
        plt.savefig(path + "all_crops_" + file_name + ".png")
    else:
        ax.set_title("Cropped Images")
        plt.show()


def overlay_mask_on_image(image, mask, negative=False):
    """
    Overlays a binary mask onto an input image.

    :param image: The input image as a numpy array.
    :param mask: The binary mask as a numpy array of the same size as the input image.
    :return: The image with the mask overlayed on it.
    """
    output = image.copy()
    mask = np.array(mask)
    mask = np.dstack((mask, mask, mask))
    if not negative:
        mask = np.invert(mask)
    output *= mask
    return output


def overlay_mask_on_image_with_score(image, mask, score):
    """
    Overlays a binary mask onto an input image.

    :param image: The input image as a numpy array.
    :param mask: The binary mask as a numpy array of the same size as the input image.
    :return: The image with the mask overlayed on it.
    """
    output = image.copy()
    mask = np.array(mask)
    mask = np.dstack((mask, mask, mask))
    color_mask = [0, 1, 0]
    for i in range(3):
        mask[:, :, i] = color_mask[i]
    mask = mask * score
    output += (mask * 255).astype(np.uint8)
    return output


def mask_and_crop_image(image, mask, bbox, bbox_margin=0):
    """
    Masks the input image with the provided binary mask and crops it to the bounding box.
    :param image (np.array): The input image as a numpy array.
    :param mask (np.array): The binary mask as a numpy array of the same size as the input image.
    :param bbox (tuple): The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
    :param bbox_margin (int): The margin to increase the bounding box size by in pixels.
    :return: The cropped image with the mask overlayed on it.
    """
    masked = overlay_mask_on_image(image, mask, negative=True)
    croped = crop_bbox(masked, bbox, bbox_margin)
    return croped
