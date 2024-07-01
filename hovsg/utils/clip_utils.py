from argparse import ArgumentParser
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import clip
import open_clip
from PIL import Image
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm


# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    """
    Compute the coordinates of the image on the plot
    :param image (np.array): the image to plot
    :param x (float): the x coordinate of the image
    :param y (float): the y coordinate of the image
    :param image_centers_area_size (int): the size of the area where the images are plotted
    :param offset (int): the offset of the image from the border of the plot
    :return: the coordinates of the top left and bottom right corner
    """
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def match_text_to_imgs(language_instr, images_list):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    imgs_feats = get_imgs_feats(images_list)
    text_feats = get_text_feats([language_instr])
    scores = imgs_feats @ text_feats.T
    scores = scores.squeeze()
    return scores, imgs_feats, text_feats


def get_nn_img(raw_imgs, text_feats, img_feats):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    scores = img_feats @ text_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_imgs = [raw_imgs[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_ids, high_to_low_imgs, high_to_low_scores


def get_img_feats(img, preprocess, clip_model):
    """
    Get the image features from the CLIP model
    :param img (np.array): the image to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :return: the image features
    """
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_img_feats_batch(imgs, preprocess, clip_model):
    """
    Get the image features from the CLIP model for a batch of images
    :param imgs (list): the images to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :return: the image features
    """
    imgs_pil = [Image.fromarray(np.uint8(img)) for img in imgs]
    imgs_in = torch.stack([preprocess(img_pil) for img_pil in imgs_pil])
    with torch.no_grad():
        img_feats = clip_model.encode_image(imgs_in.cuda()).float()
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_imgs_feats(raw_imgs, preprocess, clip_model, clip_feat_dim):
    """
    Get the image features from the CLIP model for a list of images
    :param raw_imgs (list): the images to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :return: the image features
    """
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    for img_id, img in enumerate(raw_imgs):
        imgs_feats[img_id, :] = get_img_feats(img, preprocess, clip_model)
    return imgs_feats


def get_imgs_feats_batch(raw_imgs, preprocess, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the image features from the CLIP model for a list of images
    :param raw_imgs (list): the images to get the features from
    :param preprocess (torchvision.transforms): the preprocessing function
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the image features
    """
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    img_batch = []
    for img_id, img in enumerate(raw_imgs):
        if img.shape[0] == 0 or img.shape[1] == 0:
            img = [[[0, 0, 0]]]
        img_pil = Image.fromarray(np.uint8(img))
        img_in = preprocess(img_pil)[None, ...]
        img_batch.append(img_in)
        if len(img_batch) == batch_size or img_id == len(raw_imgs) - 1:
            img_batch = torch.cat(img_batch, dim=0)
            with torch.no_grad():
                batch_feats = clip_model.encode_image(img_batch.cuda()).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
            batch_feats = np.float32(batch_feats.cpu())
            imgs_feats[img_id - len(img_batch) + 1 : img_id + 1, :] = batch_feats
            img_batch = []
    return imgs_feats


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    # in_text = ["a {} in the scene.".format(in_text)]
    text_tokens = open_clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats


def get_text_feats_62_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model with 62 templates
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    multiple_templates = [
        "{}",
        "a photo of {} in the scene.",
        "a photo of the {} in the scene.",
        "a photo of one {} in the scene.",
        "I took a picture of of {}.",
        "I took a picture of of my {}.",  # itap: I took a picture of
        "I took a picture of of the {}.",
        "a photo of {}.",
        "a photo of my {}.",
        "a photo of the {}.",
        "a photo of one {}.",
        "a photo of many {}.",
        "a good photo of {}.",
        "a good photo of the {}.",
        "a bad photo of {}.",
        "a bad photo of the {}.",
        "a photo of a nice {}.",
        "a photo of the nice {}.",
        "a photo of a cool {}.",
        "a photo of the cool {}.",
        "a photo of a weird {}.",
        "a photo of the weird {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of a clean {}.",
        "a photo of the clean {}.",
        "a photo of a dirty {}.",
        "a photo of the dirty {}.",
        "a bright photo of {}.",
        "a bright photo of the {}.",
        "a dark photo of {}.",
        "a dark photo of the {}.",
        "a photo of a hard to see {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of {}.",
        "a cropped photo of the {}.",
        "a close-up photo of {}.",
        "a close-up photo of the {}.",
        "a jpeg corrupted photo of {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of {}.",
        "a blurry photo of the {}.",
        "a pixelated photo of {}.",
        "a pixelated photo of the {}.",
        "a black and white photo of the {}.",
        "a black and white photo of {}.",
        "a plastic {}.",
        "the plastic {}.",
        "a toy {}.",
        "the toy {}.",
        "a plushie {}.",
        "the plushie {}.",
        "a cartoon {}.",
        "the cartoon {}.",
        "an embroidered {}.",
        "the embroidered {}.",
        "a painting of the {}.",
        "a painting of a {}.",
    ]
    mul_tmp = multiple_templates.copy()
    multi_temp_landmarks_other = [x.format(lm) for lm in in_text for x in mul_tmp]
    # format the text with multiple templates except for "background"
    text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)
    # average the features
    text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    return text_feats


def get_text_feats_multiple_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model with text templates
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
        # "a photo of {} in the scene.",
        # "a photo of the {} in the scene.",
        # "a photo of one {} in the scene.",
        # "I took a picture of of {}.",
        # "I took a picture of of my {}.",  # itap: I took a picture of
        # "I took a picture of of the {}.",
        # "a photo of {}.",
        # "a photo of my {}.",
        # "a photo of the {}.",
        # "a photo of one {}.",
        # "a photo of many {}.",
        # "a good photo of {}.",
        # "a good photo of the {}.",
        # "a bad photo of {}.",
        # "a bad photo of the {}.",
        # "a photo of a nice {}.",
        # "a photo of the nice {}.",
        # "a photo of a cool {}.",
        # "a photo of the cool {}.",
        # "a photo of a weird {}.",
        # "a photo of the weird {}.",
        # "a photo of a small {}.",
        # "a photo of the small {}.",
        # "a photo of a large {}.",
        # "a photo of the large {}.",
        # "a photo of a clean {}.",
        # "a photo of the clean {}.",
        # "a photo of a dirty {}.",
        # "a photo of the dirty {}.",
        # "a bright photo of {}.",
        # "a bright photo of the {}.",
        # "a dark photo of {}.",
        # "a dark photo of the {}.",
        # "a photo of a hard to see {}.",
        # "a photo of the hard to see {}.",
        # "a low resolution photo of {}.",
        # "a low resolution photo of the {}.",
        # "a cropped photo of {}.",
        # "a cropped photo of the {}.",
        # "a close-up photo of {}.",
        # "a close-up photo of the {}.",
        # "a jpeg corrupted photo of {}.",
        # "a jpeg corrupted photo of the {}.",
        # "a blurry photo of {}.",
        # "a blurry photo of the {}.",
        # "a pixelated photo of {}.",
        # "a pixelated photo of the {}.",
        # "a black and white photo of the {}.",
        # "a black and white photo of {}.",
        # "a plastic {}.",
        # "the plastic {}.",
        # "a toy {}.",
        # "the toy {}.",
        # "a plushie {}.",
        # "the plushie {}.",
        # "a cartoon {}.",
        # "the cartoon {}.",
        # "an embroidered {}.",
        # "the embroidered {}.",
        # "a painting of the {}.",
        # "a painting of a {}.",
    ]
    mul_tmp = multiple_templates.copy()
    multi_temp_landmarks_other = [x.format(lm) for lm in in_text for x in mul_tmp]
    # format the text with multiple templates except for "background"
    text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)
    # average the features
    text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    return text_feats
