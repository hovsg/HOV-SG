"""IoU"""
import numpy as np


def pixel_accuracy(eval_segm, gt_segm, ignore=[]):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: pixel accuracy
    """

    # print("unique classes: ", np.unique(gt_segm))
    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

        # per_class_acc =  np.sum(np.logical_and(curr_eval_mask, curr_gt_mask)) / np.sum(curr_gt_mask)
        # print(i, per_class_acc)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm, ignore=[]):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: mean accuracy
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if t_i != 0:
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.sum(accuracy) / float(n_cl - n_overlap)
    return mean_accuracy_


def per_class_iou(eval_segm, gt_segm, ignore=[]):
    """
    for each class, compute
    n_ii / (t_i + sum_j(n_ji) - n_ii)
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: per class IoU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    gt_cl, n_cl_gt = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(gt_cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    iou = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        # if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
        #     continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        iou[i] = n_ii / (t_i + n_ij - n_ii)

    # mean_iou_ = np.sum(iou) / (n_cl_gt - n_overlap)
    return iou


def mean_iou(eval_segm, gt_segm, ignore=[]):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: mean IoU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    gt_cl, n_cl_gt = extract_classes(gt_segm)
    overlap, n_overlap = get_ignore_classes_num(gt_cl, ignore)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    iou = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        iou[i] = n_ii / (t_i + n_ij - n_ii)

    mean_iou_ = np.sum(iou) / (n_cl_gt - n_overlap)
    return mean_iou_


def frequency_weighted_iou(eval_segm, gt_segm, ignore=[]):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param ignore: list of classes to ignore
    :return: frequency weighted IoU
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_iou_ = list([0]) * n_cl

    n_ignore = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if c in ignore:
            n_ignore += np.sum(curr_gt_mask)
            continue

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_iou_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm) - n_ignore

    frequency_weighted_iou_ = np.sum(frequency_weighted_iou_) / sum_k_t_k
    return frequency_weighted_iou_


"""
Auxiliary functions used during evaluation.
"""


def get_ignore_classes_num(cl, ignore):
    """
    Returns the number of classes to ignore
    :param cl: list of classes
    :param ignore: list of classes to ignore
    :return: number of classes to ignore
    """
    overlap = [c for c in cl if c in ignore]
    return overlap, len(overlap)


def get_pixel_area(segm):
    """
    Returns the area of the segmentation
    :param segm: 2D array, segmentation
    :return: area of the segmentation
    """
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    """"
    Extracts the masks of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    """
    Extracts the classes from the segmentation
    :param segm: 2D array, segmentation
    :return: classes and number of classes
    """
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    """
    Returns the union of the classes
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :return: union of the classes
    """
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    """
    Extracts the masks of the segmentation
    :param segm: 2D array, segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    """
    Returns the size of the segmentation
    :param segm: 2D array, segmentation
    :return: size of the segmentation
    """
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    """
    Checks the size of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    """
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


"""
Exceptions
"""


class EvalSegErr(Exception):
    """
    Custom exception for errors during evaluation
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
