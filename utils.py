import torch
import numpy as np
import os

def read_pred_file(file):
    boxes = np.zeros((0,5))
    if os.path.isfile(file):
        with open(file) as f:
            lines = f.readlines()
        for l in lines:
            l = l.split(" ")
            if len(l)==6:
                [xc, yc, h, w, conf] = np.array(l[1:]).astype("float")
            else:
                conf = 1
                [xc, yc, h, w] = np.array(l[1:]).astype("float") # label file

            box = np.array(xywh2xyxy(np.array([xc, yc, h, w])).tolist() + [conf])
            boxes = np.concatenate([boxes, np.expand_dims(box, axis=0)])

    return boxes


def test_pred(labels, preds, conf_th=0.25, iou_th=0):
    fp = 0
    tp = 0
    fn = 0
    if len(labels)==0:
        for pred in preds:
            [xc, yc, h, w, conf] = pred
            if conf > conf_th:
                fp += 1

    else:
        if len(labels) == 0:
            gt_ok = []
        else:
            gt_ok = [False] * len(labels)

        for pred in preds:
            conf = pred[-1]
            match = False
            for i, label in enumerate(labels):
                iou = box_iou(
                    torch.tensor(pred[:4]).unsqueeze(0),
                    torch.tensor(label[:4]).unsqueeze(0),
                ).item()
                if conf >= conf_th and iou > iou_th:
                    tp += 1
                    gt_ok[i] = True
                    match = True

            if not match and conf >= conf_th:
                fp += 1

        fn += len(gt_ok) - sum(gt_ok)

    return fp, tp, fn


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.array): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.array): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (np.minimum(a2,b2[:,None,:])- np.maximum(a1,b1[:,None,:])).clip(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return  inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:,None] - inter + eps)


def NMS(boxes, overlapThresh=0):
    """Non maximum suppression

    Args:
        boxes (np.array): A numpy array of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2, conf) format
        overlapThresh (int, optional): iou threshold. Defaults to 0.

    Returns:
        boxes: Boxes after NMS
    """
    # Return an empty list, if no boxes given
    boxes = boxes[boxes[:, -1].argsort()]
    if len(boxes) == 0:
        return []

    indices = np.arange(len(boxes))
    rr = box_iou(boxes[:,:4], boxes[:,:4])
    for i, box in enumerate(boxes):
        temp_indices = indices[indices != i]
        if np.any(rr[i,temp_indices]>overlapThresh):
            indices = indices[indices != i]
   
    return boxes[indices]
