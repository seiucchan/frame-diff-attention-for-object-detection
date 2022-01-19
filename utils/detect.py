from utils.utils import non_max_suppression

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable


def detect_image(img, demo_img, img_size, model, device, conf_thres, nms_thres):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    # print("h-w:", max(int((imh-imw)/2), 0))  # ->0
    # print("w-h:", max(int((imw-imh)/2), 0))  # ->91
    img_transforms = transforms.Compose([
         transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0),
                         max(int((imh-imw)/2), 0),
                         max(int((imw-imh)/2), 0)),
                        (128, 128, 128)),
         transforms.ToTensor(),
     ])
    # convert image to Tensor
    demo_img = transforms.functional.to_pil_image(demo_img)
    inp = img_transforms(demo_img).float().unsqueeze_(0)
    inp = inp.to(device)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(inp)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    return detections[0] if detections[0] is not None else []
