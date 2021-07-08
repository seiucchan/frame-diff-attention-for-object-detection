from __future__ import division

from comet_ml import Experiment

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.detect import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torchsummary import summary

import cv2
from PIL import Image


f = open('comet_api.txt', 'r')
MY_API_KEY = f.read()
MY_API_KEY = MY_API_KEY.replace('\n', '')
experiment = Experiment(api_key=MY_API_KEY, project_name='futsal-analyzer')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--diff_mode", type=int ,default=0)
    opt = parser.parse_args()
    # print(opt)

    # Set hyper parameters
    hyper_params = {
    'batch_size': opt.batch_size,
    'epoch': opt.epochs,
    'gradient_accumulations': opt.gradient_accumulations,
    'model_def': opt.model_def,
    'data_config': opt.data_config,
    'pretrained_weights': opt.pretrained_weights,
    'n_cpu': opt.n_cpu,
    'img_size': opt.img_size,
    'checkpoint_interval': opt.checkpoint_interval,
    'evaluation_interval': opt.evaluation_interval,
    'compute_map': opt.compute_map,
    'multiscale_training': opt.multiscale_training,
    'diff_mode': opt.diff_mode
    }
    experiment.log_parameters(hyper_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    diff_mode = opt.diff_mode
    if diff_mode != 0:
        print("Use diff")
        train_diff_path = data_config["train_diff"]
        valid_diff_path = data_config["valid_diff"]
    else:
        train_diff_path = ""
        valid_diff_path = ""

    # Initiate model
    model = Darknet(opt.model_def)
    # model = model.to(device)
    # print(model.state_dict()['module_list.0.conv_0.weight'])
    model.apply(weights_init_normal)
   

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    if diff_mode == 1:
        model.module_list[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.state_dict()['module_list.0.conv_0.weight'] = torch.nn.init.xavier_uniform(model.module_list[0][0].weight)
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model = model.to(device)
    # Get dataloader
    dataset = ListDataset(train_path, train_diff_path, diff_mode, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # control by optimizer ???
    # if diff_mode == 1:
    #     for i, param in enumerate(model.parameters()):
    #         if i == 1:
    #             optimizer = torch.optim.Adam(param)
    # else:
    #     optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())
        
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    for epoch in range(opt.epochs):
        if epoch == 10:
            if diff_mode == 1:
                for i, param in enumerate(model.parameters()):
                    param.requires_grad = True
                    optimizer = torch.optim.Adam(model.parameters())
            model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))


            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]

               
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            model.seen += imgs.size(0)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                diff_path=valid_diff_path,
                diff_mode=diff_mode,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
           
           # person and ball metrics
            metrics = {
                'val_presision': precision.mean(),
                'val_precision_person': precision[0].mean(),
                'val_precision_ball': precision[1].mean(),
                'val_recall_ball': recall.mean(),
                'val_recall_person': recall[0].mean(),
                'val_recall_ball': recall[1].mean(),
                'val_mAP': AP.mean(),
                'val_AP_person': AP[0].mean(),
                'val_AP_ball': AP[1].mean(),
                'val_f1': f1.mean()
            }
            
            experiment.log_metrics(metrics, step=epoch)
            

        # if epoch % opt.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
        #     print("Save model: ",  f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
        





        # plot image
        # if epoch % 10 == 0:
        #     model.eval()
            # cv2_img = cv2.imread("data/obj/20191201F-netvsYSCC_00049.jpg")
            # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            # img = Image.open("data/obj/20191201F-netvsYSCC_00049.jpg")
            # demo_img = transforms.ToTensor()(Image.open("data/obj/20191201F-netvsYSCC_00049.jpg").convert('RGB'))
            # demo_diff = transforms.ToTensor()(Image.open("data/difference/20191201F-netvsYSCC/20191201F-netvsYSCC_00049.jpg").convert('L'))
            
            # demo_img = demo_img * demo_diff + demo_img
            # shape = np.array(cv2_img)
            # shape = shape.shape[:2]
            # detections = detect_image(img, demo_img, opt.img_size, model, device, conf_thres=0.5, nms_thres=0.5)
           
            # # if detections is not None:
            # #     # Rescale boxes to original image
            # #     detections = rescale_boxes(detections, opt.img_size, shape)
            # #     unique_labels = detections[:, -1].cpu().unique()
            # #     n_cls_preds = len(unique_labels)
            # #     # Bounding-box colors
            # #     cmap = plt.get_cmap("tab20b")
            # #     colors = [cmap(i) for i in np.linspace(0, 1, 20)]
            # #     bbox_colors = random.sample(colors, n_cls_preds)
            # #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # #         print(x1, y1, x2, y2, conf, cls_conf, cls_pred)
            # #         classes = ["person", "ball"]
            # #         # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            # #         box_w = x2 - x1
            # #         box_h = y2 - y1

            # #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # #         cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x1+box_w), int(y1+box_h)), color, 4)
            # #         cv2.putText(cv2_img, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            
            # experiment.log_image(demo_img, name="test_img_{}".format(epoch))
