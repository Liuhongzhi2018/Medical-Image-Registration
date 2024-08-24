import argparse
import logging
import os
import sys
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
# RuntimeError: received 0 items of ancdata
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import time

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, ImageDataset, decollate_batch, DataLoader, CacheDataset
# from monai.data import NiftiSaver, decollate_batch
from monai.handlers import StatsHandler, TensorBoardStatsHandler, stopping_fn_from_metric
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    AsDiscrete,
    Activations,
    LoadImageD,
    OrientationD,
    SpacingD,
    ToTensorD,
    NormalizeIntensityD,
    ScaleIntensityD,
    EnsureType, 
    MapLabelValue,
    LoadImaged, 
    RandRotate90d, 
    Resized, 
    ScaleIntensityd
)
from monai.metrics import (
    ConfusionMatrixMetric,
    ROCAUCMetric, 
    DiceMetric, 
)
from monai.networks.utils import one_hot

from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy

# CUDA_VISIBLE_DEVICES=1 python ankle_fracture_class_285_45_282_image.py train --model_folder "class_checkpoints"
# CUDA_VISIBLE_DEVICES=1 python ankle_fracture_class_282_95_98_103_image_level2.py train --model_folder "/mnt/lhz/Github/SEU_Ankle_MONAI/class_checkpoints"


# train/val
# images num: 578 labels num: 578
# All label with number: {1: 126, 2: 28, 3: 10, 4: 45, 5: 24, 6: 161, 7: 22, 8: 50, 0: 9, 9: 103}
# All train label with number: {1: 75, 2: 16, 3: 6, 4: 27, 5: 14, 6: 96, 7: 13, 8: 30, 0: 5, 9: 40}
# All val label with number: {1: 25, 2: 6, 3: 2, 4: 9, 5: 5, 6: 32, 7: 4, 8: 10, 0: 2, 9: 30}

# train/test
# images num: 578 labels num: 578
# All label with number: {1: 126, 2: 28, 3: 10, 4: 45, 5: 24, 6: 161, 7: 22, 8: 50, 0: 9, 9: 103}
# All train label with number: {1: 75, 2: 16, 3: 6, 4: 27, 5: 14, 6: 96, 7: 13, 8: 30, 0: 5, 9: 40}
# All val label with number: {1: 26, 2: 6, 3: 2, 4: 9, 5: 5, 6: 33, 7: 5, 8: 10, 0: 2, 9: 33}


def get_model(n_classes=1):
    
    # model = monai.networks.nets.DenseNet121(
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     out_channels=n_classes)
    
    model = monai.networks.nets.Densenet169(
        spatial_dims=3, 
        in_channels=1, 
        out_channels=n_classes)

    # model = monai.networks.nets.DenseNet(
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     out_channels=n_classes, 
    #     init_features=64, 
    #     growth_rate=32, 
    #     block_config=(6, 12, 24, 16), 
    #     bn_size=4, 
    #     act=('relu', {'inplace': True}), 
    #     norm='batch', 
    #     dropout_prob=0.0)
    
    # model = monai.networks.nets.resnet101(
    #     spatial_dims=3, 
    #     n_input_channels=1, 
    #     num_classes=n_classes)

    # model = monai.networks.nets.ResNet(
    #     block=ResNetBottleneck,   # "basic" or "bottleneck"
    #     layers=[], 
    #     block_inplanes=[64, 128, 256, 512], 
    #     spatial_dims=3, 
    #     n_input_channels=3, 
    #     conv1_t_size=7, 
    #     conv1_t_stride=1, 
    #     no_max_pool=False, 
    #     shortcut_type='B', 
    #     widen_factor=1.0, 
    #     num_classes=400, 
    #     feed_forward=True, 
    #     bias_downsample=True)
    
    # # https://docs.monai.io/en/stable/_modules/monai/networks/nets/efficientnet.html#EfficientNet
    # model = monai.networks.nets.EfficientNet(
    #     blocks_args_str="r1_k3_s22_e1_i32_o16_se0.25",
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     num_classes=n_classes,
    #     )
    
    # model = monai.networks.nets.EfficientNetBN(
    #     "efficientnet-b0",
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     num_classes=n_classes,
    #     pretrained=True
    #     )
    
    # model = monai.networks.nets.SEResNext101(
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     num_classes=n_classes,
    #     pretrained=True
    #     )

    # model = monai.networks.nets.AHNet(
    #     layers=(3, 4, 6, 3), 
    #     spatial_dims=3, 
    #     in_channels=1, 
    #     out_channels=n_classes, 
    #     psp_block_num=4, 
    #     upsample_mode='transpose', 
    #     pretrained=False, 
    #     progress=True)

    return model

def one_hot(x, n_classes):
    o = np.zeros((len(x), n_classes))
    o[range(len(x)), x] = 1
    return o


def show_train_val_curve(epoch, epoch_loss_values, test_loss, train_accuracy, val_accuracy, checkpoint_dir):
    print("Showing figure")
    epochs = np.arange(1, len(epoch_loss_values) +1)
    plt.figure("train/test", (12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Training loss")
    plt.xlabel("epoch")
    plt.plot(epochs, epoch_loss_values)

    plt.subplot(2, 2, 2)
    plt.title("Test loss")
    plt.xlabel("epoch")
    plt.plot(epochs, test_loss)

    plt.subplot(2, 2, 3)
    plt.title("Training accuracy")
    y = train_accuracy
    plt.xlabel("epoch")
    plt.plot(epochs, train_accuracy)

    plt.subplot(2, 2, 4)
    plt.title("Test accuracy")
    plt.xlabel("epoch")
    plt.plot(epochs, val_accuracy)
    plt.suptitle("Ankle Fracture Typing")
    # plt.show()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(checkpoint_dir, str(epoch)+"_Ankle_curve.png"))
    print("Saving curve figure")
    plt.close()
    

def show_confusion_matrix(epoch, n_classes, y_test, y_score, checkpoint_dir):
    
    # classes = ['0', '1', '2', '3']
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cm = confusion_matrix(y_test, y_score, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print("confusion_matrix: ", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        include_values=True,            
        cmap=plt.cm.Blues,                 
        ax=None,                        
        xticks_rotation="horizontal",          
    )
    plt.savefig(os.path.join(checkpoint_dir, str(epoch)+"_Ankle_confusion_matrix.png"), dpi=600)
    print("Saving confusion matrix figure")
    plt.close()
    

def show_ROC_curve(epoch, n_classes, y_test, y_score, checkpoint_dir):
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # print(f"y_test {y_test}\ny_score {y_score}")
    y_test_onehot = one_hot(y_test, n_classes)
    # print(f"y_test_onehot {y_test_onehot}\ny_score {y_score}")

    y_test=np.absolute(np.array(y_test_onehot))
    y_score=np.absolute(np.array(y_score))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # # Plot all ROC curves
    # class_list = ["Normal", "Class A", "Class B", "Class C"]
    class_list = ["Normal", "Class A1", "Class A2", "Class A3", "Class B1", "Class B2", "Class B3", "Class C1", "Class C2", "Others"]
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red', 'green', 'purple','brown', 'pink', 'olive','teal'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of ankle fracture')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(checkpoint_dir, str(epoch)+"_Ankle_fracture_ROC.png"))
    print("Saving ROC figure")
    plt.close()

def train(checkpoint_dir = "."):
    image_label_txt = "../data/trainvaltest_level2_2829598103_image.txt"
    images, labels_list = [], []
    train_img, train_label, test_img, test_label = [], [], [], []
    # random.seed(2023)
    n_classes = 10
    
    with open(image_label_txt) as f:
        for l in f:
            n, l = l.strip().split(" ")
            images.append(n)
            labels_list.append(int(l))
    print("images num: {} labels num: {}".format(len(images), len(labels_list)))

    cnt = {}
    for i in labels_list:
        cnt[i] = labels_list.count(i)
    print("All label with number: {}".format(cnt))

    labels = np.array(labels_list)

    train_files = [{"img": img, "label": label} for img, label in zip(images[:322], labels[:322])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[322:447], labels[322:447])]
    # val_files = [{"img": img, "label": label} for img, label in zip(images[447:], labels[447:])]
    
    train_cnt = {}
    trlabels_list = []
    for tf in train_files:
        trlabels_list.append(tf["label"])
    for i in trlabels_list:
        train_cnt[i] = trlabels_list.count(i)
    print("All train label with number: {}".format(train_cnt))
    
    val_cnt = {}
    vallabels_list = []
    for tf in val_files:
        vallabels_list.append(tf["label"])
    for i in vallabels_list:
        val_cnt[i] = vallabels_list.count(i)
    print("All val label with number: {}".format(val_cnt))

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ]
    )

    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=n_classes)])

    # # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=0, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    # print("check_data img: ", check_data["img"].shape, " check_data label: ", check_data["label"])
    # check_data img:  torch.Size([2, 1, 96, 96, 96])  check_data label:  tensor([2, 2])
    
    # create a training data loader
    # train_ds = ImageDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=False)

    # create a validation data loader
    # val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
    # val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=False)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)
    model = get_model(n_classes=n_classes).to(device)
    logging.info(f"Model: \n{model}")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    # start a typical PyTorch training
    max_epochs = 300
    wait_epoch = 0   # use for stop training early
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    pred_values_test = list()
    label_values_test = list()

    train_loss = list()
    test_loss = list()
    train_accuracy = list()
    val_accuracy = list()
    y_test = list()
    y_score = list()
    
    # # https://docs.monai.io/en/stable/_modules/monai/metrics/confusion_matrix.html#compute_confusion_matrix_metric
    # monai_metrics = {
    #     "sensitivity": ConfusionMatrixMetric(include_background=True, metric_name="sensitivity", compute_sample=False, reduction="mean", get_not_nans=False),
    #     "specificity": ConfusionMatrixMetric(include_background=True, metric_name="specificity", compute_sample=False, reduction="mean", get_not_nans=False),
    #     "accuracy": ConfusionMatrixMetric(include_background=True, metric_name="accuracy", compute_sample=False, reduction="mean", get_not_nans=False),
    #     "precision": ConfusionMatrixMetric(include_background=True, metric_name="precision", compute_sample=False, reduction="mean", get_not_nans=False),
    # }

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        pred_values = list()
        label_values = list()

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            label_values.append(labels.tolist())
            pred_values.append(outputs.argmax(dim=-1).tolist())

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size + 1
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            logging.info(
                    f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
            )
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
        label_values_list = [b for a in label_values for b in a]
        pred_values_list = [b for a in pred_values for b in a]


        train_acc = accuracy_score(y_true=label_values_list, y_pred=pred_values_list)
        train_pre = precision_score(y_true=label_values_list, y_pred=pred_values_list, average='macro', zero_division=0)
        train_recall = recall_score(y_true=label_values_list, y_pred=pred_values_list, average='macro')
        train_f1 = f1_score(y_true=label_values_list, y_pred=pred_values_list, average='macro')

        train_accuracy.append(train_acc)
        print(f"Training metrics:\n \
                accuracy: {train_acc}\n \
                precision: {train_pre}\n \
                f1 score: {train_f1}\n \
                recall: {train_recall}"
                )
        logging.info(f"Training metrics:\n \
                accuracy: {train_acc}\n \
                precision: {train_pre}\n \
                f1 score: {train_f1}\n \
                recall: {train_recall}"
                )

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            num_correct = 0.0
            metric_count = 0
            epoch_loss_val = 0
            val_step = 0

            test_pred = list()
            test_label = list()
            
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_images)
                    y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    
                    for v in val_labels:
                        # print(v.detach().cpu().numpy())
                        y_test.append(v.detach().cpu().numpy())
                    for v in val_outputs:
                        # print(v.detach().cpu().numpy())
                        y_score.append(v.detach().cpu().numpy())
                    
                    test_label.append(val_labels.tolist())
                    test_pred.append(val_outputs.argmax(dim=-1).tolist())
                        
                    val_loss = loss_function(val_outputs, val_labels)
                    epoch_len = len(val_ds) // val_loader.batch_size
                    print(f"{val_step}/{epoch_len}, val_loss: {val_loss.item():.4f}")
                    epoch_loss_val += val_loss.item()
                    
                    val_step += 1
             
                epoch_loss_val /= val_step
                test_loss.append(epoch_loss_val)
                print(f"epoch {epoch + 1} average val loss: {epoch_loss_val:.4f}")
                logging.info(f"epoch {epoch + 1} average val loss: {epoch_loss_val:.4f}")                
                
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                test_label_list = [b for a in test_label for b in a]
                test_pred_list = [b for a in test_pred for b in a]
                # print("test_label_list: ", test_label_list)
                # print("test_pred_list: ", test_pred_list)

                val_acc = accuracy_score(y_true=test_label_list, y_pred=test_pred_list)
                val_pre = precision_score(y_true=test_label_list, y_pred=test_pred_list, average='macro', zero_division=0)
                val_recall = recall_score(y_true=test_label_list, y_pred=test_pred_list, average='macro')
                val_f1 = f1_score(y_true=test_label_list, y_pred=test_pred_list, average='macro')
                # tn, fp, fn, tp = confusion_matrix(label_values_list, pred_values_list).ravel()
                # train_sen = tp / (tp + fn)
                # train_spe = tn / (tn + fp)
                val_accuracy.append(val_acc)
                print(f"Testing metrics:\n \
                        accuracy: {val_acc}\n \
                        precision: {val_pre}\n \
                        recall: {val_recall}\n \
                        f1 score: {val_f1}"
                        )
                logging.info(f"Testing metrics:\n \
                        accuracy: {val_acc}\n \
                        precision: {val_pre}\n \
                        recall: {val_recall}\n \
                        f1 score: {val_f1}"
                        )

                # del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    wait_epoch = 0
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                    print("saved new best metric model")
                    logging.info(
                        f"saved new best metric model at best accuracy: {best_metric:.4f}"
                    )
                else:
                    wait_epoch += 1
                    print(
                    "current accuracy: {:.4f} best accuracy: {:.4f} wait_epoch is {}".format(
                        acc_metric, best_metric, wait_epoch))
                    logging.info(f"current accuracy: {acc_metric:.4f} best accuracy: {best_metric:.4f} wait_epoch is {wait_epoch}")

                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, best_metric, best_metric_epoch
                    )
                )
                logging.info(
                    f"current epoch: {epoch + 1} current accuracy: {acc_metric:.4f} best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}"
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        
        # if wait_epoch > 30:
        #     break
                
        show_train_val_curve(epoch, epoch_loss_values, test_loss, train_accuracy, val_accuracy, checkpoint_dir)
        show_confusion_matrix(epoch, n_classes, test_label_list, test_pred_list, checkpoint_dir)
        show_ROC_curve(epoch, n_classes, y_test, y_score, checkpoint_dir)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    logging.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":

    """
    Usage:
        CUDA_VISIBLE_DEVICES=1 python ankle_fracture_class_285_45_282_v2.py train --model_folder "class_checkpoints"
        python run_net.py infer --data_folder "ankle_seg_data" --model_folder "checkpoints_0222" # run the inference
        CUDA_LAUNCH_BLOCKING=1 python run_net.py train --data_folder "ankle_seg_data" --model_folder "checkpoints_0222" pipeline
    """

    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument(
        "mode", metavar="mode", default="train", choices=("train", "infer"), type=str, help="mode of workflow"
    )
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="runs", type=str, help="model folder")
    args = parser.parse_args()

    monai.utils.set_determinism(seed=2024)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # cdir = os.path.join(args.model_folder, current_time)
    cdir = os.path.join(args.model_folder, "Densenet169_2829598103_level2_image"+"_"+current_time)
    os.makedirs(cdir, exist_ok=True)
    print("checkpoint_dir: ", cdir)

    monai.config.print_config()
    logging.basicConfig(
        # stream=sys.stdout,
        level=logging.INFO,
        filename=os.path.join(cdir, "log.txt"),
        filemode='a')

    if args.mode == "train":
        # data_folder = args.data_folder or os.path.join("ankle_seg_data", "imagesTr")
        # train(data_folder=data_folder, model_folder=args.model_folder)
        train(checkpoint_dir = cdir)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join("ankle_seg_data", "imagesTs")
        infer(data_folder=data_folder, model_folder=args.model_folder, prediction_folder=os.path.join(args.model_folder,"predict"))
    else:
        raise ValueError("Unknown mode.")
