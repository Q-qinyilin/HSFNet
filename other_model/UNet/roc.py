import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset,PolypDataset
from torch.utils.data import DataLoader

class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics

def evaluate(pred, gt, th):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= th).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= th).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    # pred = torch.sigmoid(pred)
    # pred = pred >= 0.5
    # pred_np = pred.reshape(1, -1).data.to("cpu").numpy()[0].astype(int)
    # target_np = gt.reshape(1, -1).data.to("cpu").numpy()[0].astype(int)

    # TP = len(np.where((pred_np == target_np) & (pred_np == 1))[0])
    # FP = len(np.where((pred_np == 1) & (pred_np != target_np))[0])
    # TN = len(np.where((pred_np == target_np) & (pred_np == 0))[0])
    # FN = len(np.where((pred_np == 0) & (pred_np != target_np))[0])


    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    # recall == sensitivity
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    mIoU = (IoU_poly + IoU_bg) / 2.0

    #Dice
    Dice = (2 * TP)/(2*TP + FN + FP)
    
    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, mIoU, Dice

def predict_img(net,thre):
    net.eval()

    db_test = PolypDataset(args.test_img, args.test_gt ,augmentations=False,transform=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=0)
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall','IoU','mIoU', 'Dice'])

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, gt, name= sampled_batch["image"], sampled_batch["gt"],sampled_batch['name']
        image, gt = image.cuda(), gt.cuda()
        output = net(image)

        _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall,_IoU_poly, _mIoU, _Dice = evaluate(output, gt, thre)
        metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU=_IoU_poly,mIoU=_mIoU, Dice = _Dice
                        )

        probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        result = mask_to_image(full_mask)
        # result.save("./result/CVC_ClinicDB/{}".format(name[0]))
    metrics_result = metrics.mean(len(testloader))
    return metrics_result['Dice']

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./vision/EndoScene/2022-07-03-17_23/2022-07-03-17_23best_epoch.pth',   # EdoScenebest_epoch
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    
    parser.add_argument('--test_img', type=str,
                    default=r'data/EndoScene/test/images', help='root dir for test_image')  # data/kvasir/Test_Folder/images
    parser.add_argument('--test_gt', type=str,
                    default=r'data/EndoScene/test/masks', help='root dir for test_gt')  # data/kvasir/Test_Folder/masks

    parser.add_argument('--output', type=str,
                    default=r'result/CVC-ClinicDB/', help='Filenames of ouput images') 

    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.test_img
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    thre=[0.01,0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0]
    b=[]
    for i in thre:
        dice=predict_img(net=net,thre=i)
        b.append(round(dice,4))
    print(b)

