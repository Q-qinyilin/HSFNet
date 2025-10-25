import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# from unet import UNet
from unet.unetpp import NestedUNet
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

def predict_img(net,device):
    net.eval()

    # db_test = PolypDataset(args.test_img, args.test_gt ,augmentations=False,transform=True)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=0)
    # metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
    #                    'ACC_overall','IoU','mIoU', 'Dice'])
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = '{}/{}'.format(args.test_img,_data_name)
        save_path = './result_map/UNet++/{}/'.format(_data_name)
        os.makedirs(save_path,exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        db_test = PolypDataset(image_root, gt_root ,augmentations=False,transform=True)
        testloader = DataLoader(db_test, batch_size=1, shuffle=True, num_workers=0)

        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, gt, name= sampled_batch["image"], sampled_batch["gt"],sampled_batch['name']
            image, gt = image.cuda(), gt.cuda()
            output = net(image)
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
            result.save(f"{save_path}/{name[0]}")
            # result.save("./result/EndoScene/{}".format(name[0]))

    # metrics_result = metrics.mean(120)
    # print("Test Result:")
    # print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
    #       'ACC_overall: %.4f, IoU: %.4f,mIoU: %.4f, Dice:%.4f'
    #       % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
    #          metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],metrics_result['IoU'],
    #           metrics_result['mIoU'], metrics_result['Dice']))

    # return full_mask > out_threshold, name


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/Train3/best_epoch.pth',   # EdoScenebest_epoch
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    
    parser.add_argument('--test_img', type=str,
                    default=r'/home/qinyl/qinyilin/Polyp_dataset/TestDataset', help='root dir for test_image')  # data/kvasir/Test_Folder/images
    parser.add_argument('--test_gt', type=str,
                    default=r'data/EndoScene/Test/masks', help='root dir for test_gt')  # data/kvasir/Test_Folder/masks

    # parser.add_argument('--input', '-i', default='./best_epoch.pth',metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)

    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
    #                     help='Filenames of ouput images')

    parser.add_argument('--output', type=str,
                    default=r'result/EndoScene/', help='Filenames of ouput images') 

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

'''
def get_output_filenames(args):
    # in_files = args.input
    in_files = args.test_img
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files
'''

def mask_to_image(mask):
    mask = (mask > 0.5).astype(np.uint8)  # 设置阈值为 0.5
    return Image.fromarray((mask * 255))
    # return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.test_img
    # out_files = get_output_filenames(args)

    net = NestedUNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")



    predict_img(net=net,device=device)

'''        if not args.no_save:
            # out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(name[0])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
'''