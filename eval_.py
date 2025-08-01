import os
import argparse
import tqdm
import sys
import time
import numpy as np

from PIL import Image
from tabulate import tabulate

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *

def evaluate(args):
    if os.path.isdir(args.result_path) is False:
        os.makedirs(args.result_path)

    method = os.path.split(args.pred_root)[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = args.metrics #['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae']
    results = []
    
    if args.verbose is True:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(args.datasets, desc='Expr - ' + method, total=len(
            args.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        datasets = args.datasets

    for dataset in datasets:
        pred_root = os.path.join(args.pred_root, dataset)
        gt_root = os.path.join(args.gt_root, dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        # threshold_Precision = np.zeros((len(preds), len(Thresholds)))
        # threshold_Recall = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))
        
        total_dice = 0.0
        total_iou = 0.0

        if args.verbose is True:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
                preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            # pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            # gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))

            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)).resize((352, 352), Image.BILINEAR))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)).resize((352, 352), Image.BILINEAR))
            
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            
            assert pred_mask.shape == gt_mask.shape

            if pred_mask.max() <= 1.0:
                pred_mask_255 = (pred_mask * 255).astype(np.uint8)
            else:
                pred_mask_255 = pred_mask.astype(np.uint8)
                
            if gt_mask.max() <= 1.0:
                gt_mask_255 = (gt_mask * 255).astype(np.uint8)
            else:
                gt_mask_255 = gt_mask.astype(np.uint8)
            
            gt_bool = (gt_mask_255 > 127).astype(bool)
            pred_bool = (pred_mask_255 > 127).astype(bool)
            
            gt_mask = gt_mask_255.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)
            pred_mask = pred_mask_255.astype(np.float64) / 255
            
            intersection = np.logical_and(gt_bool, pred_bool).sum()
            union = np.logical_or(gt_bool, pred_bool).sum()
            
            dice = (2 * intersection) / (gt_bool.sum() + pred_bool.sum() + 1e-8)
            iou = intersection / (union + 1e-8)
            
            total_dice += dice
            total_iou += iou

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
            
            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)
        
        meanDic = total_dice / len(preds)
        meanIoU = total_iou / len(preds)

        out = []
        for metric in args.metrics:
            out.append(eval(metric))

        result.extend(out)
        results.append([dataset, *result])

        csv = os.path.join(args.result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")

    if args.verbose is True:
        print(tab)
        print("#"*20, "End Evaluation", "#"*20)
        
    return tab

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gt_root', type=str,
                        default="E://dataset//Polyp_TestDataset", help='gt_root')
    parser.add_argument('--pred_root', type=str,
                        default="E://paper_code//HSF-Net", help='pred_root')
    parser.add_argument('--datasets', nargs='+',
                        default=['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'], help='List of dataset names to process')
    parser.add_argument('--result_path', type=str,
                        default="./results", help='results save path')
    parser.add_argument('--metrics',type=list,
                        default=['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm'], help='index of metrics')
    parser.add_argument('--verbose', type=str,
                        default=True, help='results save path')
    args = parser.parse_args()
    start_time = time.time()
    print(f"测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    evaluate(args)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"测试结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"测试运行时间: {execution_time:.4f}秒")

