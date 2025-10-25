import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tracemalloc import start
from thop import clever_format
from thop import profile
from ptflops import get_model_complexity_info
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchstat import stat

import time
from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset ,PolypDataset
from torch.utils.data import DataLoader, random_split

# dir_img = 'data/kvasir/Train_Folder/image/'
# dir_mask = 'data/kvasir/Train_Folder/mask/'
# val_img = 'data/kvasir/Val_Folder/image/'
# val_mask = 'data/kvasir/Val_Folder/mask/'

dir_img = '/home/qinyl/qinyilin/Polyp_dataset/TrainDataset/images/'
dir_mask = '/home/qinyl/qinyilin/Polyp_dataset/TrainDataset/masks/'
val_img = '/home/qinyl/qinyilin/Polyp_dataset/TestDataset/Kvasir/images/'
val_mask = '/home/qinyl/qinyilin/Polyp_dataset/TestDataset/Kvasir/masks/'

# dir_img = 'data/CVC-ClinicDB/train/images/'
# dir_mask = 'data/CVC-ClinicDB/train/images/'
# val_img = 'data/CVC-ClinicDB/test/images/'
# val_mask = 'data/CVC-ClinicDB/test/masks/'

# dir_img = 'data/CVC_ClinicDB/train/images/'
# dir_mask = 'data/CVC_ClinicDB/train/masks/'
# val_img = 'data/CVC_ClinicDB/test/images/'
# val_mask = 'data/CVC_ClinicDB/test/masks/'


def mkdir_time_dir():
    time_now = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
    path ="./vision/TrainDataset/{}".format(time_now)
    if not os.path.exists(path):
        os.makedirs(path)
        print("tensorboard dir mkdir success")
    return path

def train_net(net,
              device,
              epochs=5,
              batch_size=8,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    train_data = PolypDataset(image_root=dir_img, gt_root=dir_mask, augmentations=False,transform=True)
    val_data = PolypDataset(image_root=val_img, gt_root=val_mask, augmentations=False,transform=True)
    n_val = len(val_data)
    n_train = len(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    save_tensorboard=mkdir_time_dir()
    writer = SummaryWriter(log_dir=save_tensorboard,flush_secs=30)
    graph_inputs = torch.from_numpy(np.random.rand(1,3,224,224)).type(torch.FloatTensor).cuda()
    writer.add_graph(net, (graph_inputs,))

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    iter_num = 0
    val_s=0
    epoch_loss=[]
    for epoch in tqdm(range(1,epochs),desc="Epoch",total=epochs):
        since = time.time()
        net.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            imgs = batch['image']
            true_masks = batch['gt']
            # print("imgs:",imgs.size())
            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            # epoch_loss += loss.item()
            iter_num = iter_num + 1
            epoch_loss+=loss.item()
            writer.add_scalar('Loss/train', loss.item(), iter_num)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            global_step += 1

            writer.add_scalar('loss', loss, i)


        val_score = eval_net(net, val_loader, device)
        # scheduler.step(val_score)
        # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

        logging.info('Validation Dice Coeff: {}'.format(val_score))
        writer.add_scalar('Dice/val', val_score, (epoch))

            # writer.add_images('images', imgs, global_step)
            # if net.n_classes == 1:
            #     writer.add_images('masks/true', true_masks, global_step)
            #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        writer.add_scalar('Loss/epoch', np.mean(epoch_loss), (epoch))
        epoch_loss=[]

        if val_score>val_s:
            val_s=val_score
            try:
                os.mkdir(save_tensorboard)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            
            torch.save(net.state_dict(),
                       save_tensorboard +'/'+ f'best_epoch.pth')
            

        if epoch%10==0:
            try:
                os.mkdir(save_tensorboard)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            
            torch.save(net.state_dict(),
                       save_tensorboard +'/'+ f'CP_epoch{epoch + 1}.pth')
            
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='False',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(filename=r"./list/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    flops, params = get_model_complexity_info(net, (3,352,352), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)
    # stat(net,(3,224,224))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
