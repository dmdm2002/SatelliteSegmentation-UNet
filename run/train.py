import os
import matplotlib.pyplot as plt
import argparse
import cv2
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from torch.utils.tensorboard import SummaryWriter

from albumentations.core.composition import Compose, OneOf
from tqdm import tqdm

from model.nested_unet import NestedUNet
from model.unet_resnet50 import UNetWithResnet50Encoder
from util.data_loader import CustomDataset
from util.iou import iou_pytorch
import torchvision.transforms as transforms


class Train:
    def __init__(self):
        super().__init__()
        self.root = 'D:/Side/AI_FACTORY/add'
        self.output_root = 'D:/Side/AI_FACTORY/backup/UnetResnet50_newLoader'
        self.test_img = 'D:/Side/AI_FACTORY/backup/UnetResnet50_newLoader/test_img'
        os.makedirs(self.test_img, exist_ok=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        IMG_HEIGHT = 512
        IMG_WIDTH = 512

        self.train_transform = Compose([
            A.Resize(IMG_HEIGHT, IMG_WIDTH),
            OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(), ], p=0.65),
            OneOf([
                A.HueSaturationValue(),
                A.RandomBrightness(),
                A.RandomContrast(), ], p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.val_transform = Compose([
            A.Resize(IMG_HEIGHT, IMG_WIDTH),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.random_seed = 32

    def visualize(self, image, label, seg_image, ep, idx):
        f, ax = plt.subplots(1, 3, figsize=(20, 8))
        ax[0].imshow(image)
        ax[1].imshow(label, cmap='gray')
        ax[2].imshow(seg_image, cmap='gray')

        ax[0].set_title('Original Image')
        ax[1].set_title('Ground Truth')
        ax[2].set_title('UNet')

        ax[0].title.set_size(25)
        ax[1].title.set_size(25)
        ax[2].title.set_size(25)

        f.tight_layout()

        plt.savefig(f'{self.test_img}/{ep}_{idx}_sample.jpg')
        plt.close()

    def run(self):
        print('----------[Load Dataset]----------')
        train_dataset = CustomDataset(self.root, dbs=['Jeju', 'Jeolla'], train=True, transform=self.train_transform)
        val_dataset = CustomDataset(self.root, dbs=['Jeju'], train=False, transform=self.val_transform)

        tr_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        print('----------[Load Model]----------')
        # model = NestedUNet(num_classes=5, deep_supervision=True).to(self.device)
        model = UNetWithResnet50Encoder(n_classes=5).to(self.device)

        print('----------[Optim and Loss]----------')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

        print('----------[Run!!]----------')
        os.makedirs(f"{self.output_root}/ckp", exist_ok=True)
        os.makedirs(f"{self.output_root}/log", exist_ok=True)

        summary = SummaryWriter(f'{self.output_root}/log')
        best_score = {'epoch': 0, 'iou': 0, 'acc': 0, 'loss': 0}
        transform = transforms.Compose([
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        ])

        for ep in range(50):
            tr_iou_score = []
            te_iou_score = []
            tr_loss = 0
            te_loss = 0

            model.train()
            for idx, (img, gt) in enumerate(tqdm(tr_loader, desc=f'[Train {ep}/50]==>')):
                img = torch.tensor(img, device=self.device, dtype=torch.float32)
                gt = torch.tensor(gt, device=self.device, dtype=torch.float32)
                target = gt.argmax(1)

                optimizer.zero_grad()
                logits = model(img)
                loss = criterion(logits, target.long())
                loss.backward()
                optimizer.step()

                tr_iou_score.extend(iou_pytorch(logits.argmax(1), target))
                tr_loss += loss.item()

                if idx % 100 == 0:
                    print(f'miou: {torch.FloatTensor(tr_iou_score).mean()}')

            with torch.no_grad():
                model.eval()
                for idx, (img, gt) in enumerate(tqdm(val_loader, desc=f'[Validation {ep}/50]==>')):
                    img = torch.tensor(img, device=self.device, dtype=torch.float32)
                    gt = torch.tensor(gt, device=self.device, dtype=torch.float32)
                    target = gt.argmax(1)

                    logits = model(img)
                    loss = criterion(logits, target.long())

                    te_iou_score.extend(iou_pytorch(logits.argmax(1), target))
                    te_loss += loss.item()
                    if idx % 10 == 0:
                        gt_values = {
                            0: 10,
                            1: 30,
                            2: 90,
                            3: 70,
                            4: 100
                        }
                        pred = logits.argmax(1)[0]
                        pred = np.vectorize(gt_values.get)(pred.cpu().detach())
                        target = np.vectorize(gt_values.get)(target[0].cpu().detach())
                        img = transform(img[0].squeeze()).permute(1, 2, 0).cpu().detach().numpy()

                        self.visualize(img, target, pred, ep, idx)


            tr_iou_mean = torch.FloatTensor(tr_iou_score).mean()
            tr_loss_mean = tr_loss / len(tr_loader)

            te_iou_mean = torch.FloatTensor(te_iou_score).mean()
            te_loss_mean = te_loss / len(val_loader)

            if best_score['iou'] <= te_iou_mean:
                best_score['epoch'] = ep
                best_score['iou'] = te_iou_mean
                best_score['loss'] = te_loss_mean

            print('\n')
            print('-------------------------------------------------------------------')
            print(f"Epoch: {ep}/50")
            print(f"Train iou: {tr_iou_mean} | Train loss: {tr_loss_mean}")
            print(f"Test iou: {te_iou_mean} | Test loss: {te_loss_mean}")
            print('-------------------------------------------------------------------')
            print(f"Best acc epoch: {best_score['epoch']}")
            print(f"Best iou: {best_score['iou']} | Best loss: {best_score['loss']}")
            print('-------------------------------------------------------------------')

            summary.add_scalar('Train/iou', tr_iou_mean, ep)
            # summary.add_scalar('Train/acc', tr_acc_mean, ep)
            summary.add_scalar('Train/loss', tr_loss_mean, ep)

            summary.add_scalar('Test/iou', te_iou_mean, ep)
            # summary.add_scalar('Test/acc', te_acc_mean, ep)
            summary.add_scalar('Test/loss', te_loss_mean, ep)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.output_root}/ckp", f"{ep}.pth"),
            )


if __name__ == '__main__':
    tr = Train()
    tr.run()