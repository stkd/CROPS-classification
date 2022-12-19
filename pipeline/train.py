import csv
import logging
import os
from pathlib import Path

from utils import wp_score
import torch
from timm.utils import AverageMeter, accuracy
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import autoaugment
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from model import Detector


class Train:

    def __init__(self, config: dict, device: str):
        logging.info(f'Training')
        self.num_classes = config.num_classes
        self.config = config
        self.device = device

        # modules
        self.model = Detector(config.module, num_classes=33, rho = 0.05, learning_rate = config.learning_rate, momentum = config.momentum, label_smoothing = config.label_smoothing)
        self.model.to(device)

        # loss
        self.loss = CrossEntropyLoss()
        self.loss.to(device)

        # datasets
        train_set = Path(config.folder + '/train')
        val_set = Path(config.folder + '/val')

        if config.module == 'Swinv2':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((config.size, config.size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                autoaugment.RandAugment(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((config.size, config.size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            

        val_transform = transforms.Compose([
            transforms.Resize((config.size, config.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ImageFolder(train_set, transform=train_transform)
        self.train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

        val_dataset = ImageFolder(val_set, transform=val_transform)
        self.val_dataloader = DataLoader(val_dataset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model.optimizer, max_lr=config.learning_rate*5, steps_per_epoch=len(self.train_dataloader), epochs=config.epochs)
        
        logging.info(f'dataset | folder: {str(config.folder)} | train size: {len(self.train_dataloader) * config.batch_size} | val size: {len(self.val_dataloader) * config.batch_size}')

    def train(self, inputs, targets):
        self.model.train()

        outputs = self.model.training_step(inputs, targets)
        loss = self.loss(outputs, targets)
        return loss

    @torch.no_grad()
    def eval(self, inputs, targets):
        self.model.eval()

        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        return loss, acc1, acc5, outputs
        

    def run(self):
        best = -float('Inf')
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):

            train_loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            acc1_meter = AverageMeter()
            acc5_meter = AverageMeter()

            train_process = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            for train_sample in train_process:                
                train_images, train_labels = train_sample[0], train_sample[1]

                train_images = train_images.to(self.device)
                train_labels = train_labels.to(self.device)

                loss = self.train(train_images, train_labels)

                train_loss_meter.update(loss.item())
                
                self.scheduler.step()
            
            y_true = torch.tensor([]).type(torch.int16)
            y_pred = torch.tensor([]).type(torch.int16)
            val_process = tqdm(self.val_dataloader, total=len(self.val_dataloader))
            for val_sample in val_process:
                val_images, val_labels = val_sample[0], val_sample[1]

                val_images = val_images.to(self.device)
                val_labels = val_labels.to(self.device)

                loss, acc1, acc5, preds = self.eval(val_images, val_labels)
                
                val_loss_meter.update(loss.item())
                acc1_meter.update(acc1.item())
                acc5_meter.update(acc5.item())

                label = val_labels.cpu().detach()
                pred = preds.cpu().detach()
                y_true = torch.cat((y_true, label), 0)
                y_pred = torch.cat((y_pred, pred), 0)
            
            f1_dict, WP_value = wp_score(y_pred, y_true)


            logging.info(
                f'[{epoch}]  '
                f'Train Loss: {train_loss_meter.avg:.4f} | '
                f'Val Loss: {val_loss_meter.avg:.4f} | '
                f'Acc@1: {acc1_meter.avg:.2f} | '
                f'Acc@5: {acc5_meter.avg:.2f} | '
                f'WP_value: {WP_value:.4f}')

            if epoch % 1 == 0:

                if WP_value > best:
                    best = WP_value
                    best_epoch = epoch

                    self.save(epoch, is_best=True)

                self.save(epoch)
                logging.info(f'best epoch is {best_epoch}, acc is {best}')


    def save(self, epoch: int, is_best = False):
        path = Path(self.config.cache) / self.config.tag
        path.mkdir(parents=True, exist_ok=True)

        if is_best:
            cache = path / f'best.pth'
            torch.save(self.model, path / f'model.pth')

        else:
            cache = path / f'{epoch:03d}.pth'

        torch.save(self.model.state_dict(), cache)
        logging.info(f'save checkpoint to {str(cache)}')