import argparse
import pathlib

import cv2
import torch
from tqdm import tqdm
import numpy as np
from glob import glob
import csv

from timm.utils import accuracy
from torchvision import transforms

cls = [
    'asparagus',
    'bambooshoots',
    'betel',
    'broccoli',
    'cauliflower',
    'chinesecabbage',
    'chinesechives',
    'custardapple',
    'grape',
    'greenhouse',
    'greenonion',
    'kale',
    'lemon',
    'lettuce',
    'litchi',
    'longan',
    'loofah',
    'mango',
    'onion',
    'others',
    'papaya',
    'passionfruit',
    'pear',
    'pennisetum',
    'redbeans',
    'roseapple',
    'sesbania',
    'soybeans',
    'sunhemp',
    'sweetpotato',
    'taro',
    'tea',
    'waterbamboo'
]

class Pred:
    """
    fuse with infrared folder and visible folder
    """

    def __init__(self, model: dict, checkpoint_path: str):
        """
        :param model_path: path of pre-trained parameters
        """
        device = 'cpu'
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.model_weight = [0.7,0.3]

    def __call__(self, folder: str, file_list: str, output_file: str):

        file_list = list(csv.DictReader(open(file_list,'r')))

        output_file = open(f'{output_file}.csv','w')
        output_file.write(f'filename,label\n')

        outputs = {}
        # cnt_correct = 0

        for num in range(len(model)):
            modelnm = torch.load(self.checkpoint_path+f'/{model[num]}/model.pth').to(self.device)
            modelnm.eval()
            self.modelnm = modelnm
            
            for row in tqdm(file_list):
                
                path = pathlib.Path(folder + row['filename'])
                # label = row['label']

                image = self._imread(str(path),model[num])
                image = (image.unsqueeze(0)).to(self.device)

                # network forward
                with torch.no_grad():
                    output = self.modelnm(image)
                    _, pred = output.topk(1, 1, True, True)
                    output = output * self.model_weight[num]
                if num == 0:
                    outputs[row['filename']] = output
                else:
                    outputs[row['filename']] = outputs[row['filename']] + output
            
        for row in tqdm(file_list):
            _, pred = outputs[row['filename']].topk(1, 1, True, True)
            output_file.write(f'{row["filename"]},{cls[int(pred[0])]}\n')

    @staticmethod
    def _imread(path: str, modelnm: str) -> torch.Tensor:
        image_cv = cv2.imread(path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((384,384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if "Eff" in modelnm:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((380,380)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image_ts = transform(image_cv)
        
        return image_ts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='./images/private_test', help='image for test')
    parser.add_argument('--csv_path', default='./images/submission_example.csv', help='sample csv for test')
    parser = parser.parse_args()
    
    model = ['Swinv2_SAM_CE_ranaug_ocy_25_lr10_3','Efficient_SAM_CE_default']
    m = Pred(model, f'./cache')
    m(parser.img_path, parser.csv_path, 'result703')
