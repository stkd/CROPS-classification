import argparse
from glob import glob
import cv2
import shutil
import os
import random
import time
from tqdm import tqdm

def imshow(img, title='img'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cls = {
    'others': 0,
    'kale': 1,
    'sweetpotato': 2,
    'betel': 3,
    'mango': 4,
    'bambooshoots': 5,
    'tea': 6,
    'greenonion': 7,
    'papaya': 8,
    'sunhemp': 9,
    'redbeans': 10,
    'soybeans': 11,
    'taro': 12,
    'grape': 13,
    'broccoli': 14,
    'waterbamboo': 15,
    'loofah': 16,
    'litchi': 17,
    'longan': 18,
    'chinesechives': 19,
    'pennisetum': 20,
    'lemon': 21,
    'lettuce': 22,
    'cauliflower': 23,
    'pear': 24,
    'sesbania': 25,
    'custardapple': 26,
    'passionfruit': 27,
    'roseapple': 28,
    'chinesecabbage': 29,
    'greenhouse': 30,
    'onion': 31,
    'asparagus': 32
}

parser = argparse.ArgumentParser()
parser.add_argument('--set', default='train', help='train or test')

t1 = time.time()

if parser.set == 'test':
    if not os.path.exists(f'./images/test'):
            os.mkdir(f'./images/test')

    samples = glob(f'./images/private_test/*.jpg')
    for image_path in tqdm(samples):
        src = image_path.replace('\\','/')
        dst = src.replace('private_test', 'test')

        img = cv2.imread(src)
        img = cv2.resize(img, (512,512))
        cv2.imwrite(dst, img)

elif parser.set == 'train':
    print(f'total | train | val')
    print(f'--------------------')
    
    n_total_file = 0

    for c in cls.keys():
        d = './images'
        os.makedirs(d,exist_ok=True)
        if not os.path.exists(f'{d}/val'):
            os.mkdir(f'{d}/val')

        if not os.path.exists(f'{d}/val/{c}'):
            os.mkdir(f'{d}/val/{c}')

        if not os.path.exists(f'{d}/train'):
            os.mkdir(f'{d}/train')

        if not os.path.exists(f'{d}/train/{c}'):
            os.mkdir(f'{d}/train/{c}')
        
        samples = glob(f'./images/data/{c}/*.jpg')
        n_val_sample = len(samples) // 5
        n_train_sample = len(samples) - n_val_sample

        print(f'{len(samples):5d} | {n_train_sample:5d} | {n_val_sample:5d}')

        random.shuffle(samples)

        for image_path in samples[:n_train_sample]:
            src = image_path.replace('\\','/')
            dst = dst.replace('data', 'train')
            
            img = cv2.imread(src)
            img = cv2.resize(img, (512,512))
            cv2.imwrite(dst, img)
        
        for image_path in samples[n_train_sample:]:
            src = image_path.replace('\\','/')
            dst = dst.replace('data', 'val')

            shutil.copy(src, dst)

        n_total_file += (n_train_sample + n_val_sample)
    

t2 = time.time()
print(f'--------------------')
print(f'total time: {t2 - t1}')
print(f'total file: {n_total_file}')