Competition URL:https://aidea-web.tw/topic/5f632f38-7213-4d4d-bea3-117ff13c1afb \
Private Leaderboard: 20 / 446

# Requirements
efficientnet-pytorch==0.7.1
timm==0.6.11
torch==1.12.1+cu113
torchvision==0.13.1+cu113

## Install package
``` bash
pip install efficientnet-pytorch==0.7.1
pip install timm==0.6.11
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
# Dataset preprocessing
- download the traning dataset from the [official](https://aidea-web.tw/topic/5f632f38-7213-4d4d-bea3-117ff13c1afb). and unzip into image/data, Then run the following command to spilt all data into train/valid sets.
``` bash
python init_datasets.py
```

# Inference
- First, download the test dataset from the [official](https://aidea-web.tw/topic/5f632f38-7213-4d4d-bea3-117ff13c1afb). and unzip into image/private_test,Then run the following command.
``` bash
python init_datasets.py --set test
```
- Download checkpoint from [here](https://drive.google.com/drive/folders/1g1_4I2s8E9UScJC6pfOUB-NuJm9ZkGOh?usp=share_link), unzip and put into cache, Then run the following command to get inference result 
``` bash
python inference.py --img_path ./images/test
```

# Train
## 1.Efficient_b4
``` bash
python train.py --tag Efficient_SAM_CE_default --batch_size 8 --size 380
```
## 2.Swinv2
``` bash
python train.py --tag Swinv2_SAM_CE_ranaug_ocy_25_lr10_3 --batch_size 8 --size 384 --module 'Swinv2'
```

# Cite
```bibtex
@inproceedings{foret2021sharpnessaware,
  title={Sharpness-aware Minimization for Efficiently Improving Generalization},
  author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=6Tm1mposlrM}
}
```

```bibtex
@inproceesings{pmlr-v139-kwon21b,
  title={ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks},
  author={Kwon, Jungmin and Kim, Jeongseop and Park, Hyunseo and Choi, In Kwon},
  booktitle ={Proceedings of the 38th International Conference on Machine Learning},
  pages={5905--5914},
  year={2021},
  editor={Meila, Marina and Zhang, Tong},
  volume={139},
  series={Proceedings of Machine Learning Research},
  month={18--24 Jul},
  publisher ={PMLR},
  pdf={http://proceedings.mlr.press/v139/kwon21b/kwon21b.pdf},
  url={https://proceedings.mlr.press/v139/kwon21b.html},
  abstract={Recently, learning algorithms motivated from sharpness of loss surface as an effective measure of generalization gap have shown state-of-the-art performances. Nevertheless, sharpness defined in a rigid region with a fixed radius, has a drawback in sensitivity to parameter re-scaling which leaves the loss unaffected, leading to weakening of the connection between sharpness and generalization gap. In this paper, we introduce the concept of adaptive sharpness which is scale-invariant and propose the corresponding generalization bound. We suggest a novel learning method, adaptive sharpness-aware minimization (ASAM), utilizing the proposed generalization bound. Experimental results in various benchmark datasets show that ASAM contributes to significant improvement of model generalization performance.}
}
```
