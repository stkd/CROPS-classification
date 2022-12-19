Requirements
efficientnet-pytorch==0.7.1
timm==0.6.11
torch==1.12.1+cu113
torchvision==0.13.1+cu113

安裝配置可參考
pip install efficientnet-pytorch==0.7.1
pip install timm==0.6.11
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

Dataset:
將主辦方提供的每一類別的 traning data 解壓縮至 image/data 之中，每一個類別會在裡面產生一個資料夾
再執行python init_datasets.py

主辦方提供的 test data 則解壓縮至 image/private_test 之中
再執行python init_datasets.py --set test

訓練方法
1.Efficient_b4
python train.py --tag Efficient_SAM_CE_default --batch_size 8 --size 380
2.Swinv2
python train.py --tag Swinv2_SAM_CE_ranaug_ocy_25_lr10_3 --batch_size 8 --size 384 --module 'Swinv2'

推論並產生提交結果
python inference.py --img_path ./images/test