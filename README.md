# HSF-Net: Hybrid Spatial and Frequency Domain Transformer Network for Polyp Segmentation
# 1. Create environment
- Create conda environment with following command conda create -n HSFNet python=3.9
- Activate environment with following command conda activate HSFNet
- Install requirements with following command pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

# 2. Prepare datasets
- Download Train/Test datasets from following [Dataset](https://github.com/DengPingFan/PraNet)
- PVTv2's pretrained weights can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/102okWTGyitsohp81ZaleZw?pwd=eg8n)(code: eg8n)
- HSFNet pretrained weights can be download from [here](https://pan.baidu.com/s/1aSJbEu2bab4NbJ2xtMVmHQ)(Code: bh9i)
# 3. Train & Test
```bash
python Train.py
python Test.py
```
# 4. Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [UACANet](https://github.com/plemeri/UACANet/tree/main/configs), which have provided the basis for our framework.

# 5. FAQ:
If you want to improve the usability or any piece of advice, please feel free to contact me directly (chinjolin@163.com).