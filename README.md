# HSF-Net: Hybrid Spatial and Frequency Domain Transformer Network for Polyp Segmentation
# Abstract
Accurate polyp segmentation significantly contributes to the early diagnosis and prompt
clinical intervention of colorectal cancer. However, most existing methods mainly con-
centrate on pixel-level feature learning within the spatial domain, while overlooking in-
formative frequency-domain cues and ignoring the cross-scale dependencies among pixels.
To this end, we propose a novel Hybrid Spatial and Frequency Domain Transformer Net-
work (HSF-Net) for polyp segmentation. The novelty of our work is threefold. First, we
design a Frequency-aware Feature Extraction (FFE) module that explicitly decomposes
features into high- and low-frequency components. This allows the model to explicitly
capture high-frequency boundary details and low-frequency global structures, leading to
more precise polyp representations. Second, to model cross-scale dependencies, we intro-
duce a Hierarchical Feature Learning (HFL) module to adaptively fuse features across
multiple encoder stages. Finally, a Semantic Bridging Attention (SBA) module is pro-
posed, which leverages semantic priors from HFL or low-level encoder features as guid-
ance. It adaptively calibrates the low-level, high-resolution features propagated via skip
connections, effectively bridging the semantic gap between deep frequency-domain rep-
resentations and shallow spatial details. Extensive experiments on five public datasets
demonstrate the superiority of our approach. It achieves an mDice of 0.932 on Kvasir-SEG
and 0.818 on ETIS, outperforming 11 state-of-the-art techniques.

# The overall framework of HFSNet
![The overall framework of HFSNet](figtures/Fig.2.png)


# 1. Create environment
OS: Ubuntu 20.04
Python: 3.9
CUDA: 11.7

- Create conda environment with following command conda create -n HSFNet python=3.9
- Activate environment with following command conda activate HSFNet
- Install requirements with following command pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

# 2. Prepare datasets
- Download Train/Test datasets from following [Dataset](https://github.com/DengPingFan/PraNet)
- PvTv2's pretrained weights can be downloaded from [Baidu Drive](https://pan.baidu.com/s/102okWTGyitsohp81ZaleZw?pwd=eg8n)(Code: eg8n)
- HSFNet pretrained weights can be download from [Baidu Drive](https://pan.baidu.com/s/1aSJbEu2bab4NbJ2xtMVmHQ)(Code: bh9i)
# 3. Train & Test
```bash
python Train.py
python Test.py
```
# 4. Acknowledgement
We are very grateful for these excellent works [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [UACANet](https://github.com/plemeri/UACANet/tree/main/configs), which have provided the basis for our framework.

# 5. FAQ
Since our paper has not been accepted yet, we have temporarily hidden the core part of our code. Once the paper is accepted, we will release it immediately.
If you have any questions, please feel free to contact us without hesitation (chinjolin@163.com).