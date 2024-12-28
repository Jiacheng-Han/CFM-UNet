<div align="center">
<h1>CFM-UNet </h1>
<h3>Coupling Local and Global Feature Extraction Networks for Medical Image Segmentation Models</h3>
</div>

**0. Main Environments.** </br>
The environment installation procedure can be followed the steps below (python=3.10):</br>

```
conda create -n cfmunet python=3.10
conda activate cfmunet
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d
pip install mamba_ssm
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.** </br>
We provide here all the datasets used for training, testing, and validation (all standardized into PNG format and passwords are "1234"), specifically including the following aspects:

For human organs and tumor lesions, we segmented the classic liver dataset LiTS17, randomly selecting 1,030 liver slices and 670 liver tumor slices. To verify the model's generalization ability, we conducted transfer testing on 800 liver slices randomly selected from the open-source liver dataset ATLAS from MICCAI 2023.
- [LiTS17_Liver.zip](https://pan.baidu.com/s/1QXnJ6UqoXcEeW1XPTyVpPA)
- [LiTS17_Liver_tumor.zip](https://pan.baidu.com/s/1muaDBL8e8lV3gmTaO5uKNQ)
- [ATLAS.zip](https://pan.baidu.com/s/1nuFmaIp5JaQ62Qehxvf9-A)

For the human skeleton, we segmented the spinal skeleton from the open-source dataset SPIDER, randomly selecting 1,000 intact spinal slices. 

- [Kvasir-SEG.zip](https://pan.baidu.com/s/1-JYhJqagx5Q3dQEgIhEuFg)

For human tissues, we segmented all colon polyp images from the open-source dataset Kvasir-SEG.

- [SPIDER.zip](https://pan.baidu.com/s/1yG04Dk6aOEU3fUm3M-0P0Q)

**2. Train the CFM-UNet** </br>
You can try using the model in `CFM-UNet.py`.

## Acknowledgement
Thanks to [VMamba](https://github.com/MzeroMiko/VMamba) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their outstanding work.
