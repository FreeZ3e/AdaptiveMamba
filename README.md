# AdaptiveMamba
Implementation of AdaptiveMamba

## Env

Create env:
```
conda create -n adaptive_mamba python=3.10 -y
conda activate adaptive_mamba
```
Install pytorch>=2.3, cuda>=12.1, then install dependencies via pip:
```
pip install -r requirements.txt
cd kernel/selective_scan && pip install .
```

Dependencies for detection and segmentation tasks:
```
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```


## Model training and inference:

### Classification:

Train:
```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
```

Evaluate:
```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --pretrained </path/of/checkpoint>
```

### Detection and Segmentation:

Train:
```
bash ./tools/dist_train.sh </path/to/config> 8
```

Evaluate:
```
bash ./tools/dist_test.sh </path/to/config> </path/to/checkpoint> 1
```
use --tta to get mIoU(ms) in segmentation
