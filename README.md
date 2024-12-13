# CSE_597_rpp5524
This is a github repository for CSE 597 Final Project

## GRIT: Faster and Better Image captioning Transformer (ECCV 2023)

This is the code implementation for the paper titled: "GRIT: Faster and Better Image-captioning Transformer Using Dual Visual Features" (Accepted to ECCV 2022) [[Arxiv](https://arxiv.org/abs/2207.09666)].


## Introduction

This paper proposes a Transformer neural architecture, dubbed <b>GRIT</b> (Grid- and Region-based Image captioning Transformer), that effectively utilizes the two visual features to generate better captions. GRIT replaces the CNN-based detector employed in previous methods with a DETR-based one, making it computationally faster.


<div align=center>  
<img src='.github/grit.png' width="100%">
</div>

## Installation

### Requirements
* Python >= 3.9, CUDA >= 11.3
* PyTorch >= 1.12.0, torchvision >= 0.6.1
* Other packages: pycocotools, tensorboard, tqdm, h5py, nltk, einops, hydra, spacy, and timm

* First, clone the repository locally:
```shell
git clone https://github.com/davidnvq/grit.git
cd grit
```
* Then, create an environment and install PyTorch and torchvision:
```shell
conda create -n grit python=3.9
conda activate grit
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# ^ if the CUDA version is not compatible with your system; visit pytorch.org for compatible matches.
```
* Install other requirements:
```shell
pip install -r requirements.txt
python -m spacy download en
```
* Install Deformable Attention:
```shell
cd models/ops/
python setup.py build develop
python test.py
```

## Usage


### Data preparation

Download and extract COCO 2014 for image captioning including train, val, and test images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco_caption/
├── annotations/  # annotation json files and Karapthy files
├── train2014/    # train images
├── val2014/      # val images
└── test2014/     # test images
```
* Copy the files in `data/` to the above `annotations` folder. It includes `vocab.json` and some files containing Karapthy ids.

### Training

The model is trained with default settings in the configurations file in `configs/caption/coco_config.yaml`:
The training process takes around 16 hours on a machine with 8 A100 GPU.
We also provide the code for extracting pretrained features (freezed object detector), which will speed up the training significantly.

* With default configurations (e.g., 'parallel Attention', pretrained detectors on VG or 4DS, etc):
```shell
export DATA_ROOT=path/to/coco_dataset
python train_caption.py exp.name=caption_4ds
```

<!-- * **More configurations will be added here for obtaining ablation results**. -->
* To freeze the backbone and detector, we can extract the region features and initial grid features first, saving it to `dataset.hdf5_path` in the config file.

**Noted that: this additional strategy will only achieve about 134 CIDEr (as reported by some researchers). To obtain 139.2 CIDEr, please train the model with freezed backbone/detector (in Pytorch, using `if 'backbone'/'detector' in n: p.requires_grad = False`) with image augmentation at every iteration. It means that we read and process every image during training rather than loading `extracted features` from hdf5.**

Then we can run the following script to train the model:
```shell
export DATA_ROOT=path/to/coco_dataset
python train_caption.py exp.name=caption_4ds \
optimizer.freezing_xe_epochs=10 \
optimizer.freezing_sc_epochs=10 \
optimizer.finetune_xe_epochs=0 \
optimizer.finetune_sc_epochs=0 \
optimizer.freeze_backbone=True \
optimizer.freeze_detector=True
```

### Evaluation

The evaluation will be run on a single GPU.
* Evaluation on **Karapthy splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```
* Evaluation on the **online splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption_online.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption_online.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```

### Inference on RGB Image

* Perform Inference for a single image using the script `inference_caption.py`:
```
python inference_caption.py +img_path='notebooks/COCO_val2014_000000000772.jpg' \
+vocab_path='data/vocab.json' \
exp.checkpoint='path_to_caption_checkpoint'
```

