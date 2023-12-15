## GroundedSAM for OccNeRF

In order to generate per-pixel semantic labels for [OccNeRF](https://github.com/LinShan-Bin/OccNeRF) without any 3D ground truth, we leverage [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and modify it to generate semantic labels from 2D image training data.

**Updates**

- **`🔔 2023/12/14`** Release the code as a data preparing tool for [OccNeRF](https://github.com/LinShan-Bin/OccNeRF)

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Prepare environment:

```bash
git clone git@github.com:JunchengYan/GroundedSAM_OccNeRF.git
cd GroundedSAM_OccNeRF/
pip install -r requirements.txt
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install GroundingDINO:

```bash
python -m pip install -e GroundingDINO
```

Other dependency

```bash
pip install diffusers transformers accelerate scipy safetensors
```

## Run the code

### Step1: Download the pretrained weights

Prepare weight for Segment Anything:

```bash
# Place it under GroundedSAM_OccNeRF/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
```

### Step2: Prepare the models

Due to the unstable Internet connection, we do not follow the instruction of [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam.ipynb), which downloads the GroundiongDINO and BERT model using [Hugging Face](https://huggingface.co/ShilongLiu/GroundingDINO). We download the model from Hugging Face in advance and modified the code to load from local.

#### Load models from local

You can download the GroundiongDINO model from [here](https://drive.google.com/file/d/15Klhb1t-3KpeOqKVtualJG5NzbZtcf9P/view?usp=drive_link), and BERT model from [here](https://drive.google.com/file/d/1J2ZghAX1bBHA1gSCklXJcMS1xNxcYknU/view?usp=drive_link). Then unzip the file under `GroundedSAM_OccNeRF/`

```bash
unzip models--ShilongLiu--GroundingDINO.zip
unzip models--bert-base-uncased.zip
```

#### Load models from Hugging Face

You can also change the function `load_model_hf()` in `groundedsam_generate_sem_demo.py` and `groundedsam_generate_sem_nusc.py` to the original code in [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam.ipynb) to load GroundingDINO from Hugging Face. You should also change the function `get_pretrained_language_model()` in `GroundingDINO/util/get_tokenlizer.py` back to the code in [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/GroundingDINO/groundingdino/util/get_tokenlizer.py) to load BERT

### Step3: Generate semantic labels

#### Generate semantic labels on Nuscenes trainset

**Prepare data**

If you are using GroundedSAM_OccNeRF **individually**, please link your nuscenes dataset path to the `GroundedSAM_OccNeRF/` folder and download metadata according to [OccNeRF](https://github.com/LinShan-Bin/OccNeRF), then modify the `data_path` in `groundedsam_generate_sem_nusc.py`.

```bash
ln -s DATA_PATH ./data
```

We use `groundedsam_generate_sem_nusc.py` to generate semantic labels for OccNeRF self-supervised occupancy learning. Modify  `sava_path` in `groundedsam_generate_sem_nusc.py` to determine where to save the results. We need about 36 hours to generate semantic masks for Nuscenes trainset on 8 RTX3090 GPUs.

```bash
bash run.sh
```

#### Generate semantic labels on any data

We use `groundedsam_generate_sem_demo.py` to generate semantic masks of any picture on a single GPU. You can modify `data_path` and `sava_path` in `groundedsam_generate_sem_demo.py` according to your need. The pictures under `bikes/` can be used for test.

```bash
bash infer.sh
```


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{chubin2023occnerf, 
      title   = {OccNeRF: Self-Supervised Multi-Camera Occupancy Prediction with Neural Radiance Fields}, 
      author  = {Chubin Zhang and Juncheng Yan and Yi Wei and Jiaxin Li and Li Liu and Yansong Tang and Yueqi Duan and Jiwen Lu},
      journal = {arXiv preprint arXiv:TODO},
      year    = {2023}
}
```
