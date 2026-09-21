# Few-Shot Bearing Fault Diagnosis Method via Fusion of Attention and Mahalanobis Distance

PyTorch implementation of the paper "Few-Shot Bearing Fault Diagnosis Method via Fusion of Attention and Mahalanobis Distance".

## Requirements
pip install -r requirements.txt

## Hardware Requirements
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- 16GB+ GPU memory recommended

## Dataset Preparation

### CWRU Dataset
The CWRU dataset will be automatically downloaded when running the training script. The download links are provided in `metadata.txt`.

### PU Dataset
1. Download from: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download
2. Extract all .rar files
3. Organize as: Datasets/PU/K001/, Datasets/PU/KA01/, etc.

## Training

### CWRU (1-shot)
python train_1shot.py --dataset CWRU --model_name proposed_model --training_samples_CWRU 30 --episode_num_train 100

### CWRU (5-shot)
python train_5shot.py --dataset CWRU --model_name proposed_model_5shot --training_samples_CWRU 60 --episode_num_train 100

## Testing

python test_1shot.py --dataset CWRU --best_weight checkpoints/proposed_model_1shot_xxx.pth

## Contact
3081975610@qq.com
