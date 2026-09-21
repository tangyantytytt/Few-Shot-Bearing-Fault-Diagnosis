# Few-Shot Bearing Fault Diagnosis Method via Fusion of Attention and Mahalanobis Distance

PyTorch implementation of the paper "Few-Shot Bearing Fault Diagnosis Method via Fusion of Attention and Mahalanobis Distance".

## Requirements
pip install -r requirements.txt

## Training

### CWRU (1-shot)
python train_1shot.py --dataset CWRU --model_name proposed_model --training_samples_CWRU 30 --episode_num_train 100

### CWRU (5-shot)
python train_5shot.py --dataset CWRU --model_name proposed_model_5shot --training_samples_CWRU 60 --episode_num_train 100

## Testing

python test_1shot.py --dataset CWRU --best_weight checkpoints/proposed_model_1shot_xxx.pth

## Contact
3081975610@qq.com
