#### OccamNets v1
This is the repository for our new paper [OccamNets](https://arxiv.org/abs/2204.02426). In this paper, we apply Occam's razor to neural networks to use only the required network depth and required visual regions. This increases bias robustness.

#### Install the dependencies
`./requirements.sh`

#### Configuration:
- Specify the root directory (where the dataset/logs will be stored) in the `paths.root` entry inside `conf/base_config.yaml`

### Instructions for each dataset
#### BiasedMNIST
- Download BiasedMNIST from: https://drive.google.com/file/d/1_77AKsY5MoYpDnXgNkjWi9n2_mfQBW-F/view?usp=sharing
- Provide the full path for Biased MNIST in `data_dir` inside `conf/dataset/biased_mnist.yaml`
- You can also generate Biased MNIST by using/modifying: `./scripts/biased_mnist/generate.sh`

#### COCO-on-Places
- Download the dataset from: https://github.com/Faruk-Ahmed/predictive_group_invariance
- Specify the location to the dataset in `data_dir` of `conf/dataset/coco_on_places.yaml`

#### Training Scripts
-  We provide bash scripts to train OccamResNet and ResNet (including baselines and SoTA debiasing methods on both the architectures)
    - Train baseline and SoTA methods on OccamResNet/ResNet using: `./scripts/{dataset}/{dataset_shortform}_{method}.sh`
    - E.g., To train `./scripts/biased_mnist/bmnist_occam.sh` trains OccamNet with BiasedMNIST

### Relevant files for OccamNets
- Model definition: Find OccamNets in `models/occam_resnet.py`, `occam_efficient_net.py` and `occam_mobile_net.py`. 
- Training script: `trainers/occam_trainer.py`. 
- Training configuration: `conf/trainer/occam_trainer.yaml` (all of these parameters can be overridden from command line)

#### Citation
```
@article{shrestha2022occamnets,
  title={OccamNets: Mitigating Dataset Bias by Favoring Simpler Hypotheses},
  author={Shrestha, Robik and Kafle, Kushal and Kanan, Christopher},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
