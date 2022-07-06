conda create -y --name occamnets python==3.8
source activate occamnets

# Pytorch + Lightning
pip install pytorch-lightning
conda install --yes pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c anaconda protobuf

# Hydra
pip install hydra-core --upgrade
pip install hydra-joblib-launcher --upgrade
conda install --yes -c anaconda pyyaml

pip install opencv-python
pip install emnist
conda install --yes h5py

# Analysis/results
#pip install grad-cam
conda install --yes scikit-learn==1.0.1
pip install matplotlib
pip install seaborn
pip install tensorboard
pip install torch-summary
pip install netcal # Installs latest pytorch too, which may or may not be compatible
