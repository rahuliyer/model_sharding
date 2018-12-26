# A quick demonstration of model sharding

This script demonstrates how to split a model across two GPUs in PyTorch. The model is a simple neural net with 3 convolutional layers and 2 linear layers. The convolutional layers live on GPU 0 while the linear layers live on GPU 1. 

The model uses the MNIST dataset for training and testing.

## Usage ##
First setup the environment
```
conda env create -f environment.yml 
```

Activate the environment
```
conda activate model-sharding
```

Run the script
```
python model_sharding.py 
```
