###### [Overview](#Comparing-Contrastive-Representations-Through-Proprietary-Audio-Collections) | [Setup](#Setup) 


## Comparing Contrastive Representations Through Proprietary Audio Collections


## Setup

To reproduce these experiments, audio embeddings may be downloaded from INSERT LINK. Packages for the clap-testing environment must be installed in the following order to not cause conflicts. There is a bug in laion-clap that is circumvented by using version transformers version 4.30. 

```shell
# Create Environment
conda create -n clap python=3.10
conda activate clap

# Install packages
pip install msclap
pip install laion-clap
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers=4.30
pip install chardet
```
