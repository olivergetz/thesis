###### [Overview](#Comparing-Contrastive-Representations-Through-Proprietary-Audio-Collections) | [Setup](#Setup) 

## Comparing Contrastive Representations Through Proprietary Audio Collections

Recent advances in machine learning techniques have opened up new avenues
for music technologies, some of which have yet to be extensively tested on
proprietary audio collections. Contrastive Audio-Language Pretraining is a
cornerstone of generative audio due to the robust representations they pro-
vide, but may also be used for file management, information retrieval, or
downstream finetuning on other tasks. As open source projects build upon
each other to create novel tools and unveil new knowledge, their traits may
be inherited from model to model, and it is therefore important to under-
stand the benefits and drawbacks of pretrained models. Two CLAP models
are evaluated on a variety of tasks using a proprietary audio collection.

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
