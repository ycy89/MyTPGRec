# TPGRec

## About The Project
Pytorch implementation for the paper "TPGRec: Text-Enhanced and Popularity-Smoothing Graph Collaborative Filtering for Long-Tail Item Recommendation"

## Environments
- python 3.9
- torch==1.13.0+cu116
- numpy==1.25.1
- scipy==1.9.3

## Dataset
We refer to [MICRO](https://github.com/CRIPAC-DIG/MICRO) and follow these steps to prepare the dataset:
- Download 5-core reviews data & meta data from [Amazon review data 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- Install [sentence-transformers](https://www.sbert.net/docs/installation.html) and download a [pretrained model](https://www.sbert.net/docs/pretrained_models.html) to extra text-embedding of items
- Run ./data/build_data.py to create train/valid/test dataset & item_text_feat.npy.
- Processed data can be downloaded from [Google Drive](https://drive.google.com/file/d/1u8U30EVcHVd3cP9kZpVWqfmJ8rYxQc0c/view?usp=drive_link)

## Run
Train each dataset using its optimal configuration
```commandline
python main.py --dataset baby --ssl_temp 0.5 --edge_add_rate 0.2
```
```commandline
python main.py --dataset beauty --ssl_temp 0.15 --edge_add_rate 0.1
```
```commandline
python main.py --dataset clothing --ssl_temp 0.15 --edge_add_rate 0.1
```
```commandline
python main.py --dataset sports --ssl_temp 0.15 --edge_add_rate 0.1
```
