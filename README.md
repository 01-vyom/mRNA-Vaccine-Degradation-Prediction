# Improving COVID-19 mRNA vaccine degradation prediction

### [[Paper]](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/Improving_COVID_19_mRNA_Vaccine_Degradation_Prediction.pdf) |  [[Slides]](https://docs.google.com/presentation/d/1K-1k-UBP9XW0yqNx9woy-bCVn8hh0v3txNBQt4tv7HQ/edit?usp=sharing)

[Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Rahul Roy](https://www.linkedin.com/in/rahul-roy-5a7980128/)<sup>1</sup>

<sup>1</sup>[Computer & Information Science & Engineering, University of Florida, Gainesville, Florida, USA](https://www.cise.ufl.edu/)

The COVID-19 pandemic has shown devastating effects on a global scale for the past 2 years. An effective vaccine that can be distributed easily is very important to stop the COVID-19 pandemic. While mRNA vaccines are the fastest candidates, they tend to be unstable due to the spontaneous degradation of RNA molecules even with a single cut. In this paper, we develop modeling techniques to predict the rules of degradation of mRNA molecules in the COVID-19 vaccine using Deep learning techniques like Hybrid GRU-LSTM Networks, Graph Convolutional networks, and autoencoders. We state the improvement over mRNA Vaccine Degradation prediction by comparing both these methods for their MCRMSE loss values. We take the RNA structure along with their loop type as input and predict five values to understand the degradation criteria at each point in an RNA molecule using the Eterna Dataset consisting of 3029 mRNA sequences. We also used an augmented dataset generated using the ARNIE package to improve the previously described deep learning techniques. We show an overall improvement of 0.007 while using Graph Transformers as compared to the Hybrid GRU-LSTM models.
## Setup
### System and Requirements - 

- Linux OS
- Python Version - 3.7.12
- CUDA Version - 11.0
- CUDNN Version - 8.0.5
- Tensorflow - v2.6.2
### Setting up repository

```shell
git clone https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction.git
python -m venv mRNA_env
source $PWD/mRNA_env/bin/activate
```

### Installing Dependencies

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Code

Change directory to the root of the directory.

```shell
cd ./mRNA-Vaccine-Degradation-Prediction
```
### Dataset Setup
Download the dataset from the [COVID-19 mRNA Vaccine Degradation Prediction](https://www.kaggle.com/competitions/stanford-covid-vaccine/) kaggle competition using the following command:

```shell
kaggle competitions download -c stanford-covid-vaccine
```
### Training

#### Hybrid GRU-LSTM Model
To train the Hybrid GRU-LSTM model, run the following command:

```shell
python ./src/GRU_LSTM/train.py
```

Note:
- If required, change the path to the JSON file for the training data by setting the `train_data` variable to point to the appropriate path in [src/GRU_LSTM/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/GRU_LSTM/train.py).
- If required, change the path to the augmented data file by setting the `aug_data` variable to point to the appropriate path in [src/GRU_LSTM/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/GRU_LSTM/train.py).
- Number of folds can be changed by setting the `FOLD_N` variable in [src/GRU_LSTM/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/GRU_LSTM/train.py).

#### Graph Transformer Model
To train the Graph Transformer, run the following command:

```shell
python ./src/graph_transformer/train.py
```

Note:
- If required, change the path to the JSON file for the training data by setting the `path` variable to point to the appropriate path in [src/graph_transformer/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/graph_transformer/train.py).
- If required, change the path to the augmented data file by setting the `aug_data` variable to point to the appropriate path in [src/graph_transformer/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/graph_transformer/train.py).
- Number of folds can be changed by setting the `n_fold` variable in [src/graph_transformer/train.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/graph_transformer/train.py).


### Inference
#### Hybrid GRU-LSTM Model
To perform inference using the trained Hybrid GRU-LSTM model, run:

```shell
python ./src/GRU_LSTM/inference.py
```

Note:
- If required, change the path to the JSON file for the testing data by setting the `test_data` variable to point to the appropriate path in [src/GRU_LSTM/inference.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/GRU_LSTM/inference.py).
- Number of folds can be changed by setting the `FOLD_N` variable in [src/GRU_LSTM/inference.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/GRU_LSTM/inference.py).
- The output will be a `.csv` of the public and private test data in the `./result/` folder.

#### Graph Transformer Model
To perform inference using the trained Graph Transformer model, run:

```shell
python ./src/graph_transformer/inference.py
```

Note:

- If required, change the path to the JSON file for the testing data by setting the `path` variable to point to the appropriate path in [src/graph_transformer/inference.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/graph_transformer/inference.py).
- Number of folds can be changed by setting the `FOLD_N` variable in [src/graph_transformer/inference.py](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/src/graph_transformer/inference.py).
- The output will be a `.csv` of the public and private test data in the `./result/` folder.
## Results
Our algorithm achieves the following performance on the Private and Public Test Set:

| Technique name                                                           | Public Test Set (MCRMSE) | Private Test Set (MCRMSE) |
| ------------------------------------------------------------------------ | ------------------------ | ------------------------- |
| Weighted Average Ensemble GRU-LSTM Models                                | 0.24966                  | 0.36468                   |
| Weighted Average of GCN Transformer with and without Auto-Encoder Models | 0.24421                  | 0.35776                   |

Note:
- These performance are calculated over three columns [reactivity, deg_Mg_pH10, and deg_Mg_50C] as suggested in the [kaggle competition](https://www.kaggle.com/competitions/stanford-covid-vaccine/overview/evaluation).

## Acknowledgement

The Hybrid GRU-LSTM Model is based on [1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8237341/#supplemental-informationtitle) and the Graph Transformer is based on [2](https://www.kaggle.com/code/mrkmakr/covid-ae-pretrain-gnn-attn-cnn/) open-source implementation.

Licensed under the [MIT License](https://github.com/01-vyom/mRNA-Vaccine-Degradation-Prediction/blob/main/LICENSE.md)
