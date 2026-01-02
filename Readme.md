# PrivSTD: Differentially Private Spatio-temporal Trajectory Density Data Publication

## 1 Introduction

This is a description of the code used for the experiments described in the paper entitled *PrivSTD: Differentially Private Spatio-temporal Trajectory Density Data Publication*. The code is available at [4open.science](https://github.com/YeShuzhan/PrivSTD).

We evaluated our differentially **Priv**ate **S**patio-temporal  **T**rajectory **D**ensity data publication method (PrivSTD) and other methods published recently, e.g., UG,AG,AHP,MWEM,PrivBayes,HB-Striped,VDR,  in terms of effectiveness and efficiency for spatio-temporal trajectory density data publication. The publications of comparing methods are shown in Table 1.

**Table** **1**: The original papers information of community search algorithms

| ALGORITHM |                         PAPER SOURCE                         | YEAR |
| --------- | :----------------------------------------------------------: | ---- |
| UG  |           ICDE, Differentially private grids for geospatial data            | 2013 |
| AG | ICDE, Differentially private grids for geospatial data |   2013   |
| MWEM  | NIPS, A simple and practical algorithm for differentially private data release. | 2020 |
| PrivBayes  | TODS, PrivBayes: Private Data Release via Bayesian Networks. | 2017 |
| HB-Striped  | VLDB, Understanding hierarchical methods for differentially private histograms | 2013 |
| VDR  | SIGMOD, A neural approach to spatio-temporal data release with user-level differential privacy | 2023 |
| PrivSTD  | Our methods | / |

## 2 Requirements

The experiments have been run on a Linux server with an Intel Xeon 2.1GHz CPU, 128 GB main memory, and a RTX3090 GPU. All programs are developed using Python 3.90 and incorporate additional Python packages.

## 3 DATASETS

Our experiment involves three open real world dataset popularly deployed by existing works. Each dataset represents a spatio-temporal tracjectory density data.

## 4 Usage
Please download the requirements.txt and use the following code to install the required Python environment:
```
pip install -r requirements.txt
```
Then, we compare our method with different methods, that are, AGM-Tri, AGM-FCL, CPGM-DP.Before running with the programs, you need to download the datasets and put the code in the same directory with the datasets. These comparing methods can be run through a command with parameters including: a *dataset name*, a *alpha*, a *epsilon*. To run these algorithms, we need to run the following commands separately:
For example:

```
python AG.py --dataset_name Email --alpha 3 --epsilon 3
```

Output:

For each method, we output the following process and running time respectively. Here is the output of AGN-FCL for the above example:

```


```

Our methods can be run through a command with parameters including: a *dataset name*, a *alpha*, a *epsilon*, and a **. To run these algorithms, we need to run the following commands separately:
For example:

```
Python PrivAGS.py --dataset_name Email --alpha 3 --epsilon 3 --numbins 500

```

Output:
```


```

After the process completing, the synthetic file *Synthetic_graph.txt* and *Synthetic_attribute.txt* will be saved at the dataset folder.
