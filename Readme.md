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

Our experiment involves three open real world dataset popularly deployed by existing works. Each dataset represents a spatio-temporal tracjectory density data. All datasets used in this project are stored in the `PrivSTD_codes/data` directory of this GitHub repository.

## 4 Usage
Please download the requirements.txt and use the following code to install the required Python environment:
```
pip install -r requirements.txt
```
Then, we compare our method with different methods, that are, AGM-Tri, AGM-FCL, CPGM-DP.Before running with the programs, you need to download the datasets and put the code in the same directory with the datasets. These comparing methods can be run through a command with parameters including: a *method name*, a *model name*, a *epsilon*, a *dataset name*, a *alpha*, a *learning rate*,a *cell_size*,a *truncate size* and a *epoch*. Some of the parameters listed here may not be required by certain methods, but they do not affect the execution. To run these algorithms, we need to run the following commands separately:
For example:

```
python run.py UG --model UG --eps 0.5 --dataset CABS_SF --alpha 1.0 --lr 0.001 --sample_size 384 --truncate_size 3 --epoch 20
```

Output:

For each method, we output the following process and running time respectively. Here is the output of AGN-FCL for the above example:

```
[INFO] Running command:
python UG.py --model UG --eps 3.0 --dataset CABS_SF --alpha 1.0 --lr 0.001 --cell_size 384 --truncate_size 3 --epoch 20
2026-01-05 19:32:26,892 config: {'datasets': {'name': 'CABS_SF', 'cell_size': 384, 'time_grid': 24, 'patch_size': 1, 'truncate_size': '3'}, 'net': {'img_size': 128, 'window_size': 8, 'in_chans': 1, 'embed_dim': 180, 'depth': [6, 6, 6, 6], 'num_heads': 4, 'mlp_ratio': 2, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'resi_connection': '1conv'}, 'train': {'model': 'UG', 'loss_type': 'l2', 'loss_weight': 1.0, 'epochs': 20, 'lr': 0.001, 'alpha': 1.0, 'batch_size': 3, 'eval_freq': 10, 'save_dir': './results', 'gpu': '0', 'comment': '', 'reg_beta': 0.1, 'window_size': 5, 'data_norm': False, 'gsure_weight': 0.05, 'beta0': 1.0, 'eta_min': 1e-06, 'eta_max': 0.05, 'v_floor': 1e-10, 'Lstar_gamma': 0.1}, 'privacy': {'eps': 3.0, 'scheme': 'c1', 'n_bands': 4}, 'test': {'test_size': 20000, 'sm': [20]}, 'path': {'pretrain': None}, 'is_train': True}
2026-01-05 19:32:26,962 max_lon: 37.80900010207742, min_lon: 37.60701506734957, max_lat: -122.2108775387225, min_lat: -122.45235172061756
2026-01-05 19:32:26,962 number of samples: 846654
2026-01-05 19:32:28,277 counts shape: (385, 384, 24)
2026-01-05 19:32:28,321 counts_max: 111, counts_min: 0, counts_mean: 0.23861748060966811, median: 0.0, counts_sum: 846653
2026-01-05 19:32:28,322 UniformGrid m: 504 (from c=10.0)
2026-01-05 19:32:28,877 noisy_max: 80.20313262939453, noisy_min: -12.073646545410156, noisy_mean: 0.1378733217716217, noisy_median: 0.07636275887489319, noisy_sum: 840528.75
2026-01-05 19:32:28,942 data_rec reconstruction completed in 0.06 seconds
2026-01-05 19:32:28,943 data_rec shape: (385, 384, 24)
2026-01-05 19:32:28,952 data_rec stats - max: 108.02452850341797, min: -12.980752944946289, mean: 0.23689134418964386, sum: 840528.375
2026-01-05 19:32:32,569 Forecast queries completed
2026-01-05 19:32:33,885 Hotspot queries completed
2026-01-05 19:32:33,894 data saved at./results/CABS_SF/384/eps_3.0/published_data_rec_UG.npy
2026-01-05 19:32:33,894 UG finished.


```

Our methods can be run through a command with the same parameters :
For example:

```
python run.py PrivSTD --model PrivSTD --eps 3.0 --dataset CABS_SF --alpha 1.0 --lr 0.001 --cell_size 384 --truncate_size 3 --epoch 20

```

Output:
```
[INFO] Running command:
python PrivSTD.py --model PrivSTD --eps 3.0 --dataset CABS_SF --alpha 1.0 --lr 0.001 --cell_size 384 --truncate_size 3 --epoch 20
2026-01-05 19:34:00,093 config: {'datasets': {'name': 'CABS_SF', 'cell_size': 384, 'time_grid': 24, 'patch_size': 1, 'truncate_size': '3'}, 'net': {'img_size': 128, 'window_size': 8, 'in_chans': 1, 'embed_dim': 180, 'depth': [6, 6, 6, 6], 'num_heads': 4, 'mlp_ratio': 2, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'resi_connection': '1conv'}, 'train': {'model': 'PrivSTD', 'loss_type': 'l2', 'loss_weight': 1.0, 'epochs': 20, 'lr': 0.001, 'alpha': 1.0, 'batch_size': 3, 'eval_freq': 10, 'save_dir': './results', 'gpu': '0', 'comment': '', 'reg_beta': 0.1, 'window_size': 5, 'data_norm': False, 'gsure_weight': 0.05, 'beta0': 1.0, 'eta_min': 1e-06, 'eta_max': 0.05, 'v_floor': 1e-10, 'Lstar_gamma': 0.1}, 'privacy': {'eps': 3.0, 'scheme': 'c1', 'n_bands': 4}, 'test': {'test_size': 20000, 'sm': [20]}, 'path': {'pretrain': None}, 'is_train': True}
2026-01-05 19:34:00,162 max_lon: 37.80900010207742, min_lon: 37.60701506734957, max_lat: -122.2108775387225, min_lat: -122.45235172061756
2026-01-05 19:34:00,162 number of samples: 846654
2026-01-05 19:34:01,472 counts shape: (385, 384, 24)
2026-01-05 19:34:01,514 counts_max: 111, counts_min: 0, counts_mean: 0.23861748060966811, median: 0.0, counts_sum: 846653
2026-01-05 19:34:05,629 [DCT Select] (bh) U*=381, V*=384 ; sigma_ref=2.473021
2026-01-05 19:34:05,634 [DCT Release] time: 0.942 s
2026-01-05 19:34:05,745 [VarFloor] Using global var floor (p5) = 6.029191e+00
2026-01-05 19:34:09,037 Forecast queries completed
2026-01-05 19:34:10,416 Hotspot queries completed
2026-01-05 19:34:10,622 <---- epoch 0 ---->
2026-01-05 19:34:12,125 epoch 0, loss: 3.835677981376648
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:01<00:00, 22.57it/s]
2026-01-05 19:34:13,215 Saving model...
2026-01-05 19:34:13,223 data_rec saved at./results/CABS_SF/384/eps_3.0/published_data_rec_PrivSDT.npy
2026-01-05 19:34:13,223 <---- epoch 1 ---->
2026-01-05 19:34:13,545 epoch 1, loss: 2.0370214879512787
2026-01-05 19:34:13,546 <---- epoch 2 ---->
2026-01-05 19:34:13,852 epoch 2, loss: 2.0223832726478577
2026-01-05 19:34:13,852 <---- epoch 3 ---->
2026-01-05 19:34:14,135 epoch 3, loss: 2.0074282586574554
2026-01-05 19:34:14,135 <---- epoch 4 ---->
2026-01-05 19:34:14,415 epoch 4, loss: 1.9981556981801987
2026-01-05 19:34:14,415 <---- epoch 5 ---->
2026-01-05 19:34:14,705 epoch 5, loss: 2.0011146664619446
2026-01-05 19:34:14,706 <---- epoch 6 ---->
2026-01-05 19:34:14,998 epoch 6, loss: 2.0007294714450836
2026-01-05 19:34:14,998 <---- epoch 7 ---->
2026-01-05 19:34:15,280 epoch 7, loss: 2.0051930993795395
2026-01-05 19:34:15,281 <---- epoch 8 ---->
2026-01-05 19:34:15,556 epoch 8, loss: 2.0006527304649353
2026-01-05 19:34:15,556 <---- epoch 9 ---->
2026-01-05 19:34:15,835 epoch 9, loss: 2.003419131040573
2026-01-05 19:34:15,835 <---- epoch 10 ---->
2026-01-05 19:34:16,117 epoch 10, loss: 1.9982085078954697
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 27.51it/s]
2026-01-05 19:34:17,006 Saving model...
2026-01-05 19:34:17,020 data_rec saved at./results/CABS_SF/384/eps_3.0/published_data_rec_PrivSDT.npy
2026-01-05 19:34:17,020 <---- epoch 11 ---->
2026-01-05 19:34:17,356 epoch 11, loss: 2.0061915516853333
2026-01-05 19:34:17,356 <---- epoch 12 ---->
2026-01-05 19:34:17,642 epoch 12, loss: 2.0103926211595535
2026-01-05 19:34:17,643 <---- epoch 13 ---->
2026-01-05 19:34:17,927 epoch 13, loss: 2.004202052950859
2026-01-05 19:34:17,927 <---- epoch 14 ---->
2026-01-05 19:34:18,221 epoch 14, loss: 2.0025037229061127
2026-01-05 19:34:18,222 <---- epoch 15 ---->
2026-01-05 19:34:18,518 epoch 15, loss: 1.9977586418390274
2026-01-05 19:34:18,518 <---- epoch 16 ---->
2026-01-05 19:34:18,791 epoch 16, loss: 1.9987530708312988
2026-01-05 19:34:18,792 <---- epoch 17 ---->
2026-01-05 19:34:19,040 epoch 17, loss: 2.0005392283201218
2026-01-05 19:34:19,040 <---- epoch 18 ---->
2026-01-05 19:34:19,291 epoch 18, loss: 1.992037519812584
2026-01-05 19:34:19,291 <---- epoch 19 ---->
2026-01-05 19:34:19,573 epoch 19, loss: 1.9940429031848907
2026-01-05 19:34:19,574 PrivSTD Completed

```
If we want to run the user-level codes, just add a parameter "--user" :
For example:

```
python run.py UG --user --model UG --eps 3.0 --dataset Brightkite_Tokyo_Tokyo_JP --alpha 1.0 --lr 0.001 --cell_size 384 --truncate_size 3 --epoch 20

```

Output:
```
[INFO] Running command:
python UG_User.py --model UG --eps 3.0 --dataset Brightkite_Tokyo_Tokyo_JP --alpha 1.0 --lr 0.001 --cell_size 384 --truncate_size 3 --epoch 20
2026-01-05 19:44:51,352 config: {'datasets': {'name': 'Brightkite_Tokyo_Tokyo_JP', 'cell_size': 384, 'time_grid': 24, 'patch_size': 1, 'truncate_size': '3'}, 'net': {'img_size': 128, 'window_size': 8, 'in_chans': 1, 'embed_dim': 180, 'depth': [6, 6, 6, 6], 'num_heads': 4, 'mlp_ratio': 2, 'qkv_bias': True, 'drop_rate': 0.0, 'attn_drop_rate': 0.0, 'resi_connection': '1conv'}, 'train': {'model': 'UG', 'loss_type': 'l2', 'loss_weight': 1.0, 'epochs': 20, 'lr': 0.001, 'alpha': 1.0, 'batch_size': 3, 'eval_freq': 10, 'save_dir': './results', 'gpu': '0', 'comment': '', 'reg_beta': 0.1, 'window_size': 5, 'data_norm': False, 'gsure_weight': 0.05, 'beta0': 1.0, 'eta_min': 1e-06, 'eta_max': 0.05, 'v_floor': 1e-10, 'Lstar_gamma': 0.1}, 'privacy': {'eps': 3.0, 'scheme': 'c1', 'n_bands': 4}, 'test': {'test_size': 20000, 'sm': [20]}, 'path': {'pretrain': None}, 'is_train': True}
2026-01-05 19:44:51,352 [UG UserLevel] k=3
[read_userlevel_txt_dataset] path=./data/Brightkite_Tokyo_Tokyo_JP.txt
[read_userlevel_txt_dataset] rows=133596, users=2105, bad_lines=0
[read_userlevel_txt_dataset] min_vals(lon,lat,time)=[1.39625952e+02 3.55974250e+01 1.20918940e+09]
[read_userlevel_txt_dataset] max_vals(lon,lat,time)=[1.39793251e+02 3.57490090e+01 1.28742443e+09]
2026-01-05 19:44:51,664 [UG UserLevel] n_records=133596, n_users=2105, delta=2.257e-07
2026-01-05 19:44:51,664 max_lon: 139.793251, min_lon: 139.625952, max_lat: 35.749009, min_lat: 35.597425
2026-01-05 19:44:51,664 time_interval: 21731.953333333335 hours
2026-01-05 19:44:51,916 [TrueCounts] shape=(385, 384, 24) max=697 min=0 mean=0.037651909722222224
2026-01-05 19:44:51,950 [ Refinement] gamma = |D|/|Ds| = 133596/5589 = 23.903382
2026-01-05 19:44:51,951 [Sampled Ds] n_s=5589 (<= n_users*k = 6315)
2026-01-05 19:44:51,951 [UG UserLevel] m=25 (from c=10.0, n_users=2105)
2026-01-05 19:44:51,953 [UG Release] sigma=5.572663 (sens=k=3)
2026-01-05 19:44:51,953 noisy_counts_ug stats - max=74.79664611816406, min=-22.116579055786133, mean=0.3619377017021179, sum=5429.0654296875
2026-01-05 19:44:51,953 [UG] Starting overlap-conserving reconstruction (einsum)...
2026-01-05 19:44:51,982 [UG] Reconstruction time: 0.023s ; data_rec stats max=0.31620603799819946 min=-0.09349879622459412 mean=0.0015301071107387543 sum=5429.06494140625
2026-01-05 19:44:57,773 Forecast queries completed
2026-01-05 19:44:58,533 Hotspot queries completed
2026-01-05 19:44:58,555 Saved:
2026-01-05 19:44:58,555 ./results/Brightkite_Tokyo_Tokyo_JP/384/eps_3.0_userlevel_k3/published_data_rec_UG_userlevel.npy
2026-01-05 19:44:58,555 ./results/Brightkite_Tokyo_Tokyo_JP/384/eps_3.0_userlevel_k3/published_data_rec_UG_userlevel_gamma.npy
2026-01-05 19:44:58,555 UG (user-level, overlap-conserving, gamma-calibrated) finished.

```
After execution, the result datasets(.npy file) will be saved in the `/PrivSTD_codes/codes/results/cell_size/eps` directory, where `cell_size` and `eps` correspond to the input parameters.

Additionally, when running the PrivSTD code, the query files generated for evaluation—including **Cell Count**, **Hotspot Query**, and **Forecasting Query**—will be stored in the `/PrivSTD_codes/codes/data/dataset_name` directory.

It is also worth noting that the original VDR code has been publicly released at https://ddangchani.github.io/VDR/, and therefore is not described in detail here.

