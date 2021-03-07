
## Pre-requisites
#### Dataset files
Download [dataset](https://drive.google.com/file/d/1hMP57PnvPsxj2A33upB9W81b4zJD-Rsi/view?usp=sharing) and put it in home directory of the project. To extract the files:

`tar -xvf data.tar.gz`

#### Dependencies
The code base been tested on CentOS Linux 7 using Python3.6.7. The dependencies can be installed using:

```sh
# Install [Anaconda 3](https://www.anaconda.com/) for managing Python packages and environments.
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh
source ~/.bashrc

# Set up a new conda environment and activate it.
conda create -n meta-gcn-tsp-env python=3.6.7
source activate meta-gcn-tsp-env

# Install all dependencies
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install numpy==1.15.4 scipy==1.1.0 matplotlib==3.0.2 seaborn==0.9.0 pandas==0.24.2 networkx==2.2 scikit-learn==0.20.2 tensorflow-gpu==1.12.0 tensorboard==1.12.0 Cython
pip3 install tensorboardx==1.5 fastprogress==0.1.18

```

## Steps to train/validate/test different models

* Scripts to train/test **meta-GCN model** on a variety of graph sizes N, numbers of modes M, scales L, respectively:
  * `meta_main_Size.py`
  * `meta_main_Mode.py`
  * `meta_main_Scale.py`

* Scripts to train/test **multi-GCN**  on a variety of graph sizes N, numbers of modes M, scales L, respectively:
  * `multi_main_Size.py` 
  * `multi_main_Scale.py`
  * `multi_main_Mode.py`
 
* Script to train/test the **original-GCN** model
  * `main.py`

## Configuration files
All config files are present in the subfolders of `configs/`:
* Config files for **multi-GCN** model are in `mode_multi`, `size_multi` or `scaling_multi_50` folders. They are named as **multi_config.json**.
* Config files for **meta-GCN** model are in  `mode`, `size` or `scaling_50`. They are named as **meta_config.json**.
* Config files for **original-GCN** model are in `mode`, `size` or `scaling_50` subfolders.


## Examples

* To train and test the **meta-GCN** model on graphs of various sizes N:
```sh
python3 meta_main_Size.py  --config configs/size/meta_config.json
python3 meta_main_Size.py --config configs/size/meta_config.json --Testing True
```

* To train and test the **meta-GCN** model on graphs of various scales L:
```sh
python3 meta_main_Scale.py  --config configs/scaling_50/meta_config.json
python3 meta_main_Scale.py --config configs/scaling_50/meta_config.json --Testing True
```

* To train and test the **meta-GCN** model on graphs with various numbers of modes M:
```sh
python3 meta_main_Mode.py  --config configs/mode/meta_config.json
python3 meta_main_Mode.py  --config configs/mode/meta_config.json --Testing True
```

* To train and test the **multi-GCN** model on graphs of various sizes N:
```sh
python3 multi_main_Size.py --config configs/size_multi/multi_config.json
python3 multi_main_Size.py  --config configs/size_multi/multi_config.json --Testing True
```

* To train and test the **original-GCN** model with graphs of size N=100:
```sh
python3 main.py  --config configs/tsp100.json
python3 main.py  --config configs/tsp100.json --Testing True
```


## Pre-trained models
Download the [pre-trained models](https://drive.google.com/file/d/1FbJ49A1h0NraTE6Ee42GZtJrCmQyl_V9/view?usp=sharing) and put in the home folder of the project. To extract the files:

`tar - xvf pre_trained_models.tar.gz`

This will extract the pre-trained models in folders named `pre_trained_models/`. 

The config files are by default configured to load/save models in the folder `outputs/`.
To perform testing using the pre-trained models, we need to change the default load/save path of the models in configuration files inside configs folder.


For instance, to perform testing on the pre trained **meta-GCN** model with various modes M, kindly perform the following steps:

1. Open configs/mode/meta_config.json

2. Change **"output_dir_name": "outputs"** to **"output_dir_name": "pre_trained_models"**. 

    *The **"output_dir_name"** key controls the storage location of model.*

  
3. Run 

    `python3 meta_main_Mode.py  --config configs/mode/meta_config.json --Testing True`



## Credits 

Our code is built upon code provided by Joshi et al. https://github.com/chaitjo/graph-convnet-tsp/ 

