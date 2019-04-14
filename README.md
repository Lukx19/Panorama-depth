# Panorama Depth

Using original code from Omnidepth pytorch implementation
A PyTorch reimplementation of the Omnidepth paper from Zioulis et al., ECCV 2018:

**Notable difference with the paper:** PyTorch's weight decay for the Adam solver does not seem to function the same way as Caffe's. Hence, I do not use weight decay in training. Instead, I use a learning rate schedule

# Dependencies

## Use Docker

It is possible to reproduce results by running inside nvidia docker v2. Install nvidia docker and continue by following:

```
 git pull https://github.com/Lukx19/PanoDepth-docker.git
 cd PanoDepth-docker
 docker build . -t lukx19/panodepth:latest
```

Adjust mounting paths in `run.bash` and run

```
 bash run.bash
```

## Manual setup

Required ubuntu packages:

```
sudo apt install build-essential
sudo apt install libstdc++6
```

[Set up a conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

```
conda env create -f environment.yml
source activate panodepth
```

If you want need to run scripts to modify dataset you will need following dependencies

```
sudo apt install libopenexr-dev
sudo apt install libproj-dev
```

# Dataset

To get the OmniDepth dataset, please file a request with the authors [here](http://vcl.iti.gr/360-dataset/).

Save files one level above this git folder into folder datasets. The file structure will look like this:

```
- datasets
    - Matterport
        - 0_0151156dd8254b07a241a2ffaf0451d41_color_0_Left_Down_0.0.png
        - ...
    - Stanford
        - area3
            - 0_area_31_color_0_Left_Down_0.0.png
        - ...
    - SceneNet
    - SunCG
- Panorama-depth
    - train.py
    - test.py
    - ...
```

Original datasets do not include sparse points. They need to be extraxted by running `gen_sparse_points` in `genDataset` folder. This operation is CPU intensive.

```
python gen_sparse_points ../../datasets
```

# Usage

## Train

You need to have visdom server running on your computer. If you don't than you need to run

```
visdom &
```

The visualizations can be viewed at `localhost:8097`. If running visdom and the network on a server, you will need to tunnel to the server to view it locally. For example:

```
ssh -N -L 8888:localhost:8097 <username>@<server-ip>
```

allows you to view the training visualizations at localhost:8888 on your machine.

Example of training command:

```
 python train.py test1 --network_type RectNet --loss_type Revis_all --add_points --gpu_ids 0,1
```

This will train RectNet with addition of sparse points and use Revis_all loss function. All checkpoints and final trained model will be saved to folder `./experiments/RectNet_Revis_all_test1`. The training without sparse points is possible by removing `--add_points` flag.

## Test

Run this if you use model with sparse points

```
 python test.py ./experiments/RectNet_Revis_all_100pts/ --network_type RectNet --add_points --gpu_ids 0
```

If you want to only run RGB to depth model than run following

```
 python test.py ./experiments/RectNet_Revis --network_type RectNet --gpu_ids 0
```

If you want to also save outputs of the network to depth and image files please add following commandline arguments

```
--save_results --test_list ./data_splits/original_p100_d20_test_split_top50.txt
```

This will run same testing with only top 50 elements of test data split.

# Credit

If you do use this repository, please make sure to cite the authors' original paper:

```
Zioulis, Nikolaos, et al. "OmniDepth: Dense Depth Estimation for Indoors Spherical Panoramas."
Proceedings of the European Conference on Computer Vision (ECCV). 2018.
```
