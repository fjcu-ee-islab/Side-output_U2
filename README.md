# Side-output_U2
## X-ray Video Frame Interpolation Using Nested Architecture

Given two consecutive frames, video frame interpolation aims at generating intermediate frames by judging the direction and speed of the movement of the object.
While most existing methods focus on single-frame interpolation, we think that multi-frame interpolation is also important, so we propose a nested convolutional neural network for multi-frame video interpolation.

In order to make the model capable of multi-frame interpolation, we use a two-stage optical flow compute architecture to obtain the optical flow at arbitrary time. In the first stage, we first use the U-net architecture for optical flow compute, and then calculate the optical flow linearly to get the optical flow at arbitrary time. In the second stage, we use the nested U2-net model to optimize the arbitrary-time flow. However, the architecture is still slightly inadequate in processing large displacement motion or arbitrary-time flow. Therefore, we use the side output method to strengthen the model's multi-scale and multi-level feature transfer capabilities, and let our model have better interpolation results.

At the same time, we also use the video frame interpolation technology in special X-ray imaging medical videos, and interpolate single-frame for large displacement motion and multi-frame for small displacement motion. The experimental results show that, our method has better results in single frame interpolation than existing methods. In multi-frame interpolation, although our method produces slightly worse results than other methods in fewer frames interpolation, however, with the prediction of intermediate frames as the number increases, our method has more advantages than other methods.


![image](https://github.com/fjcu-ee-islab/Sideoutput_U2/blob/main/figure/model.png)  

<Br/>
<Br/>

# Anaconda environment

```
conda create -n slomo_test
conda activate slomo_test
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install scikit-image
pip install tqdm
pip install natsort
pip install ffmpeg
pip install opencv-python
conda install -c conda-forge tensorboardx

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

```


<Br/>

# Download datasets
```
mkdir data
```
 you can download the datasets as follow
* [Xray_large](https://drive.google.com/file/d/11nWfyS0sFQNNXRTilj-rntXccloB2bT9/view?usp=sharing)
* [Xray_small](https://drive.google.com/file/d/1N7UTCnmEsnPBxoJMGSYoeGUdS2FXccfO/view?usp=sharing)
* [Adobe240fps](https://drive.google.com/file/d/1u30NFgV6UCioyqQqTdMrlte_iesvosOw/view?usp=sharing)
* [UCF101](https://drive.google.com/file/d/1F1gyzLPoWnOAycpAJPqRPSqAtMXXfzIK/view?usp=sharing)

and then, move the dataset to the ./data/


<Br/>
<Br/>

# Training

you need to modify your `--trian_file` and `--val_file` path to your path in each dataset.
(must be absolute path)


## Xray_large
`--trian_file your path/data/xray_large/train`

`--val_file your path/data/xray_large/val`
```
sh train_xray_large.sh
sh train_xray_large_window.sh
```

## Xray_small
`--trian_file your path/data/xray_small/train`

`--val_file your path/data/xray_small/val`
```
sh train_xray_small.sh
sh train_xray_small_window.sh
```

## Adobe240fps
`--trian_file your path/data/adobe/train`

`--val_file your path/data/adobe/validation`
```
sh train_adobe.sh
```

## UCF101
`--trian_file your path/data/adobe/train`

`--val_file your path/data/adobe/validation`
```
sh train_ucf101.sh
```

<Br/>
<Br/> 

# Download pre-trained models

```
mkdir pretrained_models
```
Download pre-trained models
## Xray_large
* [xray_large](https://drive.google.com/file/d/1MT2EL-Qj49LLoOyFlJSTES359rVr9OJn/view?usp=sharing)
* [xray_large_window](https://drive.google.com/file/d/1TuQjdeUBsk5EpOnB_hGQi9rfBTaXxm9s/view?usp=sharing)

## Xray_small
* [xray_small](https://drive.google.com/file/d/1VR4MlogSSMTij3X43GtLNGhNWEb43h2h/view?usp=sharing)
* [xray_small_window](https://drive.google.com/file/d/13HPr6GjGzxp3n3wDwtpmDM4_-eCI-O6M/view?usp=sharing)

## Adobe240fps
* [adobe](https://drive.google.com/file/d/1vMD9Qpqe5NBUwLMfG9D84ZLtGMsCywx7/view?usp=sharing)

## UCF101
* [ucf101](https://drive.google.com/file/d/1vI3wunNkDdve1PaZy6sJWMo_BYTpYfba/view?usp=sharing)

Please move the pretrained_models to the ./pretrained_models/


<Br/>
<Br/>

# Testing
you need to modify your `--val_file` and `--resume` path to your path in each dataset.
(must be absolute path)

## Xray_large

`--val_file your path/data/xray_large/test`

`--resume your path/pretrained_models/xray_large` and `--resume your path/pretrained_models/xray_large_window`

```
sh test_xray_single.sh
```

## UCF101

`--val_file your path/data/ucf101/test`

`--resume your path/pretrained_models/ucf101.tar`

```
sh test_ucf101.sh
```

## Xray_small

`--val_file your path/data/xray_large/1_inter`

`--val_file your path/data/xray_large/3_inter`

`--val_file your path/data/xray_large/7_inter`

`--resume your path/pretrained_models/xray_small.tar` and `--resume your path/pretrained_models/xray_small_window.tar`

```
sh test_xray_multi.sh
```

## Adobe240fps

`--val_file your path/data/adobe/adobe_test/test_1`

`--val_file your path/data/adobe/adobe_test/test_3`

`--val_file your path/data/adobe/adobe_test/test_7`

`--resume your path/pretrained_models/adobe.tar`

```
sh test_adobe.sh
```
