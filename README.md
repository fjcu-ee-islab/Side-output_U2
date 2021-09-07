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
conda env create -f slomo.yaml
conda activate slomo_test
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

you need to cmodify your `--trian_file` and `--val_file` path to your path.
(must be absolute path)

## Xray_large
```
sh train_side_sl.sh
sh train_window_utwo_sl.sh
```

## Xray_small
```
sh train_mef_utwo_sl.sh
sh train_window_utwo_sl_small.sh
```

## Adobe240fps
```
sh train_adobe.sh
```

## UCF101
```
sh train_ucf101.sh
```



<Br/>
<Br/> 

# Download pre-trained models
## Xray_large
* [side_sl](https://drive.google.com/file/d/1MT2EL-Qj49LLoOyFlJSTES359rVr9OJn/view?usp=sharing)
* [window_utwo_sl](https://drive.google.com/file/d/1TuQjdeUBsk5EpOnB_hGQi9rfBTaXxm9s/view?usp=sharing)

## Xray_small
* [mef_utwo_sl](https://drive.google.com/file/d/1VR4MlogSSMTij3X43GtLNGhNWEb43h2h/view?usp=sharing)
* [window_utwo_sl_small](https://drive.google.com/file/d/13HPr6GjGzxp3n3wDwtpmDM4_-eCI-O6M/view?usp=sharing)

## Adobe240fps
* [u2_sl_adobe](https://drive.google.com/file/d/1vMD9Qpqe5NBUwLMfG9D84ZLtGMsCywx7/view?usp=sharing)

## UCF101
* [u2_sl_ucf101](https://drive.google.com/file/d/1vI3wunNkDdve1PaZy6sJWMo_BYTpYfba/view?usp=sharing)

```
mkdir pretrained_models
```
Please input the dataset to the ./pretrained_models/


<Br/>
<Br/>

# Testing
## Xray_large
```
sh test_xray_single.sh
```

## UCF101
```
sh test_ucf101.sh
```

## Xray_small
```
sh test_xray_multi.sh
```

## Adobe240fps
```
sh test_adobe.sh
```
