# X-ray Video Frame Interpolation Using Nested Architecture

Given two consecutive frames, video frame interpolation aims at generating intermediate frames by judging the direction and speed of the movement of the object.
While most existing methods focus on single-frame interpolation, we think that multi-frame interpolation is also important, so we propose a nested convolutional neural network for multi-frame video interpolation.

In order to make the model capable of multi-frame interpolation, we use a two-stage optical flow compute architecture to obtain the optical flow at arbitrary time. In the first stage, we first use the U-net architecture for optical flow compute, and then calculate the optical flow linearly to get the optical flow at arbitrary time. In the second stage, we use the nested U2-net model to optimize the arbitrary-time flow. However, the architecture is still slightly inadequate in processing large displacement motion or arbitrary-time flow. Therefore, we use the side output method to strengthen the model's multi-scale and multi-level feature transfer capabilities, and let our model have better interpolation results.

At the same time, we also use the video frame interpolation technology in special X-ray imaging medical videos, and interpolate single-frame for large displacement motion and multi-frame for small displacement motion. The experimental results show that, our method has better results in single frame interpolation than existing methods. In multi-frame interpolation, although our method produces slightly worse results than other methods in fewer frames interpolation, however, with the prediction of intermediate frames as the number increases, our method has more advantages than other methods.

# 環境建立

Anaconda虛擬環境
```
conda create -n slomo_test
conda activate slomo_test
```

套件安裝
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install scikit-image
pip install natsort
conda install -c conda-forge tensorboardx
conda install -c conda-forge nvidia-apex
```

# 資料集下載

可以下載訓練和測試資料再這個網址


# 模型訓練

模型修改
```
sh train_m_plusplus.sh
sh train_m_plustwo.sh
sh train_m_plusu.sh
sh train_m_twoplus.sh
sh train_m_twotwo.sh
sh train_m_twou.sh
sh train_m_uplus.sh
sh train_m_utwo.sh
sh train_m_uu.sh
```

u2+sideoutput

```
sh train_side_l.sh
sh train_side_lr.sh
sh train_side_r.sh
sh train_side_s.sh
sh train_side_sl.sh
sh train_side_slr.sh
sh train_side_sr.sh
```


model eff
```
sh train_mef_utwo.sh
sh train_mef_utwo_sl.sh
sh train_mef_uu.sh
```

window
```
sh train_window_utwo_sl.sh
sh train_window_utwo_sl_small.sh
```

adobe
```
sh train_adobe.sh
```

ucf101
```
sh train_ucf101.sh
```


# 預訓練模型下載

預訓練模型下載

模型修改
* plusplus
* plustwo
* plusu
* twoplus
* twotwo
* twou
* uplus
* utwo
* uu


u2+sideoutput
* side_l
* side_lr
* side_r
* side_s
* side_sl
* side_slr
* side_sr



model eff
* mef_utwo
* mef_utwo_sl
* mef_uu


window
* window_utwo_sl
* window_utwo_sl_small


adobe
* u2_sl_adobe


ucf101
* u2_sl_ucf101


# 模型測試

模型修改
```
sh test_m.sh
```

側邊輸出
```

```

多幀插值有效性
```

```

模型修改有效性
```

```

window
```

```

adobe
```

```

ucf101
```

```
