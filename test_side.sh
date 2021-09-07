#utwo
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_utwo.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_r
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_r \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_r.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_l
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_l \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_l.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_s
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_s \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_s.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_lr
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_lr \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_lr.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_sr
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_sr \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_sr.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_sl
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_sl.tar \
    --val_batch_size 1 \
    --gpus 1 \

#side_slr
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
	--model utwo_side_slr \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/side_slr.tar \
    --val_batch_size 1 \
    --gpus 1 \


