
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
    --val_sample_rate 4 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
    --val_batch_size 1 \
	--gpus 1 \


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 3 \
	--flow_scale 1 \
    --val_sample_rate 2 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
    --val_batch_size 1 \
	--gpus 1 \


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
	--name window_u2_sl_small_7 \
    --val_batch_size 1 \
	--gpus 1 \


