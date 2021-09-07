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
	--name xray_large \
    --write_images \


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/backup/mine/data_train/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/utwo_sl_window_large.tar \
	--name window_u2_sl \
    --val_batch_size 1 \
    --gpus 1 \
	--name xray_large_window \
    --write_images \

