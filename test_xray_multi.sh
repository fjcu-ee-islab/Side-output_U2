


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/1_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
	--name xray_small_1 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \



CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 3 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/3_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
	--name xray_small_3 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \



CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
	--name xray_small_7 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \



CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/1_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/utwo_sl_window_small.tar \
	--name xray_small_window_1 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \


CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 3 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/3_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/utwo_sl_window_small.tar \
	--name xray_small_window_3 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \



CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/utwo_sl_window_small.tar \
	--name xray_small_window_7 \
    --val_batch_size 1 \
	--gpus 1 \
    --write_images \



