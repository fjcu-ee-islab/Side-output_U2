CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model uu \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_uu.tar \
	--name uu_small_7 \
    --val_batch_size 1 \
	--gpus 1 \

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_1 \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2.tar \
	--name u2_small_7 \
    --val_batch_size 1 \
	--gpus 1 \



CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/mef_u2_sl.tar \
	--name u2_sl_small \
    --val_batch_size 1 \
	--gpus 1 \


