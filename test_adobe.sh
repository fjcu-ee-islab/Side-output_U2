
CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/adobe/adobe_test/test_1 \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/u2_sl_adobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_mine_1 \
    --write_images \



CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 3 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/adobe/adobe_test/test_3 \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/u2_sl_adobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_mine_3 \
    --write_images \



CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/adobe/adobe_test/test_7 \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/u2_sl_adobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_mine_7 \
    --write_images \




python psnr_adobe.py 

