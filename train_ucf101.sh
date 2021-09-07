
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
	--model utwo_side_sl \
    --flow_scale 1.0 \
	--batch_size 2 \
	--crop_size 352 352 \
	--print_freq 1 \
	--dataset VideoInterp \
    --num_interp 7 \
	--val_num_interp 7 \
	--save_freq 10 \
	--start_epoch 0 \
	--stride 32 \
    --lr_milestones 200 400 \
	--epochs 500 \
    --name u2_sl_ucf101 \
    --train_file /home/ubuntu/Downloads/slomo_mine/data/adobe/train \
    --val_file /home/ubuntu/Downloads/slomo_mine/data/adobe/validation \
    --save /home/ubuntu/Downloads/slomo_mine/model_result \
