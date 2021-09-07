
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
	--model utwo_side_sl \
    --flow_scale 1 \
    --batch_size 2 \
	--crop_size 352 352 \
	--print_freq 1 \
	--dataset VideoInterp \
    --num_interp 1 \
	--val_num_interp 1 \
	--skip_aug \
	--save_freq 10 \
	--start_epoch 0 \
	--stride 32 \
    --lr_milestones 350 450 \
    --step_size 1 \
	--epochs 500 \
	--name utwo_sl_window \
	--train_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/train \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/val \
