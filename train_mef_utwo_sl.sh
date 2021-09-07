

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
	--model utwo_side_sl \
    --flow_scale 1.0 \
	--batch_size 4 \
	--crop_size 352 352 \
	--print_freq 1 \
	--dataset VideoInterp \
    --num_interp 7 \
	--val_num_interp 7 \
	--skip_aug \
	--save_freq 5 \
	--start_epoch 0 \
	--stride 32 \
    --lr_milestones 350 450 \
	--epochs 500 \
    --name train_xray_small \
    --train_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/small_move_train \
    --val_file /home/ubuntu/Downloads/slomo_mine/data/xray_small/7_inter \
