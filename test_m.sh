#uu
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model uu \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_uu.tar \
    --val_batch_size 1 \
    --gpus 1 \

#plusu
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model plusu \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_plusu.tar \
    --val_batch_size 1 \
    --gpus 1 \

#uplus
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model uplus \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_uplus.tar \
    --val_batch_size 1 \
    --gpus 1 \

#plusplus
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model plusplus \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_plusplus.tar \
    --val_batch_size 1 \
    --gpus 1 \

#twou
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model twou \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_twou.tar \
    --val_batch_size 1 \
    --gpus 1 \

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

#twotwo
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model twotwo \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_twotwo.tar \
    --val_batch_size 1 \
    --gpus 1 \

#twoplus
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model twoplus \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_twoplus.tar \
    --val_batch_size 1 \
    --gpus 1 \

#plustwo
CUDA_VISIBLE_DEVICES=0 python3 eval.py \
	--model plustwo \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/slomo_mine/data/xray_large/test \
    --resume /home/ubuntu/Downloads/slomo_mine/pretrained_models/m_plustwo.tar \
    --val_batch_size 1 \
    --gpus 1 \

