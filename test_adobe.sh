
CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_1 \
    --resume /home/ubuntu/Downloads/Side-output_U2-main/pretrained_models/adobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_result_1 \
    --write_images \



CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 3 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_3 \
    --resume /home/ubuntu/Downloads/Side-output_U2-main/pretrained_models/dobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_result_3 \
    --write_images \



CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 7 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_7 \
    --resume /home/ubuntu/Downloads/Side-output_U2-main/pretrained_models/adobe.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name adobe_result_7 \
    --write_images \




python psnr_adobe.py 

