

CUDA_VISIBLE_DEVICES=1 python3 eval_other.py \
	--model utwo_side_sl \
	--num_interp 1 \
	--flow_scale 1 \
	--dataset VideoInterp \
	--val_file /home/ubuntu/Downloads/Side-output_U2-main/data/ucf101/test \
    --resume /home/ubuntu/Downloads/Side-output_U2-main/pretrained_models/ucf101.tar \
    --val_batch_size 1 \
    --gpus 1 \
    --name test_ucf101_result \
    --write_images \



echo 'ours'
python3 eval_video_interpolation.py \
	--gt-dir /home/ubuntu/Downloads/Side-output_U2-main/data/ucf101/test/ \
	--motion-mask-dir /home/ubuntu/Downloads/Side-output_U2-main/data/ucf101/MotionMasks/ \
	--res-dir /home/ubuntu/Downloads/Side-output_U2-main/result_folder/ucf101_result/ \
	--res-suffix _Proposed.png \
	--buggy-motion-mask


