python src/test_BEAT.py --model_name 'BEAT' --GPU_id 0 --part 6 --lr 0.001 --dataset 'CUHK-PEDES' \
--epoch 80 --dataroot "<datapath>" --class_num 11000 \
--vocab_size 5000 --feature_length 1024 --mode 'test' --batch_size 64 --cr_beta 0.1 --rem_num 4