#!/usr/bin/env bash
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=5
#export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/hsk'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size 32 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/valid.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ${TASK_DATA_PATH}/checkpoints_5e-5__ \
                   --save_steps 300 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --validation_steps 100 \
                   --epoch 50 \
                   --max_seq_len 128 \
                   --learning_rate 5e-5 \
                   --init_checkpoint ${TASK_DATA_PATH}/checkpoints_5e-5_/step_1800 \
                   --skip_steps 50 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --init_pretraining_params ${MODEL_PATH}/params
#                    --test_set ${TASK_DATA_PATH}/msra_ner/test.tsv






export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/hsk'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 1 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ${TASK_DATA_PATH}/checkpoints_5e-5__ \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --max_seq_len 64 \
                   --learning_rate 5e-5 \
                   --init_checkpoint ${TASK_DATA_PATH}/checkpoints_5e-5__/step_13251 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --test_set ${TASK_DATA_PATH}/test.tsv



#!/usr/bin/env bash
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
#export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/hsk'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE/pretrain_model'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test false \
                   --batch_size 1 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/toy.tsv \
                   --dev_set ${TASK_DATA_PATH}/toy.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ${TASK_DATA_PATH}/toy \
                   --save_steps 300 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --validation_steps 100 \
                   --epoch 50 \
                   --max_seq_len 10 \
                   --learning_rate 5e-5 \
                   --skip_steps 50 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --init_pretraining_params ${MODEL_PATH}/params
#                    --test_set ${TASK_DATA_PATH}/msra_ner/test.tsv



#!/usr/bin/env bash
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/disk1/private/zhaoxinhao/.conda/pkgs/nccl-1.3.5-cuda9.0_0/lib:$LD_LIBRARY_PATH
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged_seg'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE/pretrain_model'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val false \
                   --do_test true \
                   --verbose false \
                   --batch_size 32 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/valid.tsv \
                   --test_set ${TASK_DATA_PATH}/test_cut_short.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --save_steps 400 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.002 \
                   --epoch 500 \
                   --validation_steps 200 \
                   --max_seq_len 32 \
                   --learning_rate 5e-5 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --checkpoints ${TASK_DATA_PATH}/classifier_concat_left_middle_right \
                   --init_pretraining_params ${MODEL_PATH}/params


                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/origin/step_420001


                   --checkpoints ${TASK_DATA_PATH}/classifier_weightedAdd_all_attention \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_weightedAdd_all_attention/test_step_367400_0.377421



                    --checkpoints ${TASK_DATA_PATH}/classifier_weightedAdd_all_attention_concat_middle \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_weightedAdd_all_attention_concat_middle/test_0.378941_step_228900


                   --checkpoints ${TASK_DATA_PATH}/classifier_concat_maxAttn1_middle \
                   --init_pretraining_params ${MODEL_PATH}/params

                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_concat_maxAttn1_middle/test_step_71900_0.377232


                   --checkpoints ${TASK_DATA_PATH}/origin1 \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/origin/step_420001



export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged_seg'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE/pretrain_model'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train true \
                   --do_val false \
                   --do_test true \
                   --verbose false \
                   --batch_size 64 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/train_toy.tsv \
                   --dev_set ${TASK_DATA_PATH}/valid_toy.tsv \
                   --test_set ${TASK_DATA_PATH}/test_toy.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --checkpoints ${TASK_DATA_PATH}/classifier_concat_maxAttn1_middle \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_concat_maxAttn1_middle/test_step_71900_0.377232 \
                   --save_steps 400 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --validation_steps 10 \
                   --epoch 10 \
                   --max_seq_len 32 \
                   --learning_rate 5e-5 \
                   --skip_steps 5 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1




export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=2
#export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged_seg'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 64 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --max_seq_len 32 \
                   --learning_rate 5e-5 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --test_set ${TASK_DATA_PATH}/test_cut_short.tsv \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifie_windowAdd_left_right_concat_middle/test_step_314100_0.385589

                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/origin/test_step_338500_0.377255  step_420001

                    --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifie_windowAdd_left_right_concat_middle/test_step_314100_0.385589
                    #                   --init_checkpoint ${TASK_DATA_PATH}/origin32/train_step_41200_0.000111

