
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
                   --batch_size 64 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/train62.tsv \
                   --test_set ${TASK_DATA_PATH}/test62.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --save_steps 400 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --epoch 100 \
                   --validation_steps 200 \
                   --max_seq_len 64 \
                   --learning_rate 5e-5 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --checkpoints ${TASK_DATA_PATH}/classifier_max_channel_attn_max_word_attn_concat_last \
                   --init_pretraining_params ${MODEL_PATH}/params

                   --checkpoints ${TASK_DATA_PATH}/classifier_weightedAdd_all_attention_concat_middle \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_concat_maxAttn1_middle/test_step_71900_0.377232





                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/origin/step_420001


                   --checkpoints ${TASK_DATA_PATH}/classifier_weightedAdd_all_attention \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_weightedAdd_all_attention/test_step_367400_0.377421



                    --checkpoints ${TASK_DATA_PATH}/classifier_weightedAdd_all_attention_concat_middle \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/classifier_weightedAdd_all_attention_concat_middle/test_0.378941_step_228900


                   --init_pretraining_params ${MODEL_PATH}/params



                   --checkpoints ${TASK_DATA_PATH}/origin1 \
                   --init_checkpoint ${TASK_DATA_PATH}/old_checkpoints/origin/step_420001

----------------------------------------cged17---------------------------------------------------------
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/disk1/private/zhaoxinhao/.conda/pkgs/nccl-1.3.5-cuda9.0_0/lib:$LD_LIBRARY_PATH
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged17'
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
                   --train_set ${TASK_DATA_PATH}/train62.tsv \
                   --test_set ${TASK_DATA_PATH}/test62.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --save_steps 400 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --epoch 70 \
                   --validation_steps 200 \
                   --max_seq_len 64 \
                   --learning_rate 5e-5 \
                   --min_f1 0.26 \
                   --max_loss 0.005 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --checkpoints ${TASK_DATA_PATH}/origin \
                   --init_checkpoint ${TASK_DATA_PATH}/origin/train_0.000443_step_16000

                   --init_pretraining_params ${MODEL_PATH}/params




-----------------cged_all_16 (15,16,17,18) ---------------------------

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/disk1/private/zhaoxinhao/.conda/pkgs/nccl-1.3.5-cuda9.0_0/lib:$LD_LIBRARY_PATH
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged_all_16'
OLD_CHECKPOINT='/data/disk1/private/wangmuzi/data/ERNIE/cged_all'
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
                   --train_set ${TASK_DATA_PATH}/train62.tsv \
                   --test_set ${TASK_DATA_PATH}/16test62.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --save_steps 400 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.002 \
                   --epoch 100 \
                   --validation_steps 1000 \
                   --max_seq_len 64 \
                   --learning_rate 5e-5 \
                   --min_f1 0.38 \
                   --max_loss 0.00001 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --checkpoints ${TASK_DATA_PATH}/origin \
                   --init_pretraining_params ${MODEL_PATH}/params

                   --init_checkpoint ${OLD_CHECKPOINT}/classifier_avgPool_left_middle_right_31_concat_middle/test_0.392157_step_191600

                   --init_checkpoint ${TASK_DATA_PATH}/origin/test_0.398041_step_78000



-----------------cged_all_17 (15,16,17,18) ---------------------------

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH='/usr/local/cuda-9.0/lib64'
#export LD_LIBRARY_PATH='~/.conda/envs/env36/lib'
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/data/disk1/private/zhaoxinhao/.conda/pkgs/nccl-1.3.5-cuda9.0_0/lib:$LD_LIBRARY_PATH
TASK_DATA_PATH='/data/disk1/private/wangmuzi/data/ERNIE/cged_all'
OLD_CHECKPOINT='/data/disk1/private/wangmuzi/data/ERNIE/cged_seg'
MODEL_PATH='/data/disk1/private/wangmuzi/data/ERNIE/pretrain_model'
python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 64 \
                   --num_labels 9 \
                   --label_map_config ${TASK_DATA_PATH}/label_map.json \
                   --train_set ${TASK_DATA_PATH}/train62.tsv \
                   --test_set ${TASK_DATA_PATH}/17test62.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.02 \
                   --epoch 100 \
                   --validation_steps 200 \
                   --max_seq_len 64 \
                   --learning_rate 5e-5 \
                   --skip_steps 100 \
                   --num_iteration_per_drop_scope 1 \
                   --random_seed 1 \
                   --init_checkpoint ${TASK_DATA_PATH}/origin/test_0.398041_step_78000

                   --init_pretraining_params ${MODEL_PATH}/params