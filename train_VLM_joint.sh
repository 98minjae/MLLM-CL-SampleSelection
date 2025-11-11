#!/bin/bash
#SBATCH -p suma_a6000
#SBATCH --job-name=random
#SBATCH --output=logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --exclude=node37
##SBATCH -c 32

ulimit -u 200000
source ~/.bashrc
ml purge
conda init bash
conda activate sampling
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5

#!/bin/bash
# CIL CONFIG
# NOTE="v7_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_0625_fisherGradSim_samplewise_param1_emamean_emapool320_warmup_scenario17_10000_sample0_0625_emaratiox_warmup0_z1.54"
# NOTE="v7_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_03125_sft_scenario18_10000_random0_0625_seed1"
# NOTE="v7_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_0625_cluster_warmup_coincide_scenario16_10000_sample0_0625_emaratio0_01_modelema0_99_k1_seed1"
# NOTE="v7_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_0625_sft_scenario16_10000_nosampling_seed1"
# NOTE="NEURIPS_3_2_SAR_t0_125_LORA_llava_lr2e-5_bs1_gradacc32_iter0_03125_score_scenario8_10000_top_fisher0_0625_seed1"
# NOTE="NEURIPS_3_2_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_sft_scenario8_10000_joint"
# NOTE="v12_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_infobudgetprob_0_1_1_scenario18_10000_random0_0625_seed1_2_24"
# NOTE="v12_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_infoBudgetv2_scenario18_10000_random0_0625_seed1_ema0_999"
# NOTE="v25_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_budgetprob2_scenario20_10000_random0_0625_seed1_ema0_9_1_45_v2"
NOTE="ICLR_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_joint_scenario20_10000_random0_0625"
# NOTE=""
# NOTE="QWENv2_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_125_sft_scenario16_10000_random0_0625_seed1"
# NOTE="check_zscore_full_18"
MODE="sft"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b4
RND_SEED=1
SELECTION_METHOD="coincide"
EMBEDDING="representation"
SETUP="joint"
WARMUP_SAMPLE_RATIO=0.0625

SCENARIO=16

MASTER_PORT=22357
SAMPLING="bottom"
EMA_RATIO=0.1

RESUME_TRAIN_ROUND=-1
LOAD_CHECKPOINT="./client_states_v9_NEURIPS_DISJOINT_Memonly_LORA_llava_lr2e-5_bs1_gradacc32_iter0_0625_infoBatch_scenario15_10000_random0_25_seed1/server_model_round6.pth"

# Information
START_EMA_UPDATE_STEP=320
MIN_Z_SCORE_THRESHOLD=2.04
KMEANS=1
MUTUAL_PARAM=1
if [ "$WARMUP_SAMPLE_RATIO" == "0.0625" ]; then
    # MIN_Z_SCORE_THRESHOLD=2.24
    # MIN_Z_SCORE_THRESHOLD=2.1
    MIN_Z_SCORE_THRESHOLD=2.4
    KMEANS=1
elif [ "$WARMUP_SAMPLE_RATIO" == "0.125" ]; then
    MIN_Z_SCORE_THRESHOLD=1.15
    # MIN_Z_SCORE_THRESHOLD=1.75
    KMEANS=2
elif [ "$WARMUP_SAMPLE_RATIO" == "0.25" ]; then
    MIN_Z_SCORE_THRESHOLD=0.67
    # MIN_Z_SCORE_THRESHOLD=1.3
    KMEANS=4
fi


if [ "$SCENARIO" == "16" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.0625
    NUM_TASKS=1
elif [ "$SCENARIO" == "17" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.125
    NUM_TASKS=1
elif [ "$SCENARIO" == "17" ]; then
    EVAL_PERIOD=16000
    NUM_ITER=0.125
    NUM_TASKS=1
fi
# 0.1 -> 1.28, 0.2 -> 0.84, 0.3 -> 0.52

MAX_Z_SCORE_THRESHOLD=5
INFO_UPDATE_MODE="max"
EMA_UPDATE_MODE="none"
EMA_AVERAGE=False
EMA_POOL_SIZE=100


SOFTMAX_UPDATE_TEMP=0.5
# fed args
NUM_ROUNDS=1
NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MEMORY_SIZE=10000000
IS_STREAMONLY=True
IS_SAR=Fakse
IS_SAR_TASK=False

LORA_ENABLE=True
IA3_ENABLE=False

USE_TASK_ID=False
USE_PROMPT=False

SAVE_OPTIM=True

USE_TASK_VECTOR=False
USE_FISHER=False

GENERATOR_OUTPUT_SIZE=1024
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=4
PROMPT_TOP_K=1
MODEL_EMA_RATIO=0.99

BATCHSIZE=1

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=2e-5 
    BITS=16
elif [ "$MODEL_ARCH" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    LR=7e-6
    BITS=16
else
    echo "Undefined setting"
    exit 1
fi

MM_PROJECTOR_LR=0 #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9

# --master_port 29500
# --num_gpus=4
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
GPU=0
# LOAD_CHECKPOINT="client_states_fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100/server_model_round4.pth"
# # --deepspeed ./deepspeed_script/zero2.json \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port $MASTER_PORT \
    train_VLM_CL.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs $NUM_ITER \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps 16 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME \
    --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --decay_ratio $DECAY_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --is_streamonly $IS_STREAMONLY \
    --is_sar $IS_SAR \
    --is_sar_task $IS_SAR_TASK \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --ia3_enable $IA3_ENABLE \
    --use_task_id $USE_TASK_ID \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --ema_ratio $EMA_RATIO \
    --model_ema_ratio $MODEL_EMA_RATIO \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --use_task_vector $USE_TASK_VECTOR \
    --use_fisher $USE_FISHER \
    --fedours False \
    --selection_method $SELECTION_METHOD \
    --embedding $EMBEDDING \
    --k_means $KMEANS \
    --warmup_sample_ratio $WARMUP_SAMPLE_RATIO \
    --start_ema_update_step $START_EMA_UPDATE_STEP \
    --min_z_score_threshold $MIN_Z_SCORE_THRESHOLD \
    --max_z_score_threshold $MAX_Z_SCORE_THRESHOLD \
    --ema_update_mode $EMA_UPDATE_MODE \
    --softmax_update_temp $SOFTMAX_UPDATE_TEMP \
    --sampling $SAMPLING \
    --ema_average $EMA_AVERAGE \
    --ema_pool_size $EMA_POOL_SIZE \
    --info_update_mode $INFO_UPDATE_MODE \
    --setup $SETUP \
    --mutual_param $MUTUAL_PARAM \
    --eval_period $EVAL_PERIOD \
    --resume_train_round $RESUME_TRAIN_ROUND \
    --load_checkpoint $LOAD_CHECKPOINT \
    --output_dir "./results/test/" > ./nohup/${NOTE}.log 2>&1 

# --selection_method $SELECTION_METHOD \

# --load_checkpoint $LOAD_CHECKPOINT \

# --eval_period $EVAL_PERIOD
# lr_scheduler_type
# --include localhost:7 \