## Train
```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 train.py \
  --deepspeed ds_zero3_config.json \
  --output_dir generator-qwen2-7b-ft \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --save_steps 500 \
  --dataset_name MrLight/rag-visa-datasets-v2 \
  --bf16 \
  --tf32 True \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 8
```

## Train Lora
```bash
deepspeed --master_port 60000 train.py \
  --deepspeed ds_zero3_config.json \
  --output_dir qwen2-7b-lora-wiki-2e \
  --lora \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --save_steps 4000 \
  --dataset_name wiki \
  --bf16 \
  --tf32 True \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.02
```