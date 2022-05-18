source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate tf112
python code/trainer.py \
    --bert_model_path ./empty/bert_model.ckpt \
    --bert_config_path ./empty/bert_config.json \
    --bert_vocab_path ./empty/vocab.txt \
    --output_dir ./save
# python code/wrapper.py --bert_config_path ./ecom/bert_config.json --ckpt_to_convert ./save/ecoms-30000 --output_dir ./submit-ecom --max_seq_length 115
