# Compare group sparsity versus not using group sparsity

python word2gm_trainer_repvec.py --min_count 1 --concurrent_steps 2 --rep vec --train_data data/text8 --save_path modelfiles/model_word_char_v2-no_gs --num_samples 2 --char_emb