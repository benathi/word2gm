#python word2gm_trainer.py --train_data data/text8 --save_path modelfiles/word2gm
#python word2vec_optimize.py --train_data data/text8 --eval_data evaluation_data/questions-words.txt --save_path modelfiles/word2vec
#python word2gm_trainer_repvec.py --rep vec --train_data data/text8 --save_path modelfiles/word2gm_repvec
python word2gm_trainer_repvec.py --rep gm --train_data data/text8 --save_path modelfiles/word2gm_repvec