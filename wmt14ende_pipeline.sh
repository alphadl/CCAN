databin=databin/ende/distill_de
src=en
tgt=de

model_path=wmt14ende_model
log_path=wmt14ende_log


# the name should obey this format: range**_apply**, for example: range11_apply123456
model_dir=${model_path}/$1
log_dir=${log_path}/$1

#python preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref $text/train --validpref $text/valid --testpref $text/test --destdir ${output_dir}/data-bin --workers 60 --joined-dictionary

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python train.py ${databin} --arch bert_transformer_seq2seq --share-all-embeddings --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self --max-tokens 8192 --update-freq 8 --weight-decay 0.01 --dropout 0.2 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --ddp-backend=no_c10d --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 1 --save-dir ${model_dir} --ddp-backend=no_c10d --fp16 --keep-last-epochs 1 --no-progress-bar --log-format simple --log-interval 100 --save-interval-updates 4000 --decoder-layers-to-apply-local '1,2,3,4,5,6' --win-size 9 > ${log_dir}.log  2>&1 &

# CUDA_VISIBLE_DEVICES=1 python generate_cmlm.py ${output_dir}/data-bin --path ${model_dir}/checkpoint_best.pt --task translation_self --remove-bpe --max-sentences 10 --decoding-iterations 10 --decoding-strategy mask_predict --max-sentences 100 > rst/${model_dir}.out
