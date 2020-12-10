databin=databin/ende/distill_de
src=en
tgt=de

model_path=wmt14ende_model
log_path=wmt14ende_log


# the name should obey this format: range**_apply**, for example: range11_apply123456
model_dir=${model_path}/$1
log_dir=${log_path}/$1

CUDA_VISIBLE_DEVICES=$2 python generate_cmlm.py ${databin} --path ${model_dir}/checkpoint_***.pt --task translation_self --remove-bpe --decoding-iterations 10 --decoding-strategy mask_predict --max-sentences 90 > wmt14ende_rst/$1/out.out

sh scripts/BLEU.sh wmt14/full en de wmt14ende_rst/$1/out.out
