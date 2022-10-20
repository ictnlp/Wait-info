export CUDA_VISIBLE_DEVICES=0,1,2,3

data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL

python train.py --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --save-dir ${modelfile} \
 --max-tokens 8192 --update-freq 1 \
 --fp16 \
 --save-interval-updates 1000 \
 --keep-interval-updates 500 \
 --log-interval 10
