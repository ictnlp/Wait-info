export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
ref_dir=PATH_TO_REFERENCE
testk=TEST_WAIT_K

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 1 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} --sim-decoding > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation