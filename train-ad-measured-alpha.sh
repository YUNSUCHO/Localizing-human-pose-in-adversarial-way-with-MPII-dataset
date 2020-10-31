
python trainmodel-adversarial-mode-exp24.py \
--path mpii \
--modelName trainmodel-adversarial-with-pretrain \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--gpu_device 0 \
--lr .00025 \
--print_every 1 \
--train_split 0.9167 \
--loss mse \
--optimizer_type Adam \
--epochs 230 \
--dataset  'mpii'


#with 11000 of MPII images
#pre-trained gen : 82% accuracy
#pre-trained discriminator

