


python pose-discriminator-trainmodel23upsampled.py \
--path mpii \
--modelName  mpii-pretrain-posediscriminator_0 \
--config config.default_config \
--batch_size 1 \
--use_gpu \
--gpu_device 0 \
--lr .00025 \
--print_every 5000 \
--train_split 0.90 \
--loss mse \
--optimizer_type Adam \
--epochs 230 \
--dataset  'mpii' 
