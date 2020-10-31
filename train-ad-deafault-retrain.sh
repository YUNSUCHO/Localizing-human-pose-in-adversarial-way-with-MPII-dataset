
python trainmodel-adversarial-mode-exp24.py \
--path mpii \
--modelName  trainmodel-adversarial-with-pretrain-defaultalpha-retrain \
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


#retrain the ad from the saved model(train-ad-default-alpha/model_19_2000.pt) 
#need to concatenate the training loss graph 