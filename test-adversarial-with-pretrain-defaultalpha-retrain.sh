python testImages.py \
--mode val \
--path mpii \
--config config.default_config \
--use_gpu \
--batch_size 1 \
--modelName trainmodel-adversarial-with-pretrain-defaultalpha-retrain/model_49_10000.pt

#baseline : 85.49%
#0(0.858963)
#3(0.857370)
#6(0.85548)
#9( 0.851001)
#12(0.850751)
#15( 0.859167)
#18(0.854754,)
#21( 0.856324)
#25 : 0.851456
#27(0.855437)
#30 : 0.854390
#34 : 0.854709
#35 : 0.857552
#36 : 0.856369
#40 : 0.857598
#43 : 0.857598
#44 : 0.860532
#46 : 0.859031
#48 : 0.835965
#49 : 0.864354 *
#50 : 0.857598
#51 : 0.861510
#52 : 0.861488
#53 : 0.859691
#54 : 0.858053
#55 : 0.858940


