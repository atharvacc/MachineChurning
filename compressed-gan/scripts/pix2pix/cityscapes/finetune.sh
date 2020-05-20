#!/usr/bin/env bash
python train_supernet.py --dataroot database/cityscapes \
  --supernet resnet \
  --log_dir logs/pix2pix/cityscapes/finetune \
  --restore_teacher_G_path logs/pix2pix/cityscapes/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --nepochs 100 --nepochs_decay 150 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --teacher_ngf 96 --config_str 32_32_40_48_48_48_24_24 \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA
