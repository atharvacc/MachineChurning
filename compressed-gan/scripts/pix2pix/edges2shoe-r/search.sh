#!/usr/bin/env bash
python search.py --dataroot database/edges2shoes-r \
  --restore_G_path logs/pix2pix/edges2shoes-r/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/edges2shoes-r/supernet/result.pkl \
  --ngf 48 --batch_size 32 \
  --config_set channels-48 \
  --real_stat_path real_stat/edges2shoes-r_B.npz
