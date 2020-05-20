# Training Tutorial
## Prerequisites

* Linux
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Preparations

Please refer to our [README](../README.md) for the installation, dataset preparations, and the evaluation (FID and mAP).

### Pipeline

Below we show the full pipeline for compressing pix2pix and cycleGAN models. **We provide pre-trained models after each step. You could use the pretrained models to skip some steps.** For more training details, please refer to [Appendix 6.1 Complete Pipeline](https://arxiv.org/pdf/2003.08936.pdf) of our paper.

In fact, several steps including "Train a MobileNet Teacher Model", "Pre-distillation", and "Fine-tuning the Best Model" may be omitted from the whole pipeline. We will provide a simplified pipeline soon.

## Pix2pix Model Compression

We will show the whole pipeline on `edges2shoes-r` dataset. You could change the dataset name to other datasets (`map2sat` and `cityscapes`).

##### Train a MobileNet Teacher Model

Train a MobileNet-style teacher model from scratch.
```shell
bash scripts/pix2pix/edges2shoes-r/train_mobile.sh
```
We provide a pre-trained teacher for each dataset. You could download the pre-trained model by
```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage mobile
```

and test the model by

```shell
bash scripts/pix2pix/edges2shoes-r/test_mobile.sh
```

##### Pre-distillation

(Optional) Distill and prune the original MobileNet-style model to make the model compact.

```shell
bash scripts/pix2pix/edges2shoes-r/distill.sh
```

We provide a pre-distilled teacher for each dataset. You could download the pre-distilled model by

```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage distill
```

and test the model by

```bash
bash scripts/pix2pix/edges2shoes-r/test_distill.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from a pre-trained student model to search for the efficient architectures.

```shell
bash scripts/pix2pix/edges2shoes-r/train_supernet.sh
```

We provide a trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage supernet
```

##### Select the Best Model

Evaluate all the candidate sub-networks given a specific configuration (e.g., MAC, FID).

```shell
bash scripts/pix2pix/edges2shoes-r/search.sh
```

The search result will be stored in the python `pickle` form. The pickle file is a python `list` object that stores all the candidate sub-networks information, whose element is a python `dict ` object in the form of

```
{'config_str': $config_str, 'macs': $macs, 'fid'/'mAP': $fid_or_mAP}
```

such as

```python
{'config_str': '32_32_48_32_48_48_16_16', 'macs': 4993843200, 'fid': 25.224261423597483}
```

`'config_str'` is a channel configuration description to identify a specific subnet within the "once-for-all" network.

You could use our auxiliary script `select_arch.py` to select the architecture you want.

```shell
python select_arch.py --macs 5.7e9 --fid 30 \  # macs <= 5.7e9(10x), fid >= 30
--pkl_path logs/pix2pix/edges2shoes-r/supernet/result.pkl
```

##### Fine-tuning the Best Model

(Optional) Fine-tune a specific subnet within the pre-trained "once-for-all" network. To further improve the performance of your chosen subnet, you may need to fine-tune the subnet. For example, if you want to fine-tune a subnet within the "once-for-all" network with `'config_str': 32_32_48_32_48_48_16_16`, use the following command:

```shell
bash scripts/pix2pix/edges2shoes-r/finetune.sh 32_32_48_32_48_48_16_16
```

##### Export the Model

Extract a subnet from the "once-for-all" network. We provide a code `export.py` to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `32_32_48_32_48_48_16_16`, then you can export the model by this command:

```shell
bash scripts/pix2pix/edges2shoes-r/export.sh 32_32_48_32_48_48_16_16
```

##### Cityscapes

For the Cityscapes dataset, you need to specify additional options to support mAP evaluation while training. Please refer to the scripts in [scripts/pix2pix/cityscapes](../scripts/pix2pix/cityscapes) for more details.

## CycleGAN Model Compression

The whole pipeline is almost identical to pix2pix. We will show the pipeline on `horse2zebra` dataset.

##### Train a MobileNet Teacher Model

Train a MobileNet-style teacher model from scratch.

```shell
bash scripts/cycle_gan/horse2zebra/train_mobile.sh
```

We provide a pre-trained teacher model for each dataset. You could download the model using

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra --stage mobile
```

and test the model by

```shell
bash scripts/cycle_gan/horse2zebra/test_mobile.sh
```

##### Pre-distillation

(Optional) Distill and prune the MobileNet-style model to make the model compact.

```shell
bash scripts/cycle_gan/horse2zebra/distill.sh
```

We provide a pre-distilled teacher for each dataset. You could download the pre-distilled model by

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra --stage distill
```

and test the model by

```bash
bash scripts/cycle_gan/horse2zebra/test_distill.sh
```

##### "Once-for-all" Network Training

Train a "once-for-all" network from a pre-trained student model to search for the efficient architectures.

```shell
bash scripts/cycle_gan/horse2zebra/train_supernet.sh
```

We provide a pre-trained once-for-all network for each dataset. You could download the model by

```shell
python scripts/download_model.py --model cycle_gan --task horse2zebra --stage supernet
```

##### Select the Best Model

Evaluate all the candidate sub-networks given a specific configuration (e.g., MACs and FID).

```shell
bash scripts/cycle_gan/horse2zebra/search.sh
```
You could also use our auxiliary script `select_arch.py` to select the architecture you want. The usage is the same as pix2pix.

##### Fine-tuning the Best Model

(Optional) Fine-tune a specific subnet within the pre-trained "once-for-all" network. For example, if you want to fine-tune a subnet within the "once-for-all" network with `'config_str': 32_32_48_32_48_48_16_16`, try  the following command:

```shell
bash scripts/cycle_gan/horse2zebra/finetune.sh 16_16_32_16_32_32_16_16
```

During our experiments, we observe that fine-tuning the model on horse2zebra increases FID.  **You may skip the fine-tuning.**

##### Export the Model

Extract a subnet from the supernet. We provide a code `export.py` to extract a specific subnet according to a configuration description. For example, if the `config_str` of your chosen subnet is `16_16_32_16_32_32_16_16`, then you can export the model by this command:

```shell
bash scripts/cycle_gan/horse2zebra/export.sh 16_16_32_16_32_32_16_16
```
