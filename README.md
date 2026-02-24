**Efficient Flow Matching for Sparse-View CT Reconstruction**

This repository contains the implementation of FMCT/EFMCT, flow matching-based methods for sparse-view CT reconstruction.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Environment Requirements](#environment-requirements)
- [Datasets](#datasets)
- [Pretrained Flow Matching Models](#pretrained-diffusion-models)
- [CT Reconstruction](#ct-reconstruction-methods)
- [Training Flow Matching from Scratch](#training-diffusion-models-from-scratch)

## Environment Requirements
- At least one Nvidia GPU for training/inference.
- Main dependencies are `pytorch, diffusers, astra-toolbox, tifffile`.

We provide the conda [configuration](environment.yml) to create the same environment for the benchmark.

Create and activate the main Conda environment:

```bash
conda env create -f environment.yml
conda activate efmct
```

## Datasets
- The low dose challenge dataset: [Low Dose Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/) 
- The decathlon segmentation challenge dataset: [Decathlon](http://medicaldecathlon.com)

A single example image from the Low Dose dataset is included in [here](lodochallenge/L506_000.tif) for quick testing. 

## Pretrained Flow Matching Models
| Dataset | Pretrained model |
|---------|---------------------------|
| Low dose challenge | [Flow Matching](https://drive.google.com/drive/folders/1TQEEpSZY4oEeWFMckBhi4kDgdz9wpWcH?usp=share_link) |
| Decathlon | [Flow Matching](https://drive.google.com/drive/folders/1GRqf-a3qHX1wPrtOwRd3WCILsIvx5wim?usp=share_link) |


## CT Reconstruction
Reconstruction using pretrained models can be executed with [flow_recon.py](flow_recon.py). Both FMCT and EFMCT are implemented in the same script. 

### FMCT (no velocity reuse)
| Parameter | Value |
| --------- | ----- |
| num_inference_steps | 50 |
| skip_after | 50 |
| max_skips_in_a_row| any |
| eta | any |

### EFMCT (velocity reuse)
| Parameter | Value |
| --------- | ----- |
| num_inference_steps | 50 |
| skip_after | 0 |
| max_skips_in_a_row| 10 |
| eta | 1.05 |

### Diffusion-Based Baselines
For diffusion-based comparison methods, please refer to the [DM4CT Benchmark](https://github.com/DM4CT/DM4CT) repository for implementation details.

## Training Flow Matching from Scratch
Training a flow matching model from scratch can be performed using [train_flow.py](train_flow.py).