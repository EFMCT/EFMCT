**[Efficient Flow Matching for Sparse-View CT Reconstruction](https://arxiv.org/abs/2603.00205)** (MICCAI 2026)

Jiayang Shi, Lincen Yang, Zhong Li, Tristan van Leeuwen, Daniel M. Pelt, K. Joost Batenburg

Generative models, particularly Diffusion Models (DM), have shown strong potential for Computed Tomography (CT) reconstruction serving as expressive priors for solving ill-posed inverse problems. However, diffusion-based reconstruction relies on Stochastic Differential Equations (SDEs) for forward diffusion and reverse denoising, where such stochasticity can interfere with repeated data consistency corrections in CT reconstruction. Since CT reconstruction is often time-critical in clinical and interventional scenarios, improving reconstruction efficiency is essential. In contrast, Flow Matching (FM) models sampling as a deterministic Ordinary Differential Equation (ODE), yielding smooth trajectories without stochastic noise injection. This deterministic formulation is naturally compatible with repeated data consistency operations. Furthermore, we observe that FM-predicted velocity fields exhibit strong correlations across adjacent steps. Motivated by this, we propose an FM-based CT reconstruction framework (FMCT) and an efficient variant (EFMCT) that reuses previously predicted velocity fields over consecutive steps to substantially reduce the number of Neural network Function Evaluations (NFEs), thereby improving inference efficiency. We provide theoretical analysis showing that the error introduced by velocity reuse is bounded when combined with data consistency operations. Extensive experiments demonstrate that FMCT/EFMCT achieve competitive reconstruction quality while significantly improving computational efficiency compared with diffusion-based methods.

<img src="figures/method.png" style="width:100%;" />

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Environment Requirements](#environment-requirements)
- [Datasets](#datasets)
- [Pretrained Flow Matching Models](#pretrained-flow-matching-models)
- [CT Reconstruction](#ct-reconstruction)
  - [FMCT (no velocity reuse)](#fmct-no-velocity-reuse)
  - [EFMCT (velocity reuse)](#efmct-velocity-reuse)
  - [Reproducing the Paper's Results](#reproducing-the-papers-results)
  - [Trade-off Between Efficiency and Accuracy](#trade-off-between-efficiency-and-accuracy)
  - [Diffusion-Based Baselines](#diffusion-based-baselines)
- [Training Flow Matching from Scratch](#training-flow-matching-from-scratch)
- [Proof of Proposition 1](#proof-of-proposition-1)
- [Citation](#citation)

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
| Low dose challenge | [Flow Matching](https://drive.google.com/drive/folders/1GRqf-a3qHX1wPrtOwRd3WCILsIvx5wim?usp=share_link) |
| Decathlon | [Flow Matching](https://drive.google.com/drive/folders/1TQEEpSZY4oEeWFMckBhi4kDgdz9wpWcH?usp=share_link) |


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

### Reproducing the Paper's Results
The two tables above only show how to switch velocity reuse on and off. The results reported in the paper were obtained with the dataset-specific settings below, which are set at the top of [flow_recon.py](flow_recon.py). Remember to also point `dataset` and `UNet2DModel.from_pretrained(...)` to the matching dataset and pretrained model.

| Dataset | Method | num_inference_steps | skip_after | max_skips_in_a_row | eta | gamma | cg_step_scale | NFE |
| ------- | ------ | ------------------- | ---------- | ------------------ | --- | ----- | ------------- | --- |
| Low dose challenge | FMCT | 25 | 25 | any | any | 0.5 | 1 | 25 |
| Low dose challenge | EFMCT | 50 | 0 | 8 | 1.05 | 0.5 | 1 | ~7 |
| Decathlon | FMCT | 50 | 50 | any | any | 1 | 1 | 50 |
| Decathlon | EFMCT | 100 | 0 | 10 | 1.05 | 1 | 0.25 | ~11 |

All experiments use `cg_inner = 5` and `cg_eps = 1e-4`. Since the initial noise is random, PSNR/SSIM can differ slightly from the reported values between runs.

### Trade-off Between Efficiency and Accuracy
Reconstruction efficiency and reconstruction accuracy are a trade-off, and the settings above are one operating point per dataset rather than universally optimal values. The hyperparameters should be tuned for the specific goal and dataset:

- `num_inference_steps`: more steps generally cost more NFEs and data consistency updates; the best value depends on the dataset and on `cg_step_scale`.
- `skip_after`, `max_skips_in_a_row`, `eta`: reusing the velocity earlier, for more consecutive steps, or with a looser acceptance threshold lowers the NFE, at the risk of lower accuracy.
- `gamma`, `cg_inner`, `cg_step_scale`: balance the data consistency correction against the flow matching prior. When using more sampling steps, a smaller `cg_step_scale` may be needed (e.g. 0.25 for EFMCT with 100 steps on Decathlon).

### Diffusion-Based Baselines
For diffusion-based comparison methods, please refer to the [DM4CT Benchmark](https://github.com/DM4CT/DM4CT) repository for implementation details.

## Training Flow Matching from Scratch
Training a flow matching model from scratch can be performed using [train_flow.py](train_flow.py).

## Proof of Proposition 1
The complete proof of Proposition 1, bounding the error introduced by velocity reuse, is available in [proof.pdf](proof.pdf). It derives the single-step reuse error and the accumulated error over `M` consecutive reuse steps under a `κ`-non-expansive data consistency operation.

## Citation
If you find this work useful, please cite:

```bibtex
@article{shi2026efficient,
  title={Efficient Flow Matching for Sparse-View {CT} Reconstruction},
  author={Shi, Jiayang and Yang, Lincen and Li, Zhong and van Leeuwen, Tristan
          and Pelt, Dani{\"e}l M. and Batenburg, K. Joost},
  journal={arXiv preprint arXiv:2603.00205},
  year={2026}
}
```
