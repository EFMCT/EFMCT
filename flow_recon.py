import os
import torch
import astra
from torchvision import transforms
from tqdm import tqdm
from tifffile import imwrite
import numpy as np
from diffusers import UNet2DModel, FlowMatchEulerDiscreteScheduler  
from datasets import TiffDataset
from forward_operators_ct import Operator, NoNoise
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (0,1)
])

# Create dataset
dataset = TiffDataset("lodochallenge", transform=transform)

num_angles = 40 
angles =np.linspace(0, np.pi, num_angles)

n_rows=512
n_cols=512
n_slices=1

vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, n_rows, angles)
A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)
noiser = NoNoise()

unet = UNet2DModel.from_pretrained("trained/lodochallenge/flow_matching").to(device)
unet.eval()
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

# -----------------------------
# 0) Inference hyperparameters and setup 
# -----------------------------
num_inference_steps = 50

gamma = 1      # Tikhonov term (data vs prior balance)
cg_inner = 5     # max CG iterations per diffusion step
cg_eps = 1e-4     # CG stopping tolerance
cg_step_scale = 1  # how strongly to apply CG correction to x

skip_after = 50     # set to larger than num_inference_steps to disable skipping entirely
max_skips_in_a_row = 10
eta = 1.05         # threshold for accepting reusing velocity based on data consistency improvement

if skip_after >= num_inference_steps:
    print("Skipping disabled. UNet will be evaluated at every step.")
    save_path = f"fmct/lodochallenge/{num_angles}projs"
else:
    print(f"Skipping enabled. UNet will be evaluated at most every {skip_after} steps, with a maximum of {max_skips_in_a_row} skips in a row.")
    save_path = f"efmct/lodochallenge/{num_angles}projs"

os.makedirs(save_path, exist_ok=True)

for m in range(len(dataset)):
    img = dataset[m]
    img = torch.tensor(img, device='cuda')

    # forward pass
    y = A(img)
    y_n = noiser(y)

    # Start from prior x ~ N(0,I)
    x = torch.randn(1, 1, 512, 512, device=device)  # current estimate x_k

    # Set scheduler timesteps (e.g. from 1 → 0)
    scheduler.set_timesteps(num_inference_steps)

    # initialization for skipping logic
    v_prev = None
    skips_in_a_row = 0
    nfe = 0

    t_start = time.time()
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        t_batch = t.expand(1).to(device)

        # Don't skip the last step
        near_end = (i >= len(scheduler.timesteps) - 1)

        # determine whether to skip UNet evaluation and reuse previous velocity
        use_skip = (
            (v_prev is not None) and
            (i >= skip_after) and
            (skips_in_a_row < max_skips_in_a_row) and
            (not near_end)
        )

        did_unet = True
        v_used = None  

        # Store original x to compute x0_pred from pre-step state
        x_before_step = x.clone()

        # -----------------------------
        # 1) Either compute v_pred (NFE) or reuse/extrapolate it (0 NFE)
        # -----------------------------
        if use_skip:
            with torch.no_grad():
                v_used = v_prev

                # Propose one FM step using v_used (NO UNet)
                sigma = scheduler.sigmas[i]
                sigma_next = scheduler.sigmas[i + 1]
                x_prop = x_before_step + (sigma_next - sigma) * v_used

                # adaptive control on data consistency of the proposal
                dc_prev = torch.norm(A(x_before_step) - y_n)
                dc_prop = torch.norm(A(x_prop) - y_n)

                accept = (dc_prop <= eta * dc_prev ) 

            if accept:
                x = x_prop
                did_unet = False
                skips_in_a_row += 1
            else:
                did_unet = True
                skips_in_a_row = 0

        else:
            did_unet = True
            skips_in_a_row = 0

        if did_unet:
            with torch.no_grad():
                v_pred = unet(x_before_step, t_batch).sample
                nfe += 1
                v_used = v_pred

                # FM step using true v_pred
                x = scheduler.step(v_used, t, x_before_step).prev_sample

            # Update velocity history only when we actually ran UNet
            if v_prev is None:
                v_prev = v_pred.detach()
            else:
                v_prevprev = v_prev
                v_prev = v_pred.detach()
        
        x0_pred = x - scheduler.sigmas[i+1] * v_used
        x0_pred = torch.clamp(x0_pred, -1, 1)  # Clip to valid range
        
        # Initialize x_cg to x0_pred (will be updated in CG block if it runs)
        x_cg = x0_pred.clone()

        # -----------------------------
        # Conjugate gradient data step
        # -----------------------------
        # We solve: (A^T A + gamma I) x_cg = A^T y_n + gamma x0_pred
        if i <= int(len(scheduler.timesteps)):
            with torch.no_grad():
                # Initial CG guess: start from x0_pred
                x_cg = x0_pred.clone()

                # Right-hand side b = A^T y_n + gamma * x0_pred
                b = A.T(y_n) + gamma * x0_pred

                # Residual r = b - (A^T A + gamma I) x_cg
                Ax_cg = A(x_cg)
                ATA_x_cg = A.T(Ax_cg)
                r = b - (ATA_x_cg + gamma * x_cg)

                p = r.clone()
                rsold = torch.sum(r * r)

                for j in range(cg_inner):
                    Ap = A.T(A(p)) + gamma * p
                    denom = torch.sum(p * Ap)
                    # Avoid division by zero
                    if denom.abs() < 1e-12:
                        break

                    alpha = rsold / denom
                    x_cg = x_cg + alpha * p
                    r = r - alpha * Ap

                    rsnew = torch.sum(r * r)
                    if torch.sqrt(rsnew) < cg_eps:
                        break

                    p = r + (rsnew / rsold) * p
                    rsold = rsnew

        delta_x0 = x_cg - x0_pred
        x = x + cg_step_scale * delta_x0

    t_end = time.time()
    print(f"Total time for {num_inference_steps} steps: {t_end - t_start:.2f} seconds")
    print("Total function evaluations (NFE):", nfe)

    imwrite(os.path.join(save_path, f"{m:03d}.tif"), x.detach().cpu().numpy()[0])

    recon = x.detach().cpu().numpy().squeeze()
    gt = img.detach().cpu().numpy().squeeze()
    psnr_val = psnr(gt, recon, data_range=2.0)
    ssim_val = ssim(gt, recon, data_range=2.0)
    error = A(x) - y_n
    error_norm = torch.norm(error)
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"Data Fit Error: {error_norm:.4f}")

