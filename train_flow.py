import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tifffile import imwrite

from diffusers import UNet2DModel, FlowMatchEulerDiscreteScheduler 
from datasets import TiffDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "lodochallenge"
data_path = "lodochallenge/train"
unet_path = "trained/lodochallenge/flow_matching"
sample_path = "flow_matching_samples"
os.makedirs(sample_path, exist_ok=True)

os.makedirs(unet_path, exist_ok=True)
checkpoint_path = os.path.join(unet_path, "checkpoint.pth")

@torch.no_grad()
def sample_flow_matching(unet, scheduler, num_inference_steps=50, batch_size=1):
    unet.eval()

    # Start from prior x ~ N(0,I)
    x = torch.randn(batch_size, 1, 512, 512, device=device)

    # Set scheduler timesteps (e.g. from 1 → 0)
    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        # t is a discrete index; model call is same as training
        t_batch = t.expand(batch_size)  # [B]
        t_batch = t_batch.to(device)

        # Predict velocity
        v_pred = unet(x, t_batch).sample

        # Scheduler step: x_{t-Δt} = x_t - v_pred * Δt
        # FlowMatchEulerDiscreteScheduler.step() returns new sample
        step = scheduler.step(v_pred, t, x)
        x = step.prev_sample  # update state

    return x  # approximately x0 (clean image)

transform = transforms.Compose([transforms.ToTensor()])
dataset = TiffDataset(data_path, transform=transform)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------- 1. Model  ( UNet) ----------
unet = UNet2DModel(
    sample_size=512,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                      "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                    "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

# --------- 2. Flow-matching scheduler (for sampling later) ----------
fm_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

optimizer = optim.AdamW(unet.parameters(), lr=1e-4)
num_epochs = 200
num_train_timesteps = fm_scheduler.config.num_train_timesteps  # e.g. 1000

def save_checkpoint(epoch, model, optimizer, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model, optimizer, path):
    global start_epoch
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting from scratch.")

load_checkpoint(unet, optimizer, checkpoint_path)

# --------- 3. Training loop: FLOW MATCHING ----------
for epoch in range(start_epoch, num_epochs):
    unet.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        # x0: clean images
        x0 = batch.to(device)  # [B,1,H,W]

        # z: Gaussian source / prior sample
        z = torch.randn_like(x0)  # [B,1,H,W]

        # Sample continuous time t ~ Uniform(0,1)
        # shape: [B,1,1,1] so it broadcasts over H,W
        t_cont = torch.rand(x0.shape[0], 1, 1, 1, device=device)

        # Rectified flow interpolant: x_t = (1-t)*x0 + t*z
        xt = (1.0 - t_cont) * x0 + t_cont * z

        # Velocity target: v = dx/dt = z - x0 (constant along path)
        v_target = z - x0

        # Map t in [0,1] to discrete timesteps [0, num_train_timesteps-1]
        # UNet2DModel expects integer timesteps or embeddings compatible with scheduler.
        t_indices = (t_cont[:, 0, 0, 0] * (num_train_timesteps - 1)).long()  # [B]

        # Predict velocity with UNet
        v_pred = unet(xt, t_indices).sample  # [B,1,H,W]

        # Flow-matching loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_pred, v_target)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Flow-matching loss = {loss.item():.4f}")
    save_checkpoint(epoch, unet, optimizer, checkpoint_path)

    # Save generated sample every 5 epochs
    if epoch==0 or (epoch + 1) % 5 == 0:
        gen = sample_flow_matching(unet, fm_scheduler, num_inference_steps=50, batch_size=1)
        imwrite(os.path.join(sample_path, f"flow_matching_sample_epoch{epoch+1}.tif"), gen.cpu().numpy()[0])

# Save final model
unet.save_pretrained(unet_path)