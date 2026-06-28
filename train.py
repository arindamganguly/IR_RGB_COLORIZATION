import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.satellite_dataset import SatelliteDataset
from models.generator import Generator
from models.discriminator import Discriminator

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Correct project path fix
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

ir_path = os.path.join(base_dir, "data/train/ir")
rgb_path = os.path.join(base_dir, "data/train/rgb")

print("IR path:", ir_path)
print("RGB path:", rgb_path)

# -------------------------
# Dataset
# -------------------------
dataset = SatelliteDataset(ir_path, rgb_path)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------------------------
# Models
# -------------------------
G = Generator().to(device)
D = Discriminator().to(device)

# -------------------------
# Optimizers
# -------------------------
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# -------------------------
# Loss functions
# -------------------------
bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

# -------------------------
# Training
# -------------------------
epochs = 10   # keep small first

for epoch in range(epochs):

    for ir, rgb in loader:

        ir = ir.to(device)
        rgb = rgb.to(device)

        # ---------------------
        # Generator forward
        # ---------------------
        fake_rgb = G(ir)

        # ---------------------
        # Train Discriminator
        # ---------------------
        real_pred = D(ir, rgb)
        fake_pred = D(ir, fake_rgb.detach())

        real_loss = bce(real_pred, torch.ones_like(real_pred))
        fake_loss = bce(fake_pred, torch.zeros_like(fake_pred))

        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        fake_pred = D(ir, fake_rgb)

        adv_loss = bce(fake_pred, torch.ones_like(fake_pred))
        recon_loss = l1(fake_rgb, rgb)

        g_loss = adv_loss + 100 * recon_loss

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# -------------------------
# Save model
# -------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(G.state_dict(), "checkpoints/generator.pth")

print("Training complete. Model saved.")