

import os
import glob
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from utils.util import initialize_weights, cross_correlation, gen_plot, gradient_penalty
from model.Generator import Generator
from model.Discriminator import Discriminator
from dataset import dataset
from tqdm import tqdm
from fastdtw import fastdtw
from torchvision.transforms import ToTensor
import PIL.Image

# ── Argument parsing
parser = argparse.ArgumentParser(description="Train WGAN-GP on EMG data")
parser.add_argument("--start-epoch",   type=int, default=1,
                    help="Epoch to start from (inclusive)")
parser.add_argument("--end-epoch",     type=int, required=True,
                    help="Epoch to end at (inclusive)")
parser.add_argument("--checkpoint-dir",type=str, default="checkpoints",
                    help="Directory for saving/loading checkpoints")
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch   = args.end_epoch
ckpt_dir    = args.checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)


device           = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE    = 1e-4
BATCH_SIZE       = 64
IMAGE_SIZE1      = 8
IMAGE_SIZE2      = 24
CHANNELS_IMG     = 1
Z_DIM            = 100
NUM_EPOCHS = 3000
FEATURES_CRITIC  = 16
FEATURES_GEN     = 16
CRITIC_ITERS     = 5
LAMBDA_GP        = 10
EMBED_SIZE       = 100
NUM_CLASSES      = 8
NUM_WORKERS      = 2
SAVE_INTERVAL    = 50   # epochs between checkpoints
ROOT             = "data/noise/CA"
BASE_LOG_PATH    = "Experiment_data/TestGAN/"
BASE_WEIGHT_PATH = "weight/TestGan/"

scaler = GradScaler(device_type="cuda") if device=="cuda" else None

ds = dataset(
    root=ROOT,
    image_size1=IMAGE_SIZE1,
    image_size2=IMAGE_SIZE2,
    all=True,
    train=True,
    channel=0,
    generated_CA_root="Generated CA"
)
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device=="cuda")
)


writer = SummaryWriter(BASE_LOG_PATH)

gen    = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, IMAGE_SIZE1, IMAGE_SIZE2, EMBED_SIZE, NUM_CLASSES).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, IMAGE_SIZE1, IMAGE_SIZE2, NUM_CLASSES).to(device)
initialize_weights(gen)
initialize_weights(critic)

# compile if available
if hasattr(torch, "compile"):
    gen    = torch.compile(gen)
    critic = torch.compile(critic)

opt_gen    = optim.Adam(gen.parameters(),    lr=LEARNING_RATE, betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))


ckpts = sorted(glob.glob(f"{ckpt_dir}/checkpoint_epoch*.pt"))
if ckpts:
    state = torch.load(ckpts[-1], map_location=device)
    gen.load_state_dict(state["gen_state"])
    critic.load_state_dict(state["disc_state"])
    opt_gen.load_state_dict(state["opt_gen"])
    opt_critic.load_state_dict(state["opt_disc"])
    start_epoch = state["epoch"] + 1
    print(f"Resuming from {ckpts[-1]}, starting at epoch {start_epoch}")

x     = torch.linspace(-torch.pi, torch.pi, Z_DIM, device=device)
sin_x = torch.sin(x)
fixed_noise = torch.randn(32, Z_DIM, 1, 1, device=device)

gen.train()
critic.train()

for epoch in range(start_epoch, end_epoch + 1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{end_epoch}", leave=False)
    for real, labels in pbar:
        real, labels = real.to(device), labels.to(device)
        bs = real.size(0)

        # Critic updates
        for _ in range(CRITIC_ITERS):
            opt_critic.zero_grad()
            noise = (0.1*torch.randn(bs, Z_DIM, device=device) + 0.9*sin_x).view(bs, Z_DIM,1,1)

            # WGAN loss
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                fake = gen(noise, labels)
                D_real = critic(real, labels).view(-1)
                D_fake = critic(fake.detach(), labels).view(-1)
                loss_w = -(D_real.mean() - D_fake.mean())

            # Fixed gradient penalty
            alpha = torch.rand(bs,1,1,1,device=device)
            interp = (alpha*real + (1-alpha)*fake).requires_grad_(True)
            scores = critic(interp, labels).view(-1)
            grads  = torch.autograd.grad(
                outputs=scores,
                inputs=interp,
                grad_outputs=torch.ones_like(scores),
                create_graph=True, retain_graph=False
            )[0].view(bs,-1)
            gp = ((grads.norm(2,dim=1)-1)**2).mean()

            loss_critic = loss_w + LAMBDA_GP * gp
            if scaler:
                scaler.scale(loss_critic).backward()
                scaler.unscale_(opt_critic)
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.0)
                scaler.step(opt_critic)
                scaler.update()
            else:
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.0)
                opt_critic.step()

     
        opt_gen.zero_grad()
        with autocast(device_type="cuda", enabled=(device=="cuda")):
            fake2 = gen(noise, labels)
            loss_gen = -critic(fake2, labels).view(-1).mean()
        if scaler:
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
        else:
            loss_gen.backward()
            opt_gen.step()

        pbar.set_postfix({
            "D_loss": f"{loss_critic.item():.4f}",
            "G_loss": f"{loss_gen.item():.4f}"
        })

    # Checkpoint every SAVE_INTERVAL or at end
    if epoch % SAVE_INTERVAL == 0 or epoch == end_epoch:
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch":     epoch,
            "gen_state": gen.state_dict(),
            "disc_state": critic.state_dict(),
            "opt_gen":    opt_gen.state_dict(),
            "opt_disc":   opt_critic.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


print("Training complete.")  


