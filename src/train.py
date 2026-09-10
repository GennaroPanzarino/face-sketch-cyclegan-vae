import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from dataset import FaceSketchDataset
from models import Generator, Discriminator
from losses import adversarialLoss, cycleLoss, KLDLoss, reconstructionLoss

# Parametri di configurazione
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train')

IMAGE_SIZE = 256
LATENT_DIM = 512
MAX_SAMPLES = 3000
BATCH_SIZE = 8

# RESUME TRAINING CONFIGURATION
RESUME_TRAINING = True  # Imposta a True per continuare il training
START_EPOCH = 51  # L'epoca da cui ripartire
ADDITIONAL_EPOCHS = 50  # Quante epoche in più eseguire
TOTAL_EPOCHS = START_EPOCH + ADDITIONAL_EPOCHS - 1  # Es: 51 + 50 - 1 = 100

LR = 2e-4
BETA1 = 0.5

# Loss Lambdas per ResNet
LAMBDA_CYCLE = 10.0
LAMBDA_KL = 0.0
LAMBDA_RECON = 2.0

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"Device utilizzato: {DEVICE}")

    # 1. Caricamento Dati Parallelo
    train_dataset = FaceSketchDataset(root_dir=TRAIN_DIR, max_samples=MAX_SAMPLES, image_size=IMAGE_SIZE)
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Numero di campioni di training: {len(train_dataset)}")

    # 2. Modelli e Ottimizzatori
    G_photo2sketch = Generator(latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)
    G_sketch2photo = Generator(latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)
    D_photo = Discriminator().to(DEVICE)
    D_sketch = Discriminator().to(DEVICE)

    # 3. Caricamento Checkpoint (se RESUME_TRAINING è True)
    if RESUME_TRAINING:
        p2s_path = os.path.join(CHECKPOINT_DIR, 'G_photo2sketch_final.pt')
        s2p_path = os.path.join(CHECKPOINT_DIR, 'G_sketch2photo_final.pt')

        if os.path.exists(p2s_path) and os.path.exists(s2p_path):
            G_photo2sketch.load_state_dict(torch.load(p2s_path, map_location=DEVICE))
            G_sketch2photo.load_state_dict(torch.load(s2p_path, map_location=DEVICE))
            print(f"-> Checkpoint precedenti caricati con successo da {CHECKPOINT_DIR}!")
            print(f"-> Continuo il training dall'epoca {START_EPOCH} fino all'epoca {TOTAL_EPOCHS}.")
        else:
            print("-> Impossibile trovare i file finali dei pesi. Il training ripartirà dall'epoca 1.")

    optimizer_G = optim.Adam(
        list(G_photo2sketch.parameters()) + list(G_sketch2photo.parameters()),
        lr=LR, betas=(BETA1, 0.999)
    )
    optimizer_D_photo = optim.Adam(D_photo.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D_sketch = optim.Adam(D_sketch.parameters(), lr=LR, betas=(BETA1, 0.999))

    scaler = GradScaler(enabled=torch.cuda.is_available())
    history = {'G': [], 'D': []}

    # 4. Training Loop
    start_ep = START_EPOCH if RESUME_TRAINING else 1
    for epoch in range(start_ep, TOTAL_EPOCHS + 1):
        G_photo2sketch.train()
        G_sketch2photo.train()
        D_photo.train()
        D_sketch.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch in train_loader:
            real_photo = batch['photo'].to(DEVICE, non_blocking=True)
            real_sketch = batch['sketch'].to(DEVICE, non_blocking=True)

            # --- Generatori con FP16 ---
            optimizer_G.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                fake_sketch, mu_p2s, logvar_p2s = G_photo2sketch(real_photo)
                fake_photo, mu_s2p, logvar_s2p = G_sketch2photo(real_sketch)

                reconstructed_photo, _, _ = G_sketch2photo(fake_sketch)
                reconstructed_sketch, _, _ = G_photo2sketch(fake_photo)

                loss_adv_photo2sketch = adversarialLoss(D_sketch(fake_sketch), isReal=True)
                loss_adv_sketch2photo = adversarialLoss(D_photo(fake_photo), isReal=True)

                loss_cycle_photo = cycleLoss(real_photo, reconstructed_photo)
                loss_cycle_sketch = cycleLoss(real_sketch, reconstructed_sketch)

                loss_recon = reconstructionLoss(reconstructed_photo, real_photo) + \
                             reconstructionLoss(reconstructed_sketch, real_sketch)

                loss_G = (
                        loss_adv_photo2sketch + loss_adv_sketch2photo +
                        LAMBDA_CYCLE * (loss_cycle_photo + loss_cycle_sketch) +
                        LAMBDA_RECON * loss_recon
                )

            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(list(G_photo2sketch.parameters()) + list(G_sketch2photo.parameters()), 5.0)
            scaler.step(optimizer_G)
            scaler.update()

            # --- Discriminatore Foto ---
            optimizer_D_photo.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                loss_D_photo_real = adversarialLoss(D_photo(real_photo), isReal=True)
                loss_D_photo_fake = adversarialLoss(D_photo(fake_photo.detach()), isReal=False)
                loss_D_photo = 0.5 * (loss_D_photo_real + loss_D_photo_fake)

            scaler.scale(loss_D_photo).backward()
            scaler.step(optimizer_D_photo)
            scaler.update()

            # --- Discriminatore Schizzio ---
            optimizer_D_sketch.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                loss_D_sketch_real = adversarialLoss(D_sketch(real_sketch), isReal=True)
                loss_D_sketch_fake = adversarialLoss(D_sketch(fake_sketch.detach()), isReal=False)
                loss_D_sketch = 0.5 * (loss_D_sketch_real + loss_D_sketch_fake)

            scaler.scale(loss_D_sketch).backward()
            scaler.step(optimizer_D_sketch)
            scaler.update()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += (loss_D_photo.item() + loss_D_sketch.item())

        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        history['G'].append(avg_g_loss)
        history['D'].append(avg_d_loss)

        print(f"Epoca [{epoch}/{TOTAL_EPOCHS}]  Loss_G: {avg_g_loss:.4f}  Loss_D: {avg_d_loss:.4f}")

        # Salva ogni 10 epoche e all'ultima
        if epoch % 10 == 0 or epoch == TOTAL_EPOCHS:
            torch.save(G_photo2sketch.state_dict(), os.path.join(CHECKPOINT_DIR, f'G_photo2sketch_epoch{epoch}.pt'))
            torch.save(G_sketch2photo.state_dict(), os.path.join(CHECKPOINT_DIR, f'G_sketch2photo_epoch{epoch}.pt'))
            torch.save(G_photo2sketch.state_dict(), os.path.join(CHECKPOINT_DIR, 'G_photo2sketch_final.pt'))
            torch.save(G_sketch2photo.state_dict(), os.path.join(CHECKPOINT_DIR, 'G_sketch2photo_final.pt'))
            print(f"  -> Checkpoint salvato all'epoca {epoch}")

    print("Training completato e pesi salvati con successo.")


if __name__ == '__main__':
    main()