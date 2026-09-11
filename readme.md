Face-Sketch CycleGAN is an unsupervised image-to-image translation system designed to transform face photographs into hand-drawn sketches and the other way around. It utilizes a Cycle-Consistent Adversarial Network (CycleGAN) with ResNet-based generators and does not require paired training data.

You'll find an interactive Gradio web app that allows you to upload a photo to generate a corresponding sketch. Alternatively, you can draw a sketch directly on a canvas to create the matching photo.

The architecture is built on the CycleGAN framework (Zhu et al., 2017) and incorporates ResNet blocks in the generators (Johnson et al., 2016) to improve output fidelity. There are two generators, one converting photos to sketches and the other doing the reverse. Each generator contains six residual blocks. Additionally, there are two PatchGAN discriminators that assess realism at the patch level, one for each domain. The loss functions used in the training include adversarial losses (LSGAN-style and MSE), cycle-consistency loss (L1), and reconstruction loss (L1).

The model's flow is as follows:

```
photo ──► [Generator: photo→sketch] ──► fake sketch ──► [Generator: sketch→photo] ──► reconstructed photo
                                              │
                                              ▼
                                    [Discriminator: sketch]
```

For training, the model uses the dataset available at [Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches), which consists of photo–sketch pairs. Although the dataset provides paired data, the training adheres to the unpaired CycleGAN paradigm by shuffling those pairs, aligning with the method’s intention for domains lacking direct correspondence.

The project's structure is organized as follows:

```
├── data/
│   └── raw/                  # dataset (train/val/test, photos/ and sketches/)
├── models/                   # saved model checkpoints (.pt)
├── outputs/                  # generated samples, loss curves
├── src/
│   ├── dataset.py            # PyTorch Dataset/DataLoader
│   ├── models.py             # Generator (ResNet) and Discriminator (PatchGAN)
│   ├── losses.py             # adversarial, cycle-consistency, reconstruction losses
│   ├── train.py              # training loop
│   └── app.py                # Gradio demo
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

To set up the environment, you can either create a local virtual environment or use Docker.

For a local virtual environment, run:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # adjust CUDA version as needed
pip install gradio numpy pillow
```

If you prefer Docker, ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend and an NVIDIA GPU with up-to-date drivers. Build the Docker image with:

```bash
docker build -t cyclegan-im2im .
```

To train the model, execute:

```bash
python src/train.py
```

You can adjust key hyperparameters, which are defined at the top of `train.py`. These include `IMAGE_SIZE`, `MAX_SAMPLES`, `BATCH_SIZE`, `NUM_EPOCHS`, `LAMBDA_CYCLE`, and `LAMBDA_RECON`. Checkpoints will be saved to the `models/` directory, and a loss curve plot will be generated and saved as `outputs/loss_curve.png` once training concludes.

For running the demo, use the following commands:

**Locally:**

```bash
python src/app.py
```

**With Docker:**

```bash
docker run --rm --gpus all -p 7860:7860 -v ${PWD}/models:/app/models cyclegan-im2im
```

Then just open [http://localhost:7860](http://localhost:7860) in your browser.