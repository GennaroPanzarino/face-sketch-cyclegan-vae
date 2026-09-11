import os
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

from models import Generator

# Definizione del dispositivo e parametri
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256

# Risoluzione dinamica dei percorsi
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')

photo2sketch_path = os.path.join(CHECKPOINT_DIR, 'G_photo2sketch_final.pt')
sketch2photo_path = os.path.join(CHECKPOINT_DIR, 'G_sketch2photo_final.pt')

# Inizializzazione e caricamento dei modelli
G_photo2sketch = Generator().to(DEVICE)
G_photo2sketch.load_state_dict(torch.load(photo2sketch_path, map_location=DEVICE))
G_photo2sketch.eval()

G_sketch2photo = Generator().to(DEVICE)
G_sketch2photo.load_state_dict(torch.load(sketch2photo_path, map_location=DEVICE))
G_sketch2photo.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def tensor_to_image(tensor, target_size=(256, 256)):
    """
    Rimuove eventuali valori NaN/Inf e denormalizza da [-1, 1] a [0, 1].
    """
    tensor = tensor.squeeze(0).detach().cpu()

    # Risolve il problema del warning 'invalid value encountered in cast'
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Riporta da [-1, 1] a [0, 1]
    tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0.0, 1.0)

    img = transforms.ToPILImage()(tensor)
    return img.resize(target_size, Image.BILINEAR)


def prepare_sketch_image(raw_input, invert_colors=False):
    """
    Gestisce la trasparenza e converte l'input in una PIL Image corretta.
    """
    if isinstance(raw_input, Image.Image):
        img = raw_input
    elif isinstance(raw_input, dict):
        immagine_array = raw_input.get("composite", raw_input.get("background"))
        img = Image.fromarray(immagine_array.astype(np.uint8))
    elif isinstance(raw_input, np.ndarray):
        img = Image.fromarray(raw_input.astype(np.uint8))
    else:
        img = raw_input

    # Gestione del canale Alpha (trasparenza)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=alpha)
        img = bg
    else:
        img = img.convert('RGB')

    if invert_colors:
        img = ImageOps.invert(img)

    return img


def foto_a_schizzio(immagine_input):
    if immagine_input is None:
        return None
    img = immagine_input.convert('RGB')
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output, _, _ = G_photo2sketch(x)
    return tensor_to_image(output)


def schizzio_disegnato_a_foto(disegno, inverti_colori):
    if disegno is None:
        return None

    img = prepare_sketch_image(disegno, invert_colors=inverti_colori)
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _, _ = G_sketch2photo(x)
    return tensor_to_image(output)


def schizzio_caricato_a_foto(schizzio_input, inverti_colori):
    if schizzio_input is None:
        return None

    img = prepare_sketch_image(schizzio_input, invert_colors=inverti_colori)
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output, _, _ = G_sketch2photo(x)
    return tensor_to_image(output)


# Interfaccia Gradio
with gr.Blocks(title="Face Sketch CycleGAN+VAE") as demo:
    gr.Markdown("# Foto ↔ Schizzio — CycleGAN con generatori VAE")

    with gr.Tab("Foto → Schizzio"):
        with gr.Row():
            input_foto = gr.Image(type="pil", label="Carica una foto", height=300)
            output_schizzio = gr.Image(type="pil", label="Schizzio generato", height=300)
        btn1 = gr.Button("Genera schizzio")
        btn1.click(fn=foto_a_schizzio, inputs=input_foto, outputs=output_schizzio)

    with gr.Tab("Disegna uno schizzio → Foto"):
        with gr.Row():
            sketch_pad = gr.Sketchpad(
                type="numpy",
                label="Disegna qui con il mouse",
                brush=gr.Brush(default_size=3, colors=["#000000"], color_mode="fixed"),
                canvas_size=(256, 256)
            )
            output_foto_da_disegno = gr.Image(type="pil", label="Foto generata", height=300)

        chk_invert1 = gr.Checkbox(label="Inverti colori dello schizzo (per dataset a sfondo nero)", value=False)
        btn2 = gr.Button("Genera da disegno")
        btn2.click(
            fn=schizzio_disegnato_a_foto,
            inputs=[sketch_pad, chk_invert1],
            outputs=output_foto_da_disegno
        )

    with gr.Tab("Carica uno schizzio → Foto"):
        with gr.Row():
            input_schizzio = gr.Image(type="pil", label="Carica un file schizzio", height=300)
            output_foto_da_file = gr.Image(type="pil", label="Foto generata", height=300)

        chk_invert2 = gr.Checkbox(label="Inverti colori dello schizzo (per dataset a sfondo nero)", value=False)
        btn3 = gr.Button("Genera da file caricato")
        btn3.click(
            fn=schizzio_caricato_a_foto,
            inputs=[input_schizzio, chk_invert2],
            outputs=output_foto_da_file
        )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)