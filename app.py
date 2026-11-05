import os
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image, ImageOps
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# Resolve paths relative to this file so the app works regardless of the
# directory from which `python app.py` is launched.
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
VGG_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH =BASE_DIR / "experiment" / "balanced_v2" / "decoder_16.pth"


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
Bootstrap(app)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=0.65)
    submit = SubmitField("Transfer Style")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"NeuralArt inference device: {device}")

if not VGG_PATH.is_file():
    raise FileNotFoundError(f"VGG weights not found: {VGG_PATH}")
if not DECODER_PATH.is_file():
    raise FileNotFoundError(f"Decoder checkpoint not found: {DECODER_PATH}")

# Load checkpoints on CPU first, then move the models to the selected device.
encoder = VGGEncoder(str(VGG_PATH)).to(device).eval()
decoder = Decoder()
decoder_state = torch.load(DECODER_PATH, map_location="cpu")
# Supports either a plain decoder state_dict or the full checkpoint format.
if isinstance(decoder_state, dict) and "decoder" in decoder_state:
    decoder_state = decoder_state["decoder"]
decoder.load_state_dict(decoder_state)
decoder = decoder.to(device).eval()



def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def pad_to_multiple_of_8(tensor):
    """Pad right/bottom so VGG's three pooling stages decode to exact dimensions."""
    height, width = tensor.shape[-2:]
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, height, width


def style_transfer(content_image, style_image, alpha=0.65):
    """Run AdaIN and return an RGB tensor cropped to the content dimensions."""
    alpha = max(0.0, min(1.0, float(alpha)))

    # Keep the original inference resize policy (short side = 512).
    content_transform = transforms.Compose([
        transforms.Resize(768),
        transforms.ToTensor(),
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)


    content_tensor, original_h, original_w = pad_to_multiple_of_8(content_tensor)
    style_tensor, _, _ = pad_to_multiple_of_8(style_tensor)

    with torch.inference_mode():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)

        target_features = adaptive_instance_normalization(
            content_features, style_features
        )
        # AdaIN strength: 0 = content features, 1 = full style features.
        target_features = (
            alpha * target_features + (1.0 - alpha) * content_features
        )
        result = decoder(target_features)

        # Remove padding introduced for the encoder/decoder's 8x spatial scale.
        result = result[..., :original_h, :original_w]
        return result.clamp(0.0, 1.0)


def save_image(image_tensor, path):
    image = image_tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    transforms.ToPILImage()(image).save(path)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = form.content_path.data
    style_filename = form.style_path.data
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if not allowed_file(form.content.data.filename):
                error = "Content image must be PNG, JPG, or JPEG."
            else:
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(UPLOAD_DIR / content_filename)
                form.content_path.data = content_filename

        if form.style.data and form.style.data.filename:
            if not allowed_file(form.style.data.filename):
                error = "Style image must be PNG, JPG, or JPEG."
            else:
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(UPLOAD_DIR / style_filename)
                form.style_path.data = style_filename

        if not error and content_filename and style_filename:
            content_path = UPLOAD_DIR / content_filename
            style_path = UPLOAD_DIR / style_filename
            try:
                with Image.open(content_path) as image:
                    content_image = ImageOps.exif_transpose(image).convert("RGB")
                with Image.open(style_path) as image:
                    style_image = ImageOps.exif_transpose(image).convert("RGB")

                alpha = form.alpha.data if form.alpha.data is not None else 0.5
                result_tensor = style_transfer(content_image, style_image, alpha)

                result_filename = f"stylized_{content_filename}"
                save_image(result_tensor, UPLOAD_DIR / result_filename)
                result_image = result_filename
            except Exception as exc:
                app.logger.exception("Style transfer failed")
                error = f"Style transfer failed: {exc}"
        elif not error:
            error = "Please upload both a content image and a style image."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(BASE_DIR / "examples", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
