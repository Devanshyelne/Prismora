import os
import re
import secrets
import shutil
import threading
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, abort, render_template, request, send_from_directory, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SelectField, SubmitField
from wtforms.validators import NumberRange

from model_assets import ensure_model_assets
from utils.models import Decoder, VGGEncoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
EXAMPLE_DIR = STATIC_DIR / "examples"
VGG_PATH = BASE_DIR / "vgg_normalised.pth"
# decoder_16.pth is the final saved decoder and matches checkpoint_latest.pth.
DECODER_PATH = BASE_DIR / "experiment" / "balanced_v2" / "decoder_final.pth"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
UPLOAD_REF_PATTERN = re.compile(r"^upload_[0-9a-f]{32}_[A-Za-z0-9_.-]+$")
RESULT_FILE_PATTERN = re.compile(r"^result_[0-9a-f]{32}\.png$")

# Browser forms submit IDs only. Filesystem paths always come from these maps.
CONTENT_EXAMPLES = {
    "virat": ("virat.jpg", "Virat (content example)"),
}
STYLE_EXAMPLES = {
    "style_11": ("11.jpg", "Style example 1 (11.jpg)"),
    "style_1499": ("1499.jpg", "Style example 2 (1499.jpg)"),
    "style_picasso": (
        "picasso_seated_nude_hr.jpg",
        "Style example 3 (Picasso)",
    ),
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)
app.config["MAX_CONTENT_LENGTH"] = 22 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
Bootstrap(app)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_upload_ref = HiddenField()
    style_upload_ref = HiddenField()
    content_example = SelectField(
        "Built-in content example",
        choices=[("", "Choose a built-in example…")]
        + [(key, value[1]) for key, value in CONTENT_EXAMPLES.items()],
    )
    style_example = SelectField(
        "Built-in style example",
        choices=[("", "Choose a built-in example…")]
        + [(key, value[1]) for key, value in STYLE_EXAMPLES.items()],
    )
    alpha = FloatField(
        "Alpha", default=0.65, validators=[NumberRange(min=0.0, max=1.0)]
    )
    submit = SubmitField("Transfer Style")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"PRISMORA AI inference device: {device}")

encoder = None
decoder = None
_model_load_lock = threading.Lock()


def load_models():
    """Load the checksum-verified encoder and final trained decoder once per worker."""
    global encoder, decoder

    if encoder is not None and decoder is not None:
        return encoder, decoder

    with _model_load_lock:
        if encoder is not None and decoder is not None:
            return encoder, decoder

        ensure_model_assets()
        loaded_encoder = VGGEncoder(VGG_PATH).to(device).eval()
        loaded_decoder = Decoder()
        decoder_state = torch.load(
            DECODER_PATH, map_location="cpu", weights_only=True
        )
        loaded_decoder.load_state_dict(decoder_state)
        loaded_decoder = loaded_decoder.to(device).eval()
        encoder, decoder = loaded_encoder, loaded_decoder

    return encoder, decoder


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
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
    encoder_model, decoder_model = load_models()

    # Keep the trained inference resize policy (short side = 768 / 512). for local host 
    # when running on render use 256 /256 for free tier
    content_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    style_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)
    content_tensor, original_h, original_w = pad_to_multiple_of_8(content_tensor)
    style_tensor, _, _ = pad_to_multiple_of_8(style_tensor)

    with torch.inference_mode():
        content_features = encoder_model(content_tensor, is_test=True)
        style_features = encoder_model(style_tensor, is_test=True)
        target_features = adaptive_instance_normalization(
            content_features, style_features
        )
        # AdaIN strength: 0 = content features, 1 = full style features.
        target_features = (
            alpha * target_features + (1.0 - alpha) * content_features
        )
        result = decoder_model(target_features)
        result = result[..., :original_h, :original_w]
        return result.clamp(0.0, 1.0)


def save_image(image_tensor, path):
    image = image_tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    transforms.ToPILImage()(image).save(path)


def example_options(examples):
    return [
        {
            "id": example_id,
            "label": label,
            "url": url_for("static", filename=f"examples/{filename}"),
        }
        for example_id, (filename, label) in examples.items()
    ]


def _display_name_for_upload(filename):
    # Stored upload names are upload_<uuid>_<secure original filename>.
    return filename.split("_", 2)[-1]


def _store_upload(file_storage):
    original_name = file_storage.filename or ""
    if not allowed_file(original_name):
        raise ValueError("Choose a PNG, JPG, or JPEG image.")

    stream = file_storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size > MAX_IMAGE_BYTES:
        raise ValueError("Each image must be 10 MB or smaller.")

    try:
        with Image.open(stream) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file is not a valid image.") from exc
    finally:
        stream.seek(0)

    safe_name = secure_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem[:100] or "image"
    stored_name = f"upload_{uuid.uuid4().hex}_{stem}{suffix}"
    destination = UPLOAD_DIR / stored_name

    # Exclusive creation prevents even a name collision from overwriting a file.
    with destination.open("xb") as output:
        shutil.copyfileobj(stream, output)

    return {
        "path": destination,
        "ref": stored_name,
        "example_id": "",
        "label": secure_filename(original_name) or "Uploaded image",
        "url": url_for("static", filename=f"uploads/{stored_name}"),
        "is_example": False,
    }


def _existing_upload(ref):
    if not UPLOAD_REF_PATTERN.fullmatch(ref or ""):
        raise ValueError("The selected upload is no longer available. Please select it again.")

    upload_root = UPLOAD_DIR.resolve()
    path = (UPLOAD_DIR / ref).resolve()
    if path.parent != upload_root or not path.is_file():
        raise ValueError("The selected upload is no longer available. Please select it again.")

    return {
        "path": path,
        "ref": ref,
        "example_id": "",
        "label": _display_name_for_upload(ref),
        "url": url_for("static", filename=f"uploads/{ref}"),
        "is_example": False,
    }


def sanitize_form_upload_refs(form):
    """Never echo an unvalidated browser-supplied upload token into the page."""
    for field in (form.content_upload_ref, form.style_upload_ref):
        if not field.data:
            continue
        try:
            _existing_upload(field.data)
        except ValueError:
            field.data = ""


def resolve_image_selection(file_storage, example_id, upload_ref, examples):
    """Resolve an upload ref or allowlisted example ID to an image and safe URL."""
    if example_id:
        example = examples.get(example_id)
        if example is None:
            raise ValueError("Choose a valid built-in example.")
        filename, label = example
        path = EXAMPLE_DIR / filename
        if not path.is_file():
            raise ValueError("A built-in example image is unavailable.")
        return {
            "path": path,
            "ref": "",
            "example_id": example_id,
            "label": label,
            "url": url_for("static", filename=f"examples/{filename}"),
            "is_example": True,
        }

    if file_storage and file_storage.filename:
        return _store_upload(file_storage)

    if upload_ref:
        return _existing_upload(upload_ref)
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image_url = None
    result_download_url = None
    content_image = None
    style_image = None
    error = None

    if request.method == "POST":
        if form.validate_on_submit():
            try:
                content_image = resolve_image_selection(
                    form.content.data,
                    form.content_example.data,
                    form.content_upload_ref.data,
                    CONTENT_EXAMPLES,
                )
                style_image = resolve_image_selection(
                    form.style.data,
                    form.style_example.data,
                    form.style_upload_ref.data,
                    STYLE_EXAMPLES,
                )

                form.content_upload_ref.data = (
                    content_image["ref"] if content_image and not content_image["is_example"] else ""
                )
                form.style_upload_ref.data = (
                    style_image["ref"] if style_image and not style_image["is_example"] else ""
                )
                form.content_example.data = (
                    content_image["example_id"] if content_image else ""
                )
                form.style_example.data = style_image["example_id"] if style_image else ""

                if not content_image or not style_image:
                    error = "Select or upload both a content image and a style reference."
                else:
                    with Image.open(content_image["path"]) as image:
                        content_pil = ImageOps.exif_transpose(image).convert("RGB")
                    with Image.open(style_image["path"]) as image:
                        style_pil = ImageOps.exif_transpose(image).convert("RGB")

                    alpha = form.alpha.data if form.alpha.data is not None else 0.65
                    result_tensor = style_transfer(content_pil, style_pil, alpha)
                    result_filename = f"result_{uuid.uuid4().hex}.png"
                    save_image(result_tensor, UPLOAD_DIR / result_filename)
                    result_image_url = url_for(
                        "static", filename=f"uploads/{result_filename}"
                    )
                    result_download_url = url_for(
                        "download_result", filename=result_filename
                    )
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Style transfer failed")
                error = "Style transfer failed. Check the Flask log for details."
        else:
            error = "Please correct the form values and try again."

    sanitize_form_upload_refs(form)
    return render_template(
        "index.html",
        form=form,
        result_image_url=result_image_url,
        result_download_url=result_download_url,
        content_image=content_image,
        style_image=style_image,
        content_examples=example_options(CONTENT_EXAMPLES),
        style_examples=example_options(STYLE_EXAMPLES),
        alpha_value=form.alpha.data if form.alpha.data is not None else 0.65,
        error=error,
    )


@app.get("/health")
def health():
    """Lightweight Render health check; it never loads model weights."""
    return {"status": "ok", "service": "prismora-ai"}, 200


@app.get("/download/<filename>")
def download_result(filename):
    """Download only a generated result file created by this application."""
    if not RESULT_FILE_PATTERN.fullmatch(filename or ""):
        abort(404)
    if not (UPLOAD_DIR / filename).is_file():
        abort(404)
    return send_from_directory(
        UPLOAD_DIR,
        filename,
        as_attachment=True,
        download_name="prismora-style-transfer.png",
        mimetype="image/png",
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
