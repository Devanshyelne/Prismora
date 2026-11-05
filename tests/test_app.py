from __future__ import annotations

from io import BytesIO
from hashlib import sha256
from html import unescape
import re

from PIL import Image
import pytest
import torch

import app as app_module


EXAMPLE_IMAGES = (
    "virat.jpg",
    "11.jpg",
    "1499.jpg",
    "picasso_seated_nude_hr.jpg",
    "stylized_virat_11.jpg",
    "stylized_virat_1499.jpg",
    "stylized_virat_hr.jpg",
)


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return app_module.app.test_client()


@pytest.fixture
def isolated_uploads(tmp_path, monkeypatch):
    static_root = tmp_path / "static"
    uploads = static_root / "uploads"
    uploads.mkdir(parents=True)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(app_module.app, "static_folder", str(static_root))
    return uploads


def image_file(color: tuple[int, int, int], name: str = "test.jpg"):
    stream = BytesIO()
    Image.new("RGB", (24, 16), color).save(stream, format="JPEG")
    stream.seek(0)
    return stream, name, "image/jpeg"


def test_app_initializes_without_loading_model_weights(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PRISMORA AI" in response.data
    assert app_module.encoder is None
    assert app_module.decoder is None


def test_health_endpoint_is_lightweight(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "service": "prismora-ai"}
    assert app_module.encoder is None


@pytest.mark.parametrize("filename", EXAMPLE_IMAGES)
def test_example_assets_are_served_and_are_valid_images(client, filename):
    response = client.get(f"/static/examples/{filename}")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    with Image.open(BytesIO(response.data)) as image:
        image.verify()


def test_homepage_has_final_branding_links_faq_and_image_fit_rules(client):
    response = client.get("/")
    html = unescape(response.get_data(as_text=True))
    assert '<title>PRISMORA AI — NEURAL STYLE STUDIO</title>' in html
    assert "NEURAL STYLE STUDIO" in html
    assert "Transforming pixels into art." in html
    assert "// LET&#39;S CONNECT" in html or "// LET'S CONNECT" in html
    assert "Transforming pixels." in html
    assert "Creating possibilities." in html
    assert "Adaptive Instance Normalization and neural style transfer." in html
    assert "Let's connect, collaborate and explore what's possible with AI." in html
    for link in (
        "https://www.linkedin.com/in/devanshyelne/",
        "https://github.com/Devanshyelne",
        "https://x.com/DevanshYel78731",
        "mailto:devanshyelne06@gmail.com",
    ):
        assert link in html
    footer = re.search(r"<footer[^>]*>(.*?)</footer>", html, flags=re.DOTALL)
    assert footer is not None
    assert re.sub(r"<[^>]+>", "", footer.group(1)).strip() == "Created by Devansh Yelne"
    assert "object-fit: contain" in html
    assert "object-fit: cover" not in html
    assert 'id="faqAccordion"' in html
    assert 'id="faqOne"' in html


def test_uploads_content_and_style_runs_transfer_and_downloads_output(
    client, isolated_uploads, monkeypatch
):
    calls = []
    example_path = app_module.EXAMPLE_DIR / "virat.jpg"
    example_hash_before = sha256(example_path.read_bytes()).hexdigest()

    def fake_transfer(content, style, alpha):
        calls.append((content.mode, style.mode, alpha))
        return torch.full((1, 3, 8, 8), 0.5)

    monkeypatch.setattr(app_module, "style_transfer", fake_transfer)
    content, content_name, content_type = image_file((220, 30, 20), "portrait.jpg")
    style, style_name, style_type = image_file((15, 40, 210), "painting.jpg")
    response = client.post(
        "/",
        data={
            "content": (content, content_name, content_type),
            "style": (style, style_name, style_type),
            "content_example": "",
            "style_example": "",
            "alpha": "0.35",
        },
        content_type="multipart/form-data",
    )

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Stylized Result" in html
    assert calls == [("RGB", "RGB", 0.35)]
    assert sorted(path.name.split("_", 1)[0] for path in isolated_uploads.iterdir()) == [
        "result",
        "upload",
        "upload",
    ]

    preview_names = re.findall(r"/static/uploads/(upload_[0-9a-f]{32}_[A-Za-z0-9_.-]+)", html)
    assert len(preview_names) == 2
    for preview_name in preview_names:
        preview = client.get(f"/static/uploads/{preview_name}")
        assert preview.status_code == 200
        assert preview.mimetype == "image/jpeg"
    assert sha256(example_path.read_bytes()).hexdigest() == example_hash_before

    result_name = re.search(r"/static/uploads/(result_[0-9a-f]{32}\.png)", html)
    assert result_name is not None
    generated = client.get(f"/static/uploads/{result_name.group(1)}")
    assert generated.status_code == 200
    assert generated.mimetype == "image/png"
    download = client.get(f"/download/{result_name.group(1)}")
    assert download.status_code == 200
    assert download.mimetype == "image/png"
    assert "attachment" in download.headers["Content-Disposition"]
    assert b"prismora-style-transfer.png" in download.headers["Content-Disposition"].encode()
    assert download.data == generated.data


def test_allowlisted_builtin_example_selection_works(client, isolated_uploads, monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module,
        "style_transfer",
        lambda content, style, alpha: calls.append((content.size, style.size, alpha))
        or torch.zeros((1, 3, 8, 8)),
    )
    response = client.post(
        "/",
        data={
            "content_example": "virat",
            "style_example": "style_11",
            "alpha": "0.8",
        },
    )
    assert response.status_code == 200
    assert "Stylized Result" in response.get_data(as_text=True)
    assert calls and calls[0][2] == pytest.approx(0.8)
    assert (isolated_uploads / next(p.name for p in isolated_uploads.iterdir())).is_file()


@pytest.mark.parametrize(
    ("filename", "payload", "expected"),
    [
        ("script.txt", b"not an image", "Choose a PNG, JPG, or JPEG image."),
        ("broken.jpg", b"not an image", "The selected file is not a valid image."),
    ],
)
def test_upload_validation_rejects_invalid_files(
    client, isolated_uploads, filename, payload, expected
):
    response = client.post(
        "/",
        data={
            "content": (BytesIO(payload), filename),
            "content_example": "",
            "style_example": "",
            "alpha": "0.65",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert expected in response.get_data(as_text=True)
    assert list(isolated_uploads.iterdir()) == []


def test_upload_size_limit_is_enforced(client, isolated_uploads, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_IMAGE_BYTES", 32)
    response = client.post(
        "/",
        data={
            "content": (BytesIO(b"x" * 33), "large.jpg"),
            "content_example": "",
            "style_example": "",
            "alpha": "0.65",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert "Each image must be 10 MB or smaller." in response.get_data(as_text=True)
    assert list(isolated_uploads.iterdir()) == []


def test_arbitrary_example_id_and_non_result_download_are_rejected(client):
    response = client.post(
        "/",
        data={"content_example": "../../etc/passwd", "style_example": "style_11"},
    )
    assert response.status_code == 200
    assert "Please correct the form values" in response.get_data(as_text=True)
    assert client.get("/download/virat.jpg").status_code == 404
    assert client.get("/download/result_00000000000000000000000000000000.png").status_code == 404
