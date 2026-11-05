# PRISMORA AI — NEURAL STYLE STUDIO

**Transforming pixels into art.**

PRISMORA AI is a Flask application for Adaptive Instance Normalization (AdaIN) neural style transfer. A pretrained VGG encoder extracts image features, AdaIN aligns content features to a reference style, and the project's trained decoder reconstructs the stylized image. The existing model architecture and trained weights are preserved.

## Features

- Upload separate content and style images or select allowlisted built-in examples.
- Adjust style strength from subtle to full transfer.
- Preview and download generated PNG results.
- Responsive showcase cards that preserve the full aspect ratio of each original image.
- Lightweight `/health` endpoint for service health checks.

## Local setup

Python 3.11 is the supported runtime. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Set a private `FLASK_SECRET_KEY` in `.env` for a persistent local session key. The checked-in `.env.example` contains no credentials. The existing local checkpoints can be validated with:

```bash
python -m scripts.fetch_model_assets
```

Start the development server:

```bash
python app.py
```

Open `http://127.0.0.1:5000`. Set `FLASK_DEBUG=1` only for local development. On the first transfer, the app loads the model into memory; startup and `/health` do not need to load the weights.

## Model artifacts

Inference requires exactly these two trained files. They are intentionally excluded from Git:

| Local path | Private S3 object key below `MODEL_ASSET_PREFIX` | Required SHA-256 |
| --- | --- | --- |
| `vgg_normalised.pth` | `vgg_normalised.pth` | `804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a` |
| `experiment/balanced_v2/decoder_16.pth` | `balanced_v2/decoder_16.pth` | `859715b98b510b975239cae92cae88aa1247e82cbceddd7484f1f55a764c0954` |

The VGG file is the encoder weights; `decoder_16.pth` is the final trained decoder checkpoint. The S3 bucket and credentials must be supplied by the project owner; no public artifact URL or secret is embedded in this repository. Upload the corresponding files to a private bucket using the keys above and set these Render environment variables:

- `MODEL_ASSET_BUCKET` — private bucket name.
- `MODEL_ASSET_PREFIX` — object prefix, default `models`.
- `AWS_DEFAULT_REGION` — bucket region.
- AWS credentials via the hosting environment's secret configuration (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, when applicable, `AWS_SESSION_TOKEN`). Grant only read access to the two required objects.
- `FLASK_SECRET_KEY` — a private, random session secret.

At service start, the `Procfile` downloads only missing files through the AWS SDK, verifies SHA-256 before installing them, and refuses to replace an existing file. Without either verified local artifacts or S3 configuration, inference cannot run; the page and health route still initialize.

## Quality checks

Install the development requirements and run:

```bash
python -m compileall -q app.py train.py utils model_assets.py scripts tests
ruff check .
pytest
```

The tests use the CPU-compatible dependencies and mock the inference call for ordinary route tests. They check the seven required example images, upload validation, allowlisted examples, generated result delivery, the download route, and model artifact checksum behavior. No GPU or checkpoint is required for CI.

## GitHub Actions and Render deployment

`.github/workflows/ci.yml` runs on pull requests targeting `main` and pushes to `main`. It installs Python 3.11 dependencies, checks syntax, runs Ruff, and runs pytest. Configure GitHub branch protection for `main` to require the **Python 3.11 / lint and tests** status check before merging.

Use the existing Render service's native GitHub connection; this repository deliberately does not add a second deployment workflow or a second service definition. In the Render dashboard, select the repository's `main` branch, set auto-deploy to **After CI Checks Pass**, use `pip install -r requirements.txt` as the build command, and set the health-check path to `/health`. Keep the start command aligned with the checked-in `Procfile`. Add the model-store and Flask secrets as Render environment variables. The GitHub and Render dashboard settings must be completed by an account administrator; they are not stored in this repository.

## Example assets

The website examples are served from `static/examples/` and are tracked by Git:

- `virat.jpg` → `11.jpg` → `stylized_virat_11.jpg`
- `virat.jpg` → `1499.jpg` → `stylized_virat_1499.jpg`
- `virat.jpg` → `picasso_seated_nude_hr.jpg` → `stylized_virat_hr.jpg`

Uploads and generated results stay under `static/uploads/`, which is ignored by Git except for its `.gitkeep` marker.
