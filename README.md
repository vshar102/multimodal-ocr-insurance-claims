# Multi-Modal OCR for Insurance Claim ID Classification

`multimodal-ocr-insurance-claims`  ·  Portfolio project  ·  MIT License

## Overview

A multi-modal OCR / document-classification pipeline that identifies scanned insurance-claim ID documents as primary or secondary IDs, supporting digitization of historical paper claims.

## Problem Statement

Digitizing historical insurance claim documents requires not just OCR text extraction but correctly classifying what type of ID document was scanned, to route it correctly downstream.

## Approach

Combines OCR-derived features with a PyTorch classification model, using a custom `project_utils.py` module (rather than only inline notebook code) for reusable data-loading/preprocessing logic, and a saved trained-model artifact.

## Tech Stack

Python, PyTorch, Pickle, custom project_utils module

## Results

See the notebook's evaluation cells for classification accuracy on the primary-vs-secondary ID task.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib pillow jupyter
```

## Usage

```bash
jupyter notebook notebook.ipynb
```
`project_utils.py` is imported directly by the notebook — keep it in the same directory.

## Project Structure

```
.
├── notebook.ipynb
├── project_utils.py       # custom data-loading / preprocessing module
├── digitizing_team.png
└── README.md
```

## Security Considerations

No API keys or credentials are committed to this repository. Where the project
calls an external API, copy `.env.example` to `.env` and supply your own key —
`.env` is git-ignored.

## Troubleshooting

- If a notebook cell fails on a missing package, install the pinned versions
  in `requirements.txt` (or the imports listed in Tech Stack above) rather than
  the latest release, since some ML libraries change APIs between minor versions.
- Large datasets and model checkpoints are intentionally excluded from this
  repository (see `.gitignore`) to keep it lightweight — see Setup for how to
  obtain or regenerate them.

## Roadmap

- [ ] Add unit tests for project_utils.py functions
- [ ] Replace the pickled dataset with a documented, reproducible data-loading step
- [ ] Add a confusion matrix and per-class precision/recall

## License

Licensed under the MIT License — see `LICENSE`.

---
*This project originated as a guided DataCamp practical exercise and was
independently re-packaged, documented, and prepared for portfolio presentation.
The premise/business framing in the notebook is illustrative, not a real client engagement.*
