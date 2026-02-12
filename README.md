# Dice Detection

Faster R-CNN with ResNet50-FPN for detecting Yahtzee dice while studying long-tailed data, augmentation, and GAN-based synthesis.

## Repository Layout

```
Dice-Detection/
├── Annotations/                 # COCO-format label files used across experiments
├── checkpoints/                 # Saved .pth models referenced in the paper
├── notebook/                    # Colab-ready experiment notebooks (training, ablations, GAN)
├── scripts/                     # Small CLI utilities (dataset download, checkpoint eval)
└── src/                         # Python modules (datasets, models, losses, augmentation, metrics)
```

## Getting Started

1. Clone the project
   ```bash
   git clone https://github.com/Adr44mo/Dice-Detection.git
   cd Dice-Detection
   ```
2. (Optional) Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

`requirements.txt` keeps the dependency list minimal (PyTorch, torchvision, roboflow, matplotlib, seaborn, tqdm, numpy, Pillow) so the same environment works for both notebooks and scripts.

## Downloading the Dataset (scripts/download_roboflow_dataset.py)

Use the download script when you prefer a reproducible CLI alternative to the notebook cells. Set your Roboflow API key and run:

```bash
export ROBOFLOW_API_KEY="your_key"
python scripts/download_roboflow_dataset.py
```

Defaults:

- Workspace: `workspace-spezm`
- Project: `dice-0sexk`
- Version: `2`
- Format: `coco`
- Destination: `./dice-2`

The script refuses to overwrite a non-empty target directory, so either delete `dice-2/` or change `OUTPUT_DIR` before redownloading. Adjust the configuration block at the top of the script if you need a different Roboflow project, export format, or version.

## Training and Experimentation (Notebooks)

- `notebook/Dataset_Creation.ipynb`: builds balanced vs Zipfian splits and exports COCO annotations.
- `notebook/augmentation_comparison.ipynb`: main training workflow with toggles for sampling methods, mosaic, random aug, focal loss, etc.
- `notebook/GAN.ipynb`: trains the conditional DCGAN and prepares synthetic dice crops/background composites.
- `notebook/6_gan_dataset_training.ipynb`: fine-tunes Faster R-CNN on GAN-augmented datasets and logs results.

Each notebook exposes simple boolean flags or probability sliders so you can rerun the ablations published in `results/`. All notebooks expect the dataset folders produced by either Roboflow or the download script above.

## Evaluating Checkpoints (scripts/eval_checkpoint_map.py)

`scripts/eval_checkpoint_map.py` computes mAP@0.5 on any COCO-formatted split without touching a notebook:

1. Edit the configuration block at the top of the script:
   - `CHECKPOINT_PATH`: path to the `.pth` file you want to score (e.g. `checkpoints/DA_MA_RA.pth`).
   - `DATASET_ROOT`: directory containing the images for the split (`dice-2/test` by default).
   - `ANNOTATION_FILE`: filename of the COCO JSON inside that folder.
   - `CUSTOM_ANNOTATION_PATH`: optional JSON copied into the dataset folder before evaluation (handy for balanced test sets).
2. Run the script:
   ```bash
   python scripts/eval_checkpoint_map.py
   ```

The script automatically:

- Copies the requested annotation JSON into the dataset folder when `USE_CUSTOM_ANNOTATION` is `True`.
- Infers how many classes the checkpoint was trained on by reading `roi_heads.box_predictor.cls_score.weight`. If the dataset metadata disagrees, the checkpoint value wins and a message is printed.
- Builds the ResNet50-FPN model, loads the checkpoint, and reports overall mAP plus per-class AP using the pure-PyTorch evaluator in `src/metrics.py`.

Set `OUTPUT_JSON` if you want the metrics persisted to disk.

## Source Modules

- `src/dataset.py`: COCO dataset loader with torchvision-style targets and custom `collate_fn`.
- `src/model.py`: Faster R-CNN factory (ResNet50-FPN) plus checkpoint save/load helpers and focal-loss RoI heads.
- `src/augmentation.py` and `src/aug/`: mosaic, copy-paste, annotation management, and difficulty-aware sampling utilities.
- `src/Loss_function.py`: implementation of focal loss used to reweight RoI heads.
- `src/training.py`: epoch loops, optimizer/scheduler helpers, and logging utilities for the notebooks.
- `src/metrics.py`: IoU, AP, and mAP calculation used by both notebooks and the evaluation script.



