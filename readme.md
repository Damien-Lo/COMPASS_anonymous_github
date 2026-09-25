# COMPASS: A Noise-Perturbation Membership Inference Attack on Vision–Language Models

COMPASS is a **membership inference attack (MIA)** for vision–language models (VLMs), built around a family of KL/Rényi-divergence metrics (**COMPASS-KL**, **COMPASS-R**) computed on model logits under small Gaussian-noise image perturbations. A held-out **reference set** is used to pick a per-metric noise level and score direction in a **label-blind** way (no access to real member/non-member labels), which is then evaluated against the real target set to report AUC and TPR@low-FPR.

This repository contains the code to:
- Build member/non-member/reference datasets for a target model.
- Tune the noise level and score direction for each COMPASS metric, label-blind.
- Evaluate the tuned MIA against real member/non-member labels (AUC, TPR@low-FPR).
- Run classical MIA baselines (Min-k%, Aug-KL, Max-Rényi entropy variants) for comparison.

---

## Supported Target Models

Set via `target_model=<name>` on any of the scripts below (`config/target_model/*.yaml`):

- **`llava-v1.5-7b`** — LLaVA-v1.5-7B.
- **`minigpt-4`** — MiniGPT-4 (Llama-2 backbone by default, `config/target_model/model_config/minigpt4_llama2.yaml`; MiniGPT-v2 variant at `model_config/minigpt_v2.yaml`).
- **`med_hulu`** — Hulu-Med (medical VLM), defaults to the public `ZJU-AI4H/Hulu-Med-32B` checkpoint on the HuggingFace Hub; override `target_model.model_path=/path/to/local/checkpoint` to use a local copy instead.

For gated HuggingFace checkpoints (e.g. `meta-llama/Llama-2-7b-chat-hf` used by MiniGPT-4), make sure you've accepted the model's license on the Hub and are logged in (`huggingface-cli login`) before running.

---

## Setup

1. **Clone this repository** and create/activate a Python environment with the packages in `requirements.txt` (`pip install -r requirements.txt`).
2. **Set `cache_dir`** in `config/path/path.yaml` to a directory HuggingFace/torch caches should use.
3. **Point at your model checkpoints**, if not using a model's default HuggingFace Hub path — see [Supported Target Models](#supported-target-models) above.

---

## Pipeline

All four scripts live in `official_scripts/`. Each is a plain SLURM (`sbatch`) script with a block of blank fill-in variables at the top (paths, model name, batch size, output directory) and no other required edits. Run them from inside the `mia/` directory (they call `python mia.py` directly, resolving Hydra's config relative to the working directory).

### 1. `build_dataset.sh` — build datasets

Builds `target_dataset.parquet` (combined member+non-member set with real labels), `member_target_dataset.parquet` / `non_member_target_dataset.parquet` (the same set split apart, used by tuning), and `reference_dataset.parquet` from three raw JSON source lists (`member_dataset`, `nonmember_dataset`, `reference_dataset` — see [Raw Dataset JSON Format](#raw-dataset-json-format) below for the exact schema). Model-agnostic.

### 2. `noise_level_tuning.sh` — label-blind noise/direction tuning

Runs a 33-point Gaussian-noise std sweep (`job_type=hyperparam_tuning`) over the **member+non-member target set concatenated with the reference set**, computing all 9 COMPASS metrics (5 COMPASS-KL variants + 4 COMPASS-R variants) at each noise level. For each metric, the noise level and score direction (flip or not) are chosen as whichever std maximizes `|mean(target scores) − mean(reference scores)|` — entirely from the tuning-only `tune_label` column, never from real membership labels.

### 3. `mia_classification.sh` — evaluation

Runs the same 33-point sweep (`job_type=evaluation`) over the real `target_dataset.parquet`, scoring against true member/non-member labels to report AUC and TPR@low-FPR at every noise level. Combine with the std/flip chosen by tuning to get the final label-blind MIA result at that specific noise level; sweeping all 33 points also lets you inspect the full AUC-vs-noise curve.

### 4. `baselines.sh` — classical baselines

A single job (no noise sweep, `job_type=evaluation`) computing standard MIA baselines (Aug-KL, Min-k%, Max-Rényi-entropy variants) on the real target set, for comparison against the COMPASS metrics above. Unlike tuning/evaluation, this generates real model descriptions rather than using placeholder text, since these baseline metrics (unlike the image-token-only COMPASS metrics) are not provably independent of description content.

---

## Metrics Reference

Every value below is a valid entry in `img_metrics.metrics_to_use`; each has its own Hydra config (settings like the top-k ratio grid) under `config/img_metrics/metrics/<name>.yaml`, wired up as Hydra defaults in `config/img_metrics/img_metrics.yaml`.

**COMPASS-KL** (top-k, noise-perturbed KL divergence) — used by `noise_level_tuning.sh` / `mia_classification.sh`:
- `max_k_no_norn_kl_div`, `max_k_renyi_05_kl_div`, `max_k_renyi_1_kl_div`, `max_k_renyi_2_kl_div`, `max_k_renyi_inf_kl_div`

**COMPASS-R** (top-k, noise-perturbed Rényi divergence) — used by `noise_level_tuning.sh` / `mia_classification.sh`:
- `max_k_renyi_divergence_025`, `max_k_renyi_divergence_05`, `max_k_renyi_divergence_2`, `max_k_renyi_divergence_4`

**Classical baselines** (no noise perturbation) — used by `baselines.sh`:
- `aug_kld` (Aug-KL), `mink` (Min-k%), `cross_entropy_mink`
- `max_prob_gap`
- `max_k_renyi_1_entro`, `max_k_renyi_2_entro`, `max_k_renyi_05_entro` (Max-Rényi-entropy)
- `min_k_renyi_1_entro`, `min_k_renyi_2_entro`, `min_k_renyi_05_entro` (Min-Rényi-entropy)
- `mod_renyi_1_entro`, `mod_renyi_2_entro`, `mod_renyi_05_entro` (Modified-Rényi-entropy)
- `renyi_1`, `renyi_2`, `renyi_05`, `renyi_05_full`

`baselines.sh` as provided only enables `aug_kld`, `mink`, `max_k_renyi_1_entro`, `max_k_renyi_05_entro` — the rest are available but not wired into any of the four scripts by default; add them to a script's `img_metrics.metrics_to_use` list to enable.

---

## Raw Dataset JSON Format

`build_dataset.sh`'s `member_dataset` / `nonmember_dataset` / `reference_dataset` arguments each take a path to a JSON file containing a **flat list of objects**, one per image. You do **not** need to include `label` or `tune_label` — those are assigned automatically based on which file (member vs. non-member vs. reference) an entry came from.

The one field every entry must have depends on the target model:

- **`llava-v1.5-7b` / `minigpt-4`** — a single `"image"` key, a string path to the image file:
  ```json
  [
    {"image": "/path/to/image_0001.jpg"},
    {"image": "/path/to/image_0002.jpg"}
  ]
  ```

- **`med_hulu`** — an `"image_paths"` key, a **list** of one or more string image paths per entry (supports multi-image samples):
  ```json
  [
    {"image_paths": ["/path/to/image_0001.jpg"]},
    {"image_paths": ["/path/to/image_0002.jpg"]}
  ]
  ```

Extra keys (e.g. `id`, `modality`, `body_part`) are preserved but ignored by the pipeline — include whatever metadata is useful for your own bookkeeping.

---

## Additional Run Parameters

Passed as Hydra overrides on the `python mia.py ...` command line (see `official_scripts/*.sh` for the full set already wired up):

- `img_metrics.metrics_to_use` — which metrics to compute.
- `img_metrics.get_raw_meta_metrics` / `img_metrics.get_proc_meta_metrics` — which raw/processed intermediate values (e.g. per-token divergences) to save alongside the final scores.
- `img_metrics.get_meta_examples` / `img_metrics.get_token_labels` — how many samples' intermediate values to save.
- `img_metrics.get_raw_images` — if set to `x`, saves the perturbed images for the first `x` member and first `x` non-member samples.
- `data.augmentations.GaussianNoise.std` — the noise std grid for this run (a list, one job per sweep point in the provided scripts).
