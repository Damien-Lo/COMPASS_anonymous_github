#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=out_baselines.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

# Baseline MIA metrics (Aug-KL, Min-k%, Max-Renyi entropy variants) -- a single job, no
# Gaussian noise. Real descriptions are generated (data.pre_gen_descriptions left empty)
# since these baselines aren't provably independent of description content the way the
# img-only noise-perturbation divergence metrics are.

# Load environment
source ~/.bashrc
conda activate

target_dataset=''  # .parquet, target_dataset.parquet from build_dataset.sh (combined
                    # member+non-member rows, real 'label' column)
model=''            # target_model config name: llava-v1.5-7b / minigpt-4 / med_hulu
batch_size=1        
out_dir=''           # single output directory (no noise sweep -- one job, no gn_set subdirs)

export PYTHONPATH=$PYTHONPATH:${python_path}

python mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines for members/non-members'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir} \
    \
    target_model=${model} \
    inference.batch_size=${batch_size} \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions="" \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=['aug_kl','max_k_renyi_1_entro','max_k_renyi_05_entro','mink'] \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=0 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.RandomAffine.use=true \
    data.augmentations.ColorJitter.use=true
