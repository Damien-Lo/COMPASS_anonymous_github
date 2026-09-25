#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=out_build_dataset.log
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G

# Builds the target/reference dataset parquet files (member + non-member target set, plus
# a separate reference set used for noise-level tuning) from raw member/non-member/
# reference JSON source lists. Model-agnostic -- swap model/member_dataset/nonmember_dataset/
# reference_dataset below for any target model or dataset.

# Load environment
source ~/.bashrc
conda activate

member_dataset=''      # .json, flat list of {"image_paths": [...], ...} entries (member-source images)
nonmember_dataset=''   # .json, same schema as member_dataset (non-member-source images)
reference_dataset=''   # .json, same schema as member_dataset (reference-set-source images, used only for tuning)
model=''               # target_model config name: llava-v1.5-7b / minigpt-4 / med_hulu
out_dir=''             # writes out_dir/datasets/{target,member_target,non_member_target,reference}_dataset.parquet

export PYTHONPATH=$PYTHONPATH:${python_path}

python mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Build target+reference dataset'" \
    job_meta_params.job_type=build_dataset \
    \
    path.output_dir=${out_dir} \
    \
    target_model=${model} \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.member_dataset=${member_dataset} \
    data.nonmember_dataset=${nonmember_dataset} \
    data.reference_datasets_list=${reference_dataset} \
    data.reference_set_sample_distribution=[] \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=[] \
    img_metrics.get_raw_meta_metrics=['losses'] \
    img_metrics.get_proc_meta_metrics=[]
