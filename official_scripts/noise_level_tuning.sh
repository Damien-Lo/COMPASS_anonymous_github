#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --output=out_tuning.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

# Full noise-sweep hyperparam_tuning (label-blind target-set-vs-reference-set divergence,
# all 9 COMPASS metrics) across a 33-point Gaussian-noise std grid. Dummy descriptions are
# used since img_metrics.parts=["img"] only scores the image-token slice of the logits,
# which is provably independent of description content under causal attention (images
# precede text in the conversation). augmentation_accumilator is forced to ['none'] on
# every metric since the 'avg'/'max' cross-setting accumulator silently collapses samples
# at batch_size>1 -- 'none' is the only output this project's analysis uses. Requires a
# uniform-resolution dataset (same vision-token count per image) for batch_size>1 to be safe.

# Load environment
source ~/.bashrc
conda activate

member_dataset=''        # .parquet, member_target_dataset.parquet from build_dataset.sh
nonmember_dataset=''     # .parquet, non_member_target_dataset.parquet from build_dataset.sh
reference_dataset=''     # .parquet, reference_dataset.parquet from build_dataset.sh
pre_gen_descriptions=''  # .json, {"sentences": [...]} of "N/A" placeholders, one per row of
                          # (target_set_size + len(reference_dataset)) -- 600 rows by default
model=''                 # target_model config name: llava-v1.5-7b / minigpt-4 / med_hulu
batch_size=1              # confirm safe value via a smoke test; requires a uniform-resolution
                           # dataset (same vision-token count per image) to be safe above 1
out_dir=''                # writes out_dir/gn_set{0..32}/ (one dir per noise-std sweep point)

STD_SETS=(
  "[0.000]" "[0.005]" "[0.0078]"
  "[0.012]" "[0.019]" "[0.03]"
  "[0.046]" "[0.072]" "[0.11]"
  "[0.18]" "[0.28]" "[0.43]"
  "[0.67]" "[1.1]" "[1.6]"
  "[2.6]" "[4.0]" "[6.2]"
  "[9.8]" "[15]" "[24]"
  "[37]" "[58]" "[91]"
  "[140]" "[220]" "[340]"
  "[540]" "[840]" "[1300]"
  "[2100]" "[3200]" "[5000]"
)

export PYTHONPATH=$PYTHONPATH:${python_path}

for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
    python mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Noise-level tuning, std set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=hyperparam_tuning \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model=${model} \
        inference.batch_size=${batch_size} \
        \
        data.save_datasets=true \
        data.target_set_size=300 \
        data.n_nm_ratio=0.5 \
        data.member_dataset=${member_dataset} \
        data.nonmember_dataset=${nonmember_dataset} \
        data.reference_datasets_list=${reference_dataset} \
        data.reference_set_sample_distribution=[] \
        data.pre_gen_descriptions=${pre_gen_descriptions} \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"] \
        img_metrics.max_k_no_norn_kl_div.augmentation_accumilator=['none'] \
        img_metrics.max_k_no_norn_kl_div.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_05_kl_div.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_05_kl_div.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_1_kl_div.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_1_kl_div.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_2_kl_div.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_2_kl_div.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_inf_kl_div.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_inf_kl_div.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_025.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_025.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_05.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_05.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_2.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_2.augmentation_setting_version_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_4.augmentation_accumilator=['none'] \
        img_metrics.max_k_renyi_divergence_4.augmentation_setting_version_accumilator=['none'] \
        img_metrics.get_raw_meta_metrics=[] \
        img_metrics.get_proc_meta_metrics=[] \
        \
        img_metrics.get_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=0 \
        \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0]' \
        data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
done
