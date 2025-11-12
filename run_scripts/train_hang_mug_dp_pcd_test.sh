# test script small epoch for hang_mug_dp with point cloud data and no augmentation
# fit with hang_mug_dp_pcd_none_test.yaml
# copy mainly from train_hang_mug_dp_pcd.sh

export variable HYDRA_FULL_ERROR=1
python diffusion_policy_code/general_dp/train.py \
        --config-dir=diffusion_policy_code/general_dp/config \
        --config-name=hang_mug_dp_pcd_none_test.yaml \
        training.seed=42 \
        training.device=cuda:0 \
        hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'