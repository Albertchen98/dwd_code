"""DwD training defaults: DINO input x4, PCA-32 tail drop, temporal convolution."""
from copy import deepcopy
from hydra.core.config_store import ConfigStore
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.dinov3_encoder import DinoV3Config
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.exp_large_scale import (
    vid2vid_2B_control_720p_t24_dino_prep_control_layer4_cr1_embedding_rectified_flow_train_keyframe as base,
)

for mode in ("offline", "online"):
    cfg = deepcopy(base)
    cfg.defaults.insert(1, {"override /data_val": "nuplan_video"})
    cfg.job.name = f"dwd_x4_taildrop_temporal_{mode}"
    cfg.job.project = "dwd"
    cfg.model.config.feature_mode = mode
    cfg.model.config.sample_dino_key_frame = False
    cfg.model.config.net.update(dict(
        sample_dino_key_frame=False, use_dino_pca=True, dino_ctrl_channels=32,
        dino_upfactor=4, dino_downsample_method="conv",
    ))
    cfg.model.config.tokenizer.compile_encode = False
    cfg.model.config.dinov3_encoder = deepcopy(DinoV3Config)
    cfg.model.config.dinov3_encoder.update(dict(
        height=2816, width=5120, use_dino_pca=True, use_processor=False,
        use_random_channel=True, use_l2_norm=False, out_layers=[23],
        forward_chunk_size=1,
        pca_mean_path="checkpoints/pca/patch_pca_mean_32.npy",
        pca_comp_path="checkpoints/pca/patch_pca_components_32.npy",
    ))
    for loader in (cfg.dataloader_train, cfg.dataloader_val):
        loader.dataset.update(dict(
            hint_keys="dino", use_anyup_processed=mode == "offline",
            anyup_pca_name="dinov3_x4_pca32", anyup_factor=4, pca_channel=32,
            strict_feature_shape=True, use_random_channel=mode == "offline",
        ))
    cfg.dataloader_val.dataset.use_random_channel = False
    cfg.trainer.max_iter = 10000
    cfg.checkpoint.save_iter = 1000
    ConfigStore.instance().store(group="experiment", package="_global_", name=f"dwd_{mode}", node=cfg)
