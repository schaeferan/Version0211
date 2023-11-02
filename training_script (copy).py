
python3 -m gen_patch_neural_rendering.main \
  --workdir=/tmp/train_run \
  --is_train=True \
  --ml_config=gen_patch_neural_rendering/configs/defaults.py \
  --ml_config.dataset.ff_base_dir=/home/ux28udoc/Desktop/scenes \
  --ml_config.dataset.name=ff_epipolar  \
  --ml_config.dataset.batch_size=4096  \
  --ml_config.lightfield.max_deg_point=4 \
  --ml_config.train.lr_init=3.0e-4 \
  --ml_config.train.warmup_steps=5000 \
  --ml_config.train.render_every_steps=500000 \
  --ml_config.dataset.normalize=True \
  --ml_config.model.init_final_precision=HIGHEST
