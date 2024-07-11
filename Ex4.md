## Evaluation Experiment 4: Example

python -m gen_patch_neural_rendering.main \
--workdir=/home/woody/iwi5/iwi5143h/run_ex31com \
--is_train=False \
--ml_config=/home/woody/iwi5/iwi5143h/ex4_neu/gen_patch_neural_rendering/configs/defaults.py \
--ml_config.dataset.xray_base_dir=/home/woody/iwi5/iwi5143h/datasets/train_ex31/scenes \
--ml_config.dataset.eval_xray_dir=/home/woody/iwi5/iwi5143h/datasets/eval_ex32/scenes \
--ml_config.dataset.XML_dir="/home/woody/iwi5/iwi5143h/TRAINING/SimpleShape.xml" \
--ml_config.dataset.eval_dataset=xray \
--ml_config.dataset.name=ff_epipolar \
--ml_config.dataset.render_style="xray" \
--ml_config.dataset.llffhold=2 \
--ml_config.dataset.num_interpolation_views=20 \
--ml_config.dataset.angle_steps=5 \
--ml_config.dataset.batch_size=16 \
--ml_config.eval.chunk=1024 \
--ml_config.dataset.normalize=True \
--ml_config.model.num_rgb_channels=3

