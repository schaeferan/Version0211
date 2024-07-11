## Evaluation Experiment 5: Example

python -m gen_patch_neural_rendering.main \
--workdir=/home/woody/iwi5/iwi5143h/run_ex31com \
--is_train=False \
--ml_config=/home/woody/iwi5/iwi5143h/ex5/gen_patch_neural_rendering/configs/defaults.py \
--ml_config.dataset.xray_base_dir=/home/woody/iwi5/iwi5143h/datasets/train_ex4/scenes \
--ml_config.dataset.eval_xray_dir=/home/woody/iwi5/iwi5143h/datasets/eval_ex4/scenes \
--ml_config.dataset.XML_dir="/home/woody/iwi5/iwi5143h/ex5/DRR.xml" \
--ml_config.dataset.eval_dataset=xray \
--ml_config.dataset.name=ff_epipolar \
--ml_config.dataset.render_style="xray" \
--ml_config.dataset.llffhold=0 \
--ml_config.dataset.num_interpolation_views=10 \
--ml_config.dataset.angle_steps=2 \
--ml_config.dataset.batch_size=16 \
--ml_config.eval.chunk=1024 \
--ml_config.dataset.normalize=True \
--ml_config.model.num_rgb_channels=3

## Explanation

--workdir: direction for loading weights and saving results
--ml_config.dataset.xray_base_dir: direction for training set images (here no need for that)
--ml_config.dataset.eval_xray_dir: direction for evaluation set images
--ml_config.dataset.XML_dir: direction for projection geometries file
--ml_config.dataset.eval_dataset: do not change
--ml_config.dataset.name=ff_epipolar: do not change
--ml_config.dataset.render_style: do not change

--ml_config.dataset.llffhold: 

The parameter llffhold specifies the interval index that determines which images from a list of images are selected for the test set. If you have a list of 200 images and set the parameter llffhold to 10, every 10th image will be extracted for the test set, starting with the image at index 0. The selected images would be: 0, 10, 20, 30, and so on. If the parameter llffhold is set to 8, every 8th image will be selected, such as: 0, 8, 16, 24, and so forth.

--ml_config.dataset.num_interpolation_views=10

num_interpolation_views=10. This value is the parameter KK from the paper, which specifies the number of reference views. NN, which denotes the total number of views, must be adjusted within the code itself. In the code, NN and KK are swapped.

--ml_config.dataset.angle_steps=2

The parameter ml_config.dataset.angle_steps=2 defines the angular resolution of the test set. When set to 1, the test set comprises 400 images with 0.5° increments, covering an angular range from 0° to 200°. Setting this parameter to 2 reduces the test set to 200 images, as images are taken at 1° intervals. When set to 20, the test set includes only 20 images, each captured at 20° intervals.

--ml_config.dataset.batch_size=16: number of rays per batch 
--ml_config.eval.chunk=1024: number of rays that are rendered in one step during evaluation
--ml_config.model.num_rgb_channels=3: do not change









