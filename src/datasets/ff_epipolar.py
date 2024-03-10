# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset for forwarding facing scene in IBRNet with reference view."""#ImageBasedRendering
from pathlib import Path

INTERNAL = False  # pylint: disable=g-statement-before-imports
import os
from os import path

if not INTERNAL:
  import cv2  # pylint: disable=g-import-not-at-top
#import imageio
import imageio.v2 as imageio
import jax
import re
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import tifffile as tiff

from gen_patch_neural_rendering.src.datasets.base import BaseDataset
from gen_patch_neural_rendering.src.utils import data_types
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import pose_utils

from gen_patch_neural_rendering.src.datasets.XML_loader import parse_projection_matrices, analyze_xml_file

class FFEpipolar(BaseDataset):
  """Forward Facing epipolar dataset."""

  def __init__(self, split, args, scene, train_ds=None):

    assert args.dataset.batching in [
        "single_image",
    ], "Only single image batching supported for now"

    if split == "test":
      self.train_images = train_ds.images
      self.train_rays = train_ds.rays
      self.train_worldtocamera = train_ds.worldtocamera
      self.train_camtoworlds = train_ds.camtoworlds

    self.num_ref_views = args.dataset.num_interpolation_views#10
    self.scene = scene

    super(FFEpipolar, self).__init__(split, args)

  def _train_init(self, args):
    super(FFEpipolar, self)._train_init(args)

    #--------------------------------------------------------------------------------------
    # Get world to camera matrices
    # To compute world to camera matrix, conver the 3x4 camtoworld matrix to 4x4
    # matrix.
    bottom = np.tile(
        np.reshape([0, 0, 0, 1.], [1, 1, 4]), (self.camtoworlds.shape[0], 1, 1))
    c2w = np.concatenate([self.camtoworlds, bottom], -2)
    self.worldtocamera = np.linalg.inv(c2w)

    #--------------------------------------------------------------------------------------
    # Compute the nearest neighbors for each train view.
    cam_train_pos = self.camtoworlds[:, :3, -1]  # Camera positions
    train2traincam = np.linalg.norm(
        cam_train_pos[:, None] - cam_train_pos[None, :], axis=-1)
    # To avoid self-mapping
    np.fill_diagonal(train2traincam, np.inf)
    self.sorted_near_cam = np.argsort(train2traincam, axis=-1)[Ellipsis, :-1]

  def _test_init(self, args):
    super(FFEpipolar, self)._test_init(args)

    if args.dataset.render_path:
      # Compute nearest cameras for render poses
      curr_camtoworlds = self.render_poses
    else:
      # Compute nearest cameras for test poses
      curr_camtoworlds = self.camtoworlds

    bottom = np.tile(
        np.reshape([0, 0, 0, 1.], [1, 1, 4]), (curr_camtoworlds.shape[0], 1, 1))
    c2w = np.concatenate([curr_camtoworlds, bottom], -2)
    self.inv_camtoworlds = np.linalg.inv(c2w)

    #--------------------------------------------------------------------------------------
    # Compute the nearest neighbors for each test/render view.
    cam_train_pos = self.train_camtoworlds[:, :3, -1]
    cam_test_pos = curr_camtoworlds[:, :3, -1]
    test2traincam = np.linalg.norm(
        cam_test_pos[:, None] - cam_train_pos[None, :], axis=-1)
    self.sorted_near_cam = np.argsort(
        test2traincam, axis=-1)[Ellipsis, :self.num_ref_views]

  def _next_train(self):
    """Sample batch for training."""
    if self.batching == "single_image":
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays.batch_shape[1],
                                      (self.batch_size,))

      #--------------------------------------------------------------------------------------
      # Get batch pixels and rays
      l_devices = jax.local_device_count()
      batch_pixels = self.images[image_index][ray_indices]
      batch_target_worldtocam = np.tile(self.worldtocamera[image_index],
                                        (l_devices, 1, 1))
      batch_rays = jax.tree_map(lambda r: r[image_index][ray_indices],
                                self.rays)

      #--------------------------------------------------------------------------------------
      # Get index of reference views
      # During training for additional regularization we chose a random number
      # of reference view for interpolation
      # Top k number of views to consider when randomly sampling
      total_views = 20
      # Number of reference views to select
      # num_select = self.num_ref_views + np.random.randint(low=-2, high=3)
      num_select = self.num_ref_views

      # Get the set of precomputed nearest camera indices
      batch_near_cam_idx = self.sorted_near_cam[image_index][:total_views]
      batch_near_cam_idx = np.random.choice(
          batch_near_cam_idx,
          min(num_select, len(batch_near_cam_idx)),
          replace=False)

      #--------------------------------------------------------------------------------------
      # Get the reference data
      ref_images = self.images[batch_near_cam_idx]
      ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 3)
      #ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 3)

      ref_cameratoworld = self.camtoworlds[batch_near_cam_idx]
      ref_worldtocamera = self.worldtocamera[batch_near_cam_idx]

      # Each of these reference data need to be shared onto each local device.
      # To support this we replicate the reference data as many times as there
      # are local devices
      target_view = data_types.Views(rays=batch_rays, rgb=batch_pixels)
      reference_views = data_types.ReferenceViews(
          rgb=np.tile(ref_images, (l_devices, 1, 1, 1)),
          target_worldtocam=batch_target_worldtocam,
          ref_worldtocamera=np.tile(ref_worldtocamera, (l_devices, 1, 1)),
          ref_cameratoworld=np.tile(ref_cameratoworld, (l_devices, 1, 1)),
          intrinsic_matrix=np.tile(self.intrinsic_matrix[None, :],
                                   (l_devices, 1, 1)),
          min_depth=np.tile(self.min_depth[None, :], (l_devices, 1)),
          max_depth=np.tile(self.max_depth[None, :], (l_devices, 1)),
      )
      return_batch = data_types.Batch(
          target_view=target_view, reference_views=reference_views)

    else:
      raise ValueError("Batching {} not implemented".format(self.batching))

    return return_batch

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    l_devices = jax.local_device_count()

    if self.render_path:
      target_view = data_types.Views(
          rays=jax.tree_map(lambda r: r[idx], self.render_rays),)
    else:
      target_view = data_types.Views(
          rays=jax.tree_map(lambda r: r[idx], self.rays), rgb=self.images[idx])

    batch_target_worldtocam = np.tile(self.inv_camtoworlds[idx],
                                      (l_devices, 1, 1))
    #--------------------------------------------------------------------------------------
    # Get the reference data
    batch_near_cam_idx = self.sorted_near_cam[idx]
    ref_images = self.train_images[batch_near_cam_idx]

    #if args.dataset.eval_dataset == "xray" :
    #    ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 1)
    ref_images = ref_images.reshape(ref_images.shape[0], self.h, self.w, 3)

    ref_cameratoworld = self.train_camtoworlds[batch_near_cam_idx]
    ref_worldtocamera = self.train_worldtocamera[batch_near_cam_idx]

    #--------------------------------------------------------------------------------------
    # Replicate these so that they may be distributed onto several devices for
    # parallel computaion.
    reference_views = data_types.ReferenceViews(
        rgb=np.tile(ref_images, (l_devices, 1, 1, 1)),
        target_worldtocam=batch_target_worldtocam,
        ref_worldtocamera=np.tile(ref_worldtocamera, (l_devices, 1, 1)),
        ref_cameratoworld=np.tile(ref_cameratoworld, (l_devices, 1, 1)),
        intrinsic_matrix=np.tile(self.intrinsic_matrix[None, :],
                                 (l_devices, 1, 1)),
        min_depth=np.tile(self.min_depth[None, :], (l_devices, 1)),
        max_depth=np.tile(self.max_depth[None, :], (l_devices, 1)),
    )

    return_batch = data_types.Batch(
        target_view=target_view, reference_views=reference_views)

    return return_batch

  def _load_images_tif(self, imgdir, w, h):
      """Function to load all images.

      Args:
        imgdir: Location of images.
        w: image width.
        h: image height.

      Returns:
        images: Loaded images.
      """

      def load_single_image(f):

          # # Load the image
          # image = io.imread(f)
          # # Resize to the specified dimensions
          # image = cv2.resize(image, dsize=(w, h))
          # # Convert to grayscale (if not already)
          # if len(image.shape) == 3:
          #     image = rgb2gray(image)
          # # Scale the pixel values to 0-255
          # image = (image * 255).astype(np.uint8)
          # return image

          return cv2.resize(tiff.imread(f), dsize=(w, h))



      # if not file_utils.file_exists(imgdir):
      #     raise ValueError("Image folder {} doesn't exist.".format(imgdir))

      imgfiles = [
          path.join(imgdir, f)
          #for f in sorted(file_utils.listdir(imgdir))
          for f in sorted(file_utils.listdir(imgdir), key=lambda x: int(re.search(r'\d+', x).group()))
          #if f.endswith(".tif")  # Filtern Sie .tif-Dateien
      ]

      images = [load_single_image(f) for f in imgfiles]
      images = np.stack(images, axis=-1)

      # Finde den globalen Mindest- und Maximalwert über alle Bilder
      min_value = np.min(images)
      max_value = np.max(images)

      ## Grauwerte
      ## Skaliere alle Bilder im Bezug auf den globalen Mindest- und Maximalwert
      #scaled_images = (images - min_value) / (max_value - min_value) * 255

      scaled_images = images / max_value

      # test = images[0]
      #
      # # Initialisiere min_value und max_value mit den Werten des ersten Bildes in der Liste
      # min_value = np.min(images[0])
      # max_value = np.max(images[0])
      #
      # # Durchlaufe die restlichen Bilder in der Liste, um den globalen Mindest- und Maximalwert zu finden
      # for image in images[1:]:
      #     image_min = np.min(image)
      #     image_max = np.max(image)
      #     min_value = min(min_value, image_min)
      #     max_value = max(max_value, image_max)
      #
      # # Nun hast du den globalen Mindest- und Maximalwert über alle Bilder
      #
      # # Skaliere jedes Bild in der Liste im Bezug auf den globalen Mindest- und Maximalwert
      # scaled_images = [(image - min_value) / (max_value - min_value) * 255 for image in images]

      return scaled_images

  def _load_images(self, imgdir, w, h):
    """Function to load all images.

    Args:
      imgdir: Location of images.
      w: image width.
      h: image height.

    Returns:
      images: Loaded images.
    """

    def imread(fs):
      if fs.endswith("png"):
        with file_utils.open_file(fs) as f:
          return imageio.imread(f)#, ignoregamma=True)
      else:
        with file_utils.open_file(fs) as f:
          return imageio.imread(f)

    def load_single_image(f):
      return cv2.resize(imread(f)[Ellipsis, :3], dsize=(w, h))

    if not file_utils.file_exists(imgdir):
      raise ValueError("Image folder {} doesn't exist.".format(imgdir))

    imgfiles = [
        path.join(imgdir, f)
        for f in sorted(file_utils.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    images = [load_single_image(f) for f in imgfiles]
    images = np.stack(images, axis=-1)
    return images

  def _load_renderings_xray(self, args):

      print("load renderings von FFEpi für Big train (XRAY version)")
      """
          Load images and camera information for evaluation.

          Args:
              args: Experiment configuration.
          """
      ####################################################################################################################

      # xml_file_path = '/home/andre/CONRAD_data/Conrad_base.xml'
      #xml_file_path = "/home/andre/workspace2/CONRAD/SimpleShape.xml"
      xml_file_path = args.dataset.XML_dir

      projection_matrices = parse_projection_matrices(xml_file_path)
      #projection_matrices = projection_matrices[9:11]
      #projection_matrices = projection_matrices[:args.dataset.eval_length]
      # self.projection_matrices = np.array(projection_matrices)

      XML_dict = analyze_xml_file(xml_file_path)
      self.XML_dict = XML_dict

      ## Überprüfen resultierenden Dictionary
      # if result_dict is not None:
      # for key, value in result_dict.items():
      # print(f"{key}: {value}")

      # Bilder laden #####################################################################################################

      basedir = path.join(args.dataset.xray_base_dir, self.scene)

      img0 = [
          os.path.join(basedir, "images", f)
          for f in sorted(file_utils.listdir(os.path.join(basedir, "images")))
          if f.endswith("tif")
      ][0]

      assert Path(img0).is_file()

      with file_utils.open_file(img0, 'rb') as f:
          # sh = imageio.imread(f).shape
          img = imageio.imread(f)
          sh = img.shape
      if sh[0] / sh[
          1] != args.dataset.xray_image_height / args.dataset.xray_image_width:
          raise ValueError("not expected height width ratio")

      imgdir = os.path.join(basedir, "images")

      #xray_image_width = 976
      #xray_image_height = 976

      images = self._load_images_tif(imgdir, args.dataset.xray_image_height,
                                     args.dataset.xray_image_height)
      print("images shape: ", images.shape)

      # Transpose such that the first dimension is number of images
      images = np.moveaxis(images, -1, 0)

      if args.model.num_rgb_channels == 3:
        # Annahme: grayscale_images ist das ursprüngliche Array mit der Form (10, 976, 976)
        # Füge eine zusätzliche Dimension hinzu, um Platz für die RGB-Kanäle zu schaffen
        images = np.expand_dims(images, axis=-1)
        # # Wiederhole den Kanal 3-mal, um eine 3-Kanal-RGB-Darstellung zu erstellen
        images = np.repeat(images, 3, axis=-1)

      images = images.astype(np.uint8)

      self.h, self.w = images.shape[1:3]
      self.resolution = self.h * self.w
      self.images = images
      self.focal = 3934.43

      self.intrinsic_matrix = np.array([[3934.43, 0, 488, 0],
                                        [0, 3934.43, 488, 0],
                                        [0, 0, 1, 0]]).astype(np.float32)
      extrinsic_matrices = []
      for P in projection_matrices:
          K = self.intrinsic_matrix[:, :3]
          # Zerlege die Projektionsmatrix P in [R | t]
          K_inverse = np.linalg.inv(K)
          [R, t] = np.dot(K_inverse, P)[:3, :].copy(), np.dot(K_inverse, P)[:3, 3].copy()

          extrinsic_matrices.append(R)

      extrinsics_array = np.array(extrinsic_matrices)
      camtoworlds = extrinsics_array

      # Convert R matrix from the form [up forward left] to [right up back]
      camtoworlds = np.concatenate(
          [-camtoworlds[:, 2:3, :], camtoworlds[:, 0:1, :], -camtoworlds[:, 1:2, :]], 1)

      # Get the min and max depth of the scene
      self.min_depth = 420
      self.max_depth = 820

      scale = 1 / self.max_depth

      camtoworlds[:, :3, 3] *= scale

      # Transformation der Kamerakoordinaten definieren
      self.cam_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                     [0, 0, 0, 1]])
      self.cam_transform_3x3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

      # bds *= scale
      camtoworlds_copy = camtoworlds.copy()
      camtoworlds_copy = pose_utils.recenter_poses(camtoworlds, None)
      camtoworlds = pose_utils.recenter_poses(camtoworlds, self.cam_transform)

      self.min_depth = scale * self.min_depth
      self.max_depth = scale * self.max_depth

      # self.min_depth = (self.min_depth,)
      # self.max_depth = (self.max_depth,)

      self.min_depth = np.array([self.min_depth])
      self.max_depth = np.array([self.max_depth])

      min = self.min_depth.item()
      max = self.max_depth.item()

      args.model.near = min
      args.model.far = max

      # # # Select the split.
      # # i_train = np.arange(images.shape[0])
      # # i_test = np.array([0])
      #
      # # Select the split.
      # i_test = np.arange(images.shape[0])[::args.dataset.llffhold]
      # print(i_test)
      # i_train = np.array(
      #     [i for i in np.arange(int(images.shape[0])) if i not in i_test])
      # print(i_train)
      #
      # if self.split == "train":
      #     indices = i_train
      # else:
      #     indices = i_test
      #
      # if self.split == "test":
      #     self.render_poses = pose_utils.generate_spiral_poses(
      #         poses, bds, self.cam_transform)

      # Select the split.
      i_train = np.arange(images.shape[0])
      i_test = np.array([0])

      if self.split == "train":
          indices = i_train
      else:
          indices = i_test

      images = images[indices]
      print("images shape[0]: ", images.shape[0])

      camtoworlds = camtoworlds[indices]
      print("poses shape[0]: ", camtoworlds.shape[0])

      projection_matrices = np.array(projection_matrices)
      projection_matrices = projection_matrices[indices]

      self.images = images
      self.camtoworlds = camtoworlds
      self.projection_matrices = projection_matrices

      self.n_examples = images.shape[0]
  def _load_renderings(self, args):
    """Load images and camera information."""
    print("load renderings von FFEpi für Big train (normale version)")
    #-------------------------------------------
    # Load images.
    #-------------------------------------------
    basedir = path.join(args.dataset.ff_base_dir, self.scene)
    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(file_utils.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]

    assert Path(img0).is_file()

    with file_utils.open_file(img0, 'rb') as f:
      # sh = imageio.imread(f).shape
      img = imageio.imread(f)
      sh = img.shape
    if sh[0] / sh[
        1] != args.dataset.ff_image_height / args.dataset.ff_image_width:
      raise ValueError("not expected height width ratio")

    factor = sh[0] / args.dataset.ff_image_height

    sfx = "_4"
    imgdir = os.path.join(basedir, "images" + sfx)
    if not file_utils.file_exists(imgdir):
      imgdir = os.path.join(basedir, "images")
      if not file_utils.file_exists(imgdir):
        raise ValueError("{} does not exist".format(imgdir))

    images = self._load_images(imgdir, args.dataset.ff_image_width,
                               args.dataset.ff_image_height)
    print("images shape: ", images.shape)

    #-------------------------------------------
    # Load poses and bds.
    #-------------------------------------------
    with file_utils.open_file(path.join(basedir, "poses_bounds.npy"),
                              "rb") as fp:
      poses_arr = np.load(fp)

    self.cam_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                   [0, 0, 0, 1]])
    # poses_arr contains an array consisting of a 3x4 pose matrices and
    # 2 depth bounds for each image. The pose matrix contain [R t] as the
    # left 3x4 matrix
    # pose_arr has shape (...,14) {3x4 + 2}
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Convert R matrix from the form [down right back] to [right up back]
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)

    # Transpose such that the first dimension is number of images
    images = np.moveaxis(images, -1, 0)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    if args.dataset.normalize:
      scale = 1. / bds.max()
    else:
      scale = 1. / (bds.min() * .75)

    poses[:, :3, 3] *= scale
    bds *= scale
    poses = pose_utils.recenter_poses(poses, self.cam_transform)

    # Get the min and max depth of the scene
    self.min_depth = np.array([bds.min()])
    self.max_depth = np.array([bds.max()])

    # Use this to set the near and far plane
    args.model.near = self.min_depth.item()
    args.model.far = self.max_depth.item()

    if self.split == "test":
      self.render_poses = pose_utils.generate_spiral_poses(
          poses, bds, self.cam_transform)
########################################################################################################################
    # Select the split.
    i_train = np.arange(images.shape[0])
    i_test = np.array([0])

    if self.split == "train":
      indices = i_train
    else:
      print("ff_epipolar.py split == test")
      indices = i_test

    images = images[indices]
    print("images shape[0]: ", images.shape[0])
    poses = poses[indices]
    print("poses shape[0]: ", poses.shape[0])

    self.images = images

    self.camtoworlds = poses[:, :3, :4]

    # intrinsic arr has H, W, fx, fy, cx, cy
    self.focal = poses[0, -1, -1] * 1. / factor
    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w

    if args.dataset.render_path and self.split == "test":
      self.n_examples = self.render_poses.shape[0]
    else:
      self.n_examples = images.shape[0]

    self.intrinsic_matrix = np.array([[self.focal, 0, (self.w / 2), 0],
                                      [0, self.focal, (self.h / 2), 0],
                                      [0, 0, 1, 0]]).astype(np.float32)

  def _generate_rays(self):
    """Generate normalized device coordinate rays for llff."""
    if self.split == "test":
      n_render_poses = self.render_poses.shape[0]
      self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
                                        axis=0)

    super()._generate_rays()

    # Split poses from the dataset and generated poses
    if self.split == "test":
      self.camtoworlds = self.camtoworlds[n_render_poses:]
      split_origins = np.split(self.rays.origins, [n_render_poses], 0)
      split_directions = np.split(self.rays.directions, [n_render_poses], 0)

      self.render_rays = data_types.Rays(
          origins=split_origins[0], directions=split_directions[0])

      self.rays = data_types.Rays(
          origins=split_origins[1], directions=split_directions[1])
