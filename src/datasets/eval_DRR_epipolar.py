"""Dataset for forwarding facing scene in NeX with reference view."""

from os import path
import matplotlib.pyplot as plt
from gen_patch_neural_rendering.src.datasets.XML_loader import parse_projection_matrices, analyze_xml_file
#import imageio
import imageio.v2 as imageio
from numpy.linalg import svd
import numpy as np
from scipy.linalg import rq

from gen_patch_neural_rendering.src.datasets.ff_epipolar import FFEpipolar
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import pose_utils
from gen_patch_neural_rendering.src.utils import data_types

class EvalXRAYEpipolar(FFEpipolar):
  """Forward Facing epipolar dataset for medical xray images."""

  def _load_renderings_xray(self, args):

    cam2worlds_path = args.dataset.cam2worlds_dir
    intrinsic_matrices_path = args.dataset.I_dir
    focals_path = args.dataset.I0_dir

    camtoworlds = np.load(cam2worlds_path)
    intrinsic_matrices = np.load(intrinsic_matrices_path)
    focals = np.load(focals_path)

    basedir = path.join(args.dataset.eval_DRR_dir, self.scene)
    imgdir = basedir

    height = 976  # args.dataset.ff_image_height
    width = 976  # args.dataset.ff_image_width

    images = self._load_images_tif(imgdir, width, height)

    # images = self._load_images_tif(imgdir, args.dataset.eval_xray_image_width,
    #                           args.dataset.eval_xray_image_height)

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

  ######################################################################################################################
    self.focal = focals
    self.intrinsic_matrix = intrinsic_matrices

    # Get the min and max depth of the scene
    self.min_depth = 420 #??
    self.max_depth = 820 #??

  ######################################################################################################################
    scale = 1 / self.max_depth

    camtoworlds[:, :3, 3] *= scale

    factor_h = 976 / height
    factor_w = 976 / width

    # Passe die Breite entsprechend an
    self.intrinsic_matrix[0, 0] /= factor_w  # Fokallänge in x-Richtung
    self.intrinsic_matrix[0, 2] /= factor_w  # Hauptpunkt in x-Richtung

    # Passe die Höhe entsprechend an
    self.intrinsic_matrix[1, 1] /= factor_h  # Fokallänge in y-Richtung
    self.intrinsic_matrix[1, 2] /= factor_h  # Hauptpunkt in y-Richtung

    self.min_depth = scale * self.min_depth
    self.max_depth = scale * self.max_depth

    self.min_depth = np.array([self.min_depth])
    self.max_depth = np.array([self.max_depth])

    min = self.min_depth.item()
    max = self.max_depth.item()

    args.model.near = min
    args.model.far = max

    # Select the split.
    i_test = np.arange(images.shape[0])[::args.dataset.llffhold]
    print("i_test: ", i_test)
    i_train = np.array(
      [i for i in np.arange(int(images.shape[0])) if i not in i_test])
    print("i_train: ", i_train)

    if self.split == "train":
      indices = i_train
    else:
      indices = i_test

    images = images[indices]
    print(images.shape)
    camtoworlds = camtoworlds[indices]
    print(camtoworlds.shape)

    self.images = images
    self.camtoworlds = camtoworlds
    self.n_examples = images.shape[0]

  def _generate_rays(self):

    pixel_center = 0.5
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
      np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
      np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
      indexing="xy")

    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    inverse_intrisics = np.linalg.inv(self.intrinsic_matrix[Ellipsis, :3, :3])

    # camera_dirs sind Richtungsvektoren im Kamerakoordinatensystem, und sie repräsentieren die Richtungen von der Kamera zu den Pixeln auf dem Bild.
    camera_dirs = (inverse_intrisics[None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    # directions sind die gleichen Richtungsvektoren, jedoch nach der Transformation in Weltkoordinaten, um die Szene zu repräsentieren.
    directions = (self.camtoworlds[:, None, None, :3, :3]
                  @ camera_dirs[None, Ellipsis, None])[Ellipsis, 0]

    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)

    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=viewdirs)





