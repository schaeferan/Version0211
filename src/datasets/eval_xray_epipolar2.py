"""Dataset for forwarding facing scene in NeX with reference view."""

import os
from os import path
from gen_patch_neural_rendering.src.datasets.XML_loader import parse_projection_matrices, analyze_xml_file
#import imageio
import imageio.v2 as imageio

import numpy as np


from gen_patch_neural_rendering.src.datasets.ff_epipolar import FFEpipolar
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import pose_utils
from gen_patch_neural_rendering.src.utils import data_types


class EvalXRAYEpipolar(FFEpipolar):
  """Forward Facing epipolar dataset for medical xray images."""


  def _load_renderings(self, args):
    """
    Load images and camera information for evaluation.

    Args:
        args: Experiment configuration.
    """

    # Bilder laden #####################################################################################################

    basedir = path.join(args.dataset.eval_xray_dir, self.scene)
    imgdir = basedir


    images = self._load_images_tif(imgdir, args.dataset.eval_xray_image_width,
                               args.dataset.eval_xray_image_height)

    # Transpose such that the first dimension is number of images
    images = np.moveaxis(images, -1, 0)

    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w

    #if self.split == "test":
      #self.render_poses = pose_utils.generate_spiral_poses(
          #poses_copy, bds, self.cam_transform)

    #i_test = np.arange(images.shape[0])[::args.dataset.llffhold]
    #i_train = np.array(
        #[i for i in np.arange(int(images.shape[0])) if i not in i_test])

    #if self.split == "train":
      #indices = i_train
    #else:
      #indices = i_test
    #images = images[indices]
    #poses = poses[indices]

    self.images = images

    ####################################################################################################################

    xml_file_path = '/home/andre/CONRAD_data/Conrad_base.xml'

    projection_matrices = parse_projection_matrices(xml_file_path)
    self.projection_matrices = projection_matrices

    XML_dict = analyze_xml_file(xml_file_path)
    self.XML_dict = XML_dict

    ## Überprüfen resultierenden Dictionary
    #if result_dict is not None:
      #for key, value in result_dict.items():
        #print(f"{key}: {value}")

  def _generate_rays(self):

    self.projection_matrices = np.array(self.projection_matrices)

    #origins_pro = np.array([-np.linalg.inv(m[:3, :3]) @ m[:, 3] for m in self.projection_matrices])
    #directions = np.array([np.linalg.inv(m[:3, :3]) for m in self.projection_matrices])

    pixel_center = 0.5  # Oder 0.0, je nach Bedarf
    x, y = np.meshgrid(
      np.arange(self.w, dtype=np.float32) + pixel_center,
      np.arange(self.h, dtype=np.float32) + pixel_center,
      indexing="xy"
    )
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)

    directions = []

    for m in self.projection_matrices:
      intrinsic_matrix = m[:3, :3]
      inverse_intrinsic = np.linalg.inv(intrinsic_matrix)
      directions.append((inverse_intrinsic @ pixels.reshape(-1, 3).T).T)

    origins_pro = np.array([-np.linalg.inv(m[:3, :3]) @ m[:, 3] for m in self.projection_matrices])
    origins_pro = origins_pro[:, None, None, :]

    directions = np.array(directions).reshape(self.projection_matrices.shape[0], self.h, self.w, 3)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    origins = np.broadcast_to(origins_pro,
                                       directions.shape)

    ## Calculate the norms of the direction vectors along the last dimension
    #norms = np.linalg.norm(directions, axis=2)
    ## Normalize the direction vectors by dividing each element by its corresponding norm
    #normalized_directions = directions / norms[:, :, np.newaxis]
    ## Extract the direction vectors from the third column of each 3x3 matrix
    #normalized_directions = normalized_directions[:, :, 2]

    #viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=directions)