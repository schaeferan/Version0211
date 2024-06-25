"""Dataset for forwarding facing scene in NeX with reference view."""

import os
from os import path
import matplotlib.pyplot as plt
from gen_patch_neural_rendering.src.datasets.XML_loader import extract_projection_matrices_DRR, process_projection_matrices
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
    """
    Load images and camera information for evaluation.

    Args:
        args: Experiment configuration.
    """
    print("load renderings von EvalXRAY wird aufgerufen")
    ####################################################################################################################

    #xml_file_path = '/home/andre/CONRAD_data/Conrad_base.xml'
    #xml_file_path = "/home/andre/workspace2/CONRAD/SimpleShape.xml"
    xml_file_path = args.dataset.XML_dir

    projection_matrices = extract_projection_matrices_DRR(xml_file_path)
    intrinsic_matrices, camtoworlds = process_projection_matrices(projection_matrices)

    intrinsic_matrix = intrinsic_matrices

    #projection_matrices = projection_matrices[::10]

    # Liste 1 mit den spezifizierten Indexpositionen
    #liste_1_indices = set(range(0, 200, 10))
    #liste_1 = [projection_matrices[i] for i in liste_1_indices]
    # Liste 2 mit den restlichen Indexpositionen
    #liste_2 = [projection_matrices[i] for i in range(200) if i not in liste_1_indices]
    #projection_matrices = liste_1

    #projection_matrices = projection_matrices[::10]
    #projection_matrices = [element for element in projection_matrices if element not in removed_elements]
    #projection_matrices = projection_matrices[:20]
    #projection_matrices = projection_matrices[89:109]
    #self.projection_matrices = np.array(projection_matrices)


    ## Überprüfen resultierenden Dictionary
    # if result_dict is not None:
    # for key, value in result_dict.items():
    # print(f"{key}: {value}")

    # Bilder laden #####################################################################################################

    basedir = path.join(args.dataset.eval_xray_dir, self.scene)

#    img0 = [
#        os.path.join(basedir, "images", f)
#        for f in sorted(file_utils.listdir(os.path.join(basedir, "images")))
#        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
#    ][0]
#    with file_utils.open_file(img0) as f:
#      sh = imageio.imread(f).shape
#    if sh[0] / sh[
#        1] != args.dataset.ff_image_height / args.dataset.ff_image_width:
#      raise ValueError("not expected height width ratio")

#    factor = 1
#    #factor = sh[0] / args.dataset.ff_image_height

    imgdir = basedir

    height = 976#args.dataset.ff_image_height
    width = 976#args.dataset.ff_image_width

    images = self._load_1tif(imgdir)
    print("86 images shape:", images.shape)

    images = images.astype(np.uint8)

    self.h, self.w = images.shape[1:3]
    self.resolution = self.h * self.w
    self.images = images
    #self.focal = 3821.2
########################################################################################################################


########################################################################################################################

    # Get the min and max depth of the scene
    self.min_depth = 420
    self.max_depth = 820

    scale = 1/self.max_depth
    camtoworlds[:, :3, 3] *= scale

    factor_h = 976 / height
    factor_w = 976 / width

    # # Passe die Breite entsprechend an
    # self.intrinsic_matrix[0, 0] /= factor_w  # Fokallänge in x-Richtung
    # self.intrinsic_matrix[0, 2] /= factor_w  # Hauptpunkt in x-Richtung
    #
    # # Passe die Höhe entsprechend an
    # self.intrinsic_matrix[1, 1] /= factor_h  # Fokallänge in y-Richtung
    # self.intrinsic_matrix[1, 2] /= factor_h  # Hauptpunkt in y-Richtung

    self.min_depth = scale * self.min_depth
    self.max_depth = scale * self.max_depth

    #self.min_depth = (self.min_depth,)
    #self.max_depth = (self.max_depth,)

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
    print("images shape: ", images.shape)
    camtoworlds = camtoworlds[indices]
    print("cam2worlds shape: ", camtoworlds.shape)
    intrinsic_matrix = intrinsic_matrices[indices]
    print("intrinsics shape: ", intrinsic_matrix)

    self.images = images
    self.camtoworlds = camtoworlds
    self.intrinsic_matrix = intrinsic_matrix
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
    camera_dirs = (inverse_intrisics[:,None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    # directions sind die gleichen Richtungsvektoren, jedoch nach der Transformation in Weltkoordinaten, um die Szene zu repräsentieren.
    directions = (self.camtoworlds[:, None, None, :3, :3] @ camera_dirs[Ellipsis, None])[Ellipsis, 0]

    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)

    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=viewdirs)
