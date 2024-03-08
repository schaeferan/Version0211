"""Dataset for forwarding facing scene in NeX with reference view."""

import os
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

    projection_matrices = parse_projection_matrices(xml_file_path)
    projection_matrices = projection_matrices[:args.dataset.eval_length]
    #self.projection_matrices = np.array(projection_matrices)

    XML_dict = analyze_xml_file(xml_file_path)
    self.XML_dict = XML_dict

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

    images = self._load_images_tif(imgdir, width, height)

    #images = self._load_images_tif(imgdir, args.dataset.eval_xray_image_width,
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
    self.focal = 3934.43
########################################################################################################################
    # # Erstelle leere Listen, um intrinsische und extrinsische Parameter für jede Projektionsmatrix zu speichern
    # intrinsics_list = []
    # extrinsics_list = []
    #
    # for P in projection_matrices:
    #   # # Wende SVD auf die Projektionsmatrix an
    #   # U, S, Vt = svd(P)
    #   #
    #   # # Extrahiere die intrinsische Matrix K
    #   # K = U[:, :3] @ np.diag(S[:3]) @ Vt[:3, :]
    #   #
    #   # # Extrahiere die extrinsische Matrix [R | T]
    #   # R = U[:, :3]
    #   # T = (1 / S[0]) * Vt[3, :]
    #   #
    #   # # Füge die intrinsischen und extrinsischen Parameter zur jeweiligen Liste hinzu
    #   # intrinsics_list.append(K)
    #   # extrinsics_list.append(np.hstack((R, T.reshape(3, 1))))
    #
    #   #########################################################################################################
    #   # Extrahiere die intrinsische Matrix
    #   K = P[:, :3]#Das ist doch nicht die intrinsic??
    #
    #   # Extrahiere die extrinsische Matrix [R | T]
    #   R = np.linalg.inv(K) @ P[:, :3]
    #   T = np.linalg.inv(K) @ P[:, 3]
    #
    #   # Füge die intrinsische und extrinsische Matrizen zur jeweiligen Liste hinzu
    #   intrinsics_list.append(K)
    #   extrinsics_list.append(np.hstack((R, T.reshape(3, 1))))
    #   #########################################################################################################
    #
    #   #M = P[:3,:3]
    #   #R2, Q2 = rq(M)
    #
    #   #K = R2
    #   #R = Q2
    #
    #   #intrinsics_list.append(K)
    #   #xtrinsics_list.append(R)
    #
    # # Konvertiere die Listen in NumPy-Arrays
    # intrinsics_array = np.array(intrinsics_list)
    # extrinsics_array = np.array(extrinsics_list)
    # extrinsic_matrices.append(np.hstack((R, t.reshape(3, 1))))
########################################################################################################################
    self.intrinsic_matrix = np.array([[3934.43, 0, 488, 0],
                                      [0, 3934.43, 488, 0],
                                      [0, 0, 1, 0]]).astype(np.float32)
    extrinsic_matrices = []
    for P in projection_matrices:

        K = self.intrinsic_matrix[:,:3]
        # Zerlege die Projektionsmatrix P in [R | t]
        K_inverse = np.linalg.inv(K)
        [R, t] = np.dot(K_inverse, P)[:3, :].copy(), np.dot(K_inverse, P)[:3, 3].copy()

        extrinsic_matrices.append(R)

    extrinsics_array = np.array(extrinsic_matrices)
    camtoworlds = extrinsics_array

########################################################################################################################

    # Convert R matrix from the form [up forward left] to [right up back]
    camtoworlds = np.concatenate(
        [-camtoworlds[:, 2:3, :], camtoworlds[:, 0:1, :], -camtoworlds[:, 1:2, :]], 1)


    # # Use this to set the near and far plane
    # args.model.near = self.min_depth.item()
    # args.model.far = self.max_depth.item()

    # Get the min and max depth of the scene
    self.min_depth = 420
    self.max_depth = 820

    scale = 1/self.max_depth

    camtoworlds[:, :3, 3] *= scale

    # Transformation der Kamerakoordinaten definieren
    self.cam_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                   [0, 0, 0, 1]])
    self.cam_transform_3x3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    #bds *= scale
    camtoworlds_copy = camtoworlds.copy()
    camtoworlds_copy = pose_utils.recenter_poses(camtoworlds, None)
    camtoworlds = pose_utils.recenter_poses(camtoworlds, self.cam_transform)

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

    #self.min_depth = (self.min_depth,)
    #self.max_depth = (self.max_depth,)

    self.min_depth = np.array([self.min_depth])
    self.max_depth = np.array([self.max_depth])

    min = self.min_depth.item()
    max = self.max_depth.item()

    args.model.near = min
    args.model.far = max

    # # Select the split.
    # i_train = np.arange(images.shape[0])
    # i_test = np.array([0])

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
    projection_matrices = np.array(projection_matrices)
    projection_matrices = projection_matrices[indices]

    self.images = images
    self.camtoworlds = camtoworlds
    self.projection_matrices = projection_matrices

    self.n_examples = images.shape[0]

  def _generate_rays(self):

    # #self.projection_matrices = np.array(self.projection_matrices)
    #
    # #origins_pro = np.array([-np.linalg.inv(m[:3, :3]) @ m[:, 3] for m in self.projection_matrices])
    # #directions = np.array([np.linalg.inv(m[:3, :3]) for m in self.projection_matrices])
    #
    # pixel_center = 0.0
    # x, y = np.meshgrid(
    #   np.arange(self.w, dtype=np.float32) + pixel_center,
    #   np.arange(self.h, dtype=np.float32) + pixel_center,
    #   indexing="xy"
    # )
    # pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    #
    # directions = []
    #
    # for m in self.projection_matrices:
    #   #M = m[:3, :3]
    #   #inv_ARR = np.linalg.inv(M)
    #   directions.append((np.linalg.inv(m[:3, :3]) @ pixels.reshape(-1, 3).T).T)
    #
    # origins_pro = np.array([-np.linalg.inv(m[:3, :3]) @ m[:, 3] for m in self.projection_matrices])
    # origins_pro = origins_pro[:, None, None, :]
    # directions = np.array(directions).reshape(self.projection_matrices.shape[0], self.h, self.w, 3)
    # #directions = (self.camtoworlds[:, None, None, :3, :3]
    # #              @ directions[None, Ellipsis, None])[Ellipsis, 0]
    # directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    #
    # origins = np.broadcast_to(origins_pro, directions.shape)
    #
    # ## Calculate the norms of the direction vectors along the last dimension
    # #norms = np.linalg.norm(directions, axis=2)
    # ## Normalize the direction vectors by dividing each element by its corresponding norm
    # #normalized_directions = directions / norms[:, :, np.newaxis]
    # ## Extract the direction vectors from the third column of each 3x3 matrix
    # #normalized_directions = normalized_directions[:, :, 2]
    #
    # #viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    pixel_center = 0.0
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

    self.rays = data_types.Rays(origins=origins, directions=directions)
