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

"""A base class for all the datasets. Based on dataset in jaxnerf."""
import queue
import threading
import jax
import numpy as np

from gen_patch_neural_rendering.src.utils import data_types
from gen_patch_neural_rendering.src.utils import data_utils


class BaseDataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, args):
    """
    Initialize the BaseDataset class.

    Args:
        split (str): Dataset split ('train' or 'test').
        args: Experiment configuration.
    """

    #Der Aufruf von super(BaseDataset, self).__init__() stellt sicher,
    #dass der Konstruktor der Elternklasse vor den spezifischen Initialisierungen der abgeleiteten Klasse aufgerufen wird.
    #Dies ist wichtig, um sicherzustellen, dass der Thread korrekt initialisiert wird und alle erforderlichen Vorbereitungen getroffen werden,
    #bevor die benutzerdefinierte Funktionalität in BaseDataset aktiviert wird.
    super(BaseDataset, self).__init__()

    self.queue = queue.Queue(6)  # Set prefetch buffer to 6 batches.
    self.daemon = True
    self.split = split

    self.use_pixel_centers = args.dataset.use_pixel_centers
    if split == "train":
      self._train_init(args)
    elif split == "test":
      self._test_init(args)
    else:
      raise ValueError(
          "the split argument should be either \"train\" or \"test\", set"
          "to {} here.".format(split))
    self.batch_size = args.dataset.batch_size // jax.host_count()
    self.batching = args.dataset.batching
    self.render_path = args.dataset.render_path

    self.resolution = self.h * self.w
    self.start()

  def __iter__(self):
    return self

  def __next__(self):
    """
    Get the next training batch or test example.

    Returns:
      batch: data_types.Batch.
    """
    x = self.queue.get()
    if self.split == "train":
      return data_utils.shard(x)
    else:
      return data_utils.to_device(x)

  def peek(self):
    """
    Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: data_types.Batch".
    """
    while self.queue.empty():
      x = None
    # Make a copy of the front of the queue.
    x = jax.tree_map(lambda x: x.copy(), self.queue.queue[0])
    if self.split == "train":
      return data_utils.shard(x)
    else:
      return data_utils.to_device(x)

  def run(self):
    """
    Run the dataset thread.
    """
    if self.split == "train":
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())

  @property
  def size(self):
    """
    Get the size of the dataset.

    Returns:
        int: The number of examples in the dataset.
    """
    return self.n_examples

  def _train_init(self, args):
    """
    Initialize training dataset.

    Args:
        args: Experiment configuration.
    """
    self._load_renderings(args)
    self._generate_rays()

    if args.dataset.batching == "single_image":

      if args.dataset.eval_dataset == "xray":

        self.images = self.images.reshape([-1, self.resolution, 3])#shape 3 ????
        self.rays = jax.tree_map(
          lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)

      else:

        self.images = self.images.reshape([-1, self.resolution, 3])
        self.rays = jax.tree_map(
            lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
    else:
      raise NotImplementedError(
          f"{args.dataset.batching} batching strategy is not implemented.")

  def _test_init(self, args):
    """
    Initialize test dataset.

    Args:
        args: Experiment configuration.
    """
    self._load_renderings(args)
    self._generate_rays()
    self.it = 0

  def _next_train(self):
    """
    Sample next test example.

    Returns:
        data_types.Batch: A batch of test data.
    """


    if self.batching == "single_image":
      # Choose a random image
      image_index = np.random.randint(0, self.n_examples, ())
      # Choose ray indices for this image
      # Ray elements have a shape of (num_train_images, resolution, _)
      ray_indices = np.random.randint(0, self.rays.batch_shape[1],
                                      (self.batch_size,))
      batch_pixels = self.images[image_index][ray_indices]
      batch_rays = jax.tree_map(lambda r: r[image_index][ray_indices],
                                self.rays)

    else:
      raise NotImplementedError(
          f"{self.batching} batching strategy is not implemented.")

    target_view = data_types.Views(rays=batch_rays, rgb=batch_pixels)
    return data_types.Batch(target_view=target_view)

  def _next_test(self):
    """Sample next test example."""
    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      rays = jax.tree_map(lambda r: r[idx], self.render_rays)
      target_view = data_types.Views(rays=rays)
      return data_types.Batch(target_view=target_view)

    else:
      rays = jax.tree_map(lambda r: r[idx], self.rays)
      pixels = self.images[idx]
      target_view = data_types.Views(rays=rays, rgb=pixels)
      return data_types.Batch(target_view=target_view)

  def _load_renderings(self, args):
    """
    Load renderings for the dataset.

    Args:
        args: Experiment configuration.
    """

    raise NotImplementedError

  def _generate_rays(self):
    """
    Generate rays for all the views in the dataset.
    """
    # Es handelt sich jedoch nicht um eine vollständige Raytracing- oder Rasterisierungsberechnung, sondern um den ersten Schritt, um die Strahlen vorzubereiten.
    # Die Berechnungen umfassen die Umwandlung von Pixelkoordinaten in Richtungsvektoren, die von der Kamera ausstrahlen.
    # Dies ist ein typischer Schritt sowohl in der Rasterisierung als auch im Raytracing, da die Kameraansicht für die Berechnung der Beleuchtung und der Sichtbarkeit von Objekten in einer 3D-Szene verwendet wird.
    # Das genaue Rendering-Verfahren (Rasterisierung oder Raytracing) hängt von den weiteren Schritten im Code ab, die hier nicht gezeigt werden.

    pixel_center = 0.5 if self.use_pixel_centers else 0.0
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.w, dtype=np.float32) + pixel_center,  # X-Axis (columns)
        np.arange(self.h, dtype=np.float32) + pixel_center,  # Y-Axis (rows)
        indexing="xy")

    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    inverse_intrisics = np.linalg.inv(self.intrinsic_matrix[Ellipsis, :3, :3])
    camera_dirs = (inverse_intrisics[None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    directions = (self.camtoworlds[:, None, None, :3, :3]
                  @ camera_dirs[None, Ellipsis, None])[Ellipsis, 0]

    test = self.camtoworlds[:, None, None, :3, -1]

    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    self.rays = data_types.Rays(origins=origins, directions=viewdirs)

#Hier ist eine schrittweise Erläuterung des Codes:

#pixel_center = 0.5 if self.use_pixel_centers else 0.0: Hier wird eine Variable pixel_center festgelegt, die angibt, ob die Pixelzentren verwendet werden sollen oder nicht.
#Wenn self.use_pixel_centers wahr ist, wird pixel_center auf 0.5 gesetzt, andernfalls auf 0.0. Dies hängt von den Anforderungen der Anwendung ab und bestimmt, ob die Strahlen durch die Mitte der Pixel oder durch die Ecken der Pixel verlaufen.

#x, y = np.meshgrid(...): Hier werden Gitter für die x- und y-Koordinaten der Pixel im Bild erstellt. Dies geschieht mithilfe von np.meshgrid, wodurch 2D-Arrays x und y erzeugt werden, die die x- und y-Koordinaten der Pixel im Bild enthalten.

#pixels = np.stack((x, y, np.ones_like(x)), axis=-1): Hier werden die Pixelkoordinaten in ein 3D-Array pixels gestapelt, wobei jede Zeile des Arrays die x-, y- und z-Koordinaten eines Pixels im Raum darstellt.
#Die z-Koordinate wird auf 1 gesetzt, um die Verwendung von Homogenkoordinaten anzudeuten.

#inverse_intrisics = np.linalg.inv(self.intrinsic_matrix[Ellipsis, :3, :3]): Hier wird die inverse intrinsische Kameramatrix berechnet.
#Diese Matrix enthält Informationen über die intrinsischen Parameter der Kamera, einschließlich der Brennweite und der Verzerrung.
#Die inverse Matrix wird verwendet, um die Richtungen der Strahlen im Raum zu berechnen.

#camera_dirs = (inverse_intrisics[None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]: Hier werden die Richtungen der Strahlen berechnet, die von der Kamera ausgehen.
#Dies geschieht, indem die inverse intrinsische Matrix auf die Pixelkoordinaten angewendet wird.
#directions = ...: Hier werden die Richtungen der Strahlen von der Kamera zu den Pixeln transformiert, um die globalen Raumrichtungen zu erhalten.
#Dies erfolgt unter Verwendung der extrinsischen Pose-Matrix (self.camtoworlds) und der berechneten Kamerarichtungen.

#origins = ...: Hier werden die Ursprünge der Strahlen festgelegt. Diese entsprechen normalerweise der Position der Kamera im Raum.
#viewdirs = ...: Hier werden die Richtungen der Strahlen normalisiert, um sicherzustellen, dass sie Einheitsvektoren sind.

#Am Ende des Codes enthält die Variable self.rays eine Datenstruktur, die die generierten Strahlen (Origins und Directions) für alle Ansichten im Datensatz enthält.
#Diese Strahlen können dann in verschiedenen Computer-Vision-Anwendungen verwendet werden, z. B. zur Berechnung von Tiefenkarten oder zur 3D-Rekonstruktion.