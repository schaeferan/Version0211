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

"""Helper functions/classes for model definition.

Some functions adopted from
https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py
"""

import chex
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np
from PIL import Image

from gen_patch_neural_rendering.src.utils import file_utils


@jit
def safe_divide(x, y):
  """Divide two arrays element-wise, handling division by zero gracefully.

      Args:
          x (jnp.ndarray): Numerator array.
          y (jnp.ndarray): Denominator array.

      Returns:
          jnp.ndarray: Element-wise division result, with zeros where division by zero occurred.
  """
  return jnp.where(y != 0, x / y, 0)


def skew(v):
  """
  Compute the skew-symmetric matrix of a vector.

     Args:
         v (jnp.ndarray): Input vector.

     Returns:
         jnp.ndarray: Skew-symmetric matrix of the input vector.
  """
  return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def invert_camera(pose, use_inverse=False):
  """Invert a camera pose matrix.

      Args:
          pose (np.ndarray): Camera pose matrix.
          use_inverse (bool): If True, compute the inverse of the rotation matrix; otherwise, transpose it.

      Returns:
          np.ndarray: Inverted camera pose matrix.
  """
  r, t = pose[Ellipsis, :3, :3], pose[Ellipsis, :3, -1:]
  r_inv = np.linalg.inv(r) if use_inverse else np.einsum("bxy->byx", r)
  t_inv = -(r_inv @ t)
  return np.concatenate([r_inv, t_inv], axis=-1)


def uint2float(img):
  """Convert an unsigned integer image array to floating-point format.

      Args:
          img (jnp.ndarray): Unsigned integer image array.

      Returns:
          jnp.ndarray: Floating-point image array scaled to [0, 1].
  """
  chex.assert_type(img, jnp.uint8)
  return img.astype(jnp.float32) / 255.


def pad_image(images, stride):
  """Pad an image to ensure its dimensions are multiples of a specified stride.

      Args:
          images (jnp.ndarray): Image data with shape (batch_size, height, width, 3).
          stride (int): Desired stride for dimensions.

      Returns:
          jnp.ndarray: Padded image data, along with the amount of padding applied to the upper sides of the image (uh) and the left sides (uw).
  """

  h, w = images.shape[1:3]  # images have shape (bs, h, w, 3)

  if h % stride > 0:
    new_h = h + stride - h % stride
  else:
    new_h = h

  if w % stride > 0:
    new_w = w + stride - w % stride
  else:
    new_w = w

  lh, uh = 0, int(new_h - h)
  lw, uw = 0, int(new_w - w)

  out = jnp.pad(images, ((0, 0), (lh, uh), (lw, uw), (0, 0)), mode="edge")

  return out, uh, uw


def posenc(x, min_deg, max_deg):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                   list(x.shape[:-1]) + [-1])
  four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)


def add_gaussian_noise(key, raw, noise_std, randomized):
  """Adds gaussian noise to `raw`, which can used to regularize it.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), arbitrary shape.
    noise_std: float, The standard deviation of the noise to be added.
    randomized: bool, add noise if randomized is True.

  Returns:
    raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
  """
  if (noise_std is not None) and randomized:
    return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
  else:
    return raw


def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10. * jnp.log(mse) / jnp.log(10.)


def save_img(img, pth):
  """Save an image to disk.

  Args:
    img: jnp.ndarry, [height, width, channels], img will be clipped to [0, 1]
      before saved to pth.
    pth: string, path to save the image to.
  """
  with file_utils.open_file(pth, "wb") as imgout:
    Image.fromarray(np.array(
        (np.clip(img, 0., 1.) * 255.).astype(jnp.uint8))).save(imgout, "PNG")
