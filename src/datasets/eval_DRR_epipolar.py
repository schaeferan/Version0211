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


class Eval_DRR_Epipolar(FFEpipolar):
  """Forward Facing epipolar dataset for medical xray images."""




