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

"""File utilities."""
import os
from os import path

INTERNAL = False


def open_file(pth, mode="rb"):

  """
  Open a file at the specified path.

    Args:
        pth (str): Path to the file to be opened.
        mode (str, optional): Mode in which the file should be opened. Defaults to "rb".

    Returns:
        file: An open file object.
  """

  return open(pth, mode=mode)


def file_exists(pth):

  """
  Check if a file exists at the specified path.

    Args:
        pth (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
  """
  return path.exists(pth)


def listdir(pth):
  """
  List the contents of a directory.

    Args:
        pth (str): Path to the directory.

    Returns:
        list: List of file and subdirectory names in the directory.

  """
  return os.listdir(pth)


def isdir(pth):
  """
  Check if a given path is a directory.

    Args:
        pth (str): Path to check.

    Returns:
        bool: True if the path is a directory, False otherwise.

  """
  return path.isdir(pth)


def makedirs(pth):
  """
  Create a directory and any missing parent directories.

    Args:
        pth (str): Path to the directory to be created.
  """
  os.makedirs(pth)
