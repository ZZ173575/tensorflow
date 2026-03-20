# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for running inference with TensorFlow SavedModels.

This module provides a simple interface for loading TensorFlow models and
running inference.

Example usage:

  model = load_model('/path/to/saved_model')
  results = run_inference(model, {'input': input_data})
"""

from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


def load_model(model_dir, tags=None):
  """Loads a TensorFlow SavedModel for inference.

  Args:
    model_dir: Path to the directory containing the SavedModel.
    tags: Optional set of tags identifying the MetaGraphDef to load.
      Defaults to {tag_constants.SERVING}.

  Returns:
    A trackable object representing the loaded SavedModel.

  Raises:
    IOError: If the SavedModel cannot be found or loaded.
  """
  if tags is None:
    tags = {tag_constants.SERVING}
  return saved_model_load.load(model_dir, tags=tags)


def get_serving_signature(model, signature_key=None):
  """Returns the serving signature function from a loaded SavedModel.

  Args:
    model: A loaded SavedModel object as returned by `load_model`.
    signature_key: Optional key identifying the signature to use.
      Defaults to DEFAULT_SERVING_SIGNATURE_DEF_KEY.

  Returns:
    The concrete function corresponding to the requested signature.

  Raises:
    ValueError: If the requested signature key is not found in the model.
  """
  if signature_key is None:
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

  if not hasattr(model, 'signatures'):
    raise ValueError('The loaded model does not have a `signatures` attribute. '
                     'Ensure the model was saved with signatures.')

  if signature_key not in model.signatures:
    available = list(model.signatures.keys())
    raise ValueError(
        f'Signature key "{signature_key}" not found in model. '
        f'Available signatures: {available}')

  return model.signatures[signature_key]


def run_inference(model, inputs, signature_key=None):
  """Runs inference using a loaded TensorFlow SavedModel.

  Args:
    model: A loaded SavedModel object as returned by `load_model`, or a
      callable (e.g. a ConcreteFunction) that accepts keyword arguments.
    inputs: A dict mapping input tensor names to numpy arrays or tensors.
    signature_key: Optional signature key to use when `model` is a SavedModel
      object with multiple signatures.  Defaults to
      DEFAULT_SERVING_SIGNATURE_DEF_KEY.  Ignored when `model` is already
      callable.

  Returns:
    A dict mapping output tensor names to numpy arrays.

  Raises:
    ValueError: If a serving signature cannot be found.
    TypeError: If `model` is not callable and does not have `signatures`.
  """
  if callable(model) and not hasattr(model, 'signatures'):
    infer_fn = model
  else:
    infer_fn = get_serving_signature(model, signature_key)

  outputs = infer_fn(**inputs)
  return {key: value.numpy() for key, value in outputs.items()}
