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
"""Tests for inference_utils."""

import os

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import inference_utils


def tearDownModule():
  file_io.delete_recursively(test.get_temp_dir())


class _SimpleModel(module.Module):
  """A simple model for testing inference."""

  @def_function.function(
      input_signature=[
          tensor_spec.TensorSpec(shape=[None], dtype=dtypes.float32,
                                 name='x')
      ])
  def __call__(self, x):
    return {'output': x * 2.0}


def _save_test_model(saved_model_dir):
  """Saves a simple test SavedModel to `saved_model_dir`."""
  model = _SimpleModel()
  saved_model_save.save(
      model,
      saved_model_dir,
      signatures={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              model.__call__
      })


class LoadModelTest(test.TestCase):

  def testLoadModelReturnsObject(self):
    saved_model_dir = os.path.join(test.get_temp_dir(), 'load_model_test')
    _save_test_model(saved_model_dir)
    loaded = inference_utils.load_model(saved_model_dir)
    self.assertIsNotNone(loaded)
    self.assertTrue(hasattr(loaded, 'signatures'))

  def testLoadModelWithExplicitServingTag(self):
    saved_model_dir = os.path.join(
        test.get_temp_dir(), 'load_model_serving_tag')
    _save_test_model(saved_model_dir)
    loaded = inference_utils.load_model(
        saved_model_dir, tags={tag_constants.SERVING})
    self.assertIsNotNone(loaded)

  def testLoadModelInvalidDirectory(self):
    with self.assertRaises(Exception):
      inference_utils.load_model(
          os.path.join(test.get_temp_dir(), 'nonexistent_model'))


class GetServingSignatureTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_model_dir = os.path.join(
        test.get_temp_dir(), 'get_signature_test')
    _save_test_model(self._saved_model_dir)
    self._model = inference_utils.load_model(self._saved_model_dir)

  def testGetDefaultServingSignature(self):
    sig = inference_utils.get_serving_signature(self._model)
    self.assertIsNotNone(sig)

  def testGetServingSignatureInvalidKey(self):
    with self.assertRaisesRegex(ValueError, 'not found in model'):
      inference_utils.get_serving_signature(self._model, 'nonexistent_key')

  def testGetServingSignatureNoSignaturesAttr(self):
    class FakeModel:
      pass

    with self.assertRaisesRegex(ValueError, '`signatures` attribute'):
      inference_utils.get_serving_signature(FakeModel())


class RunInferenceTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_model_dir = os.path.join(
        test.get_temp_dir(), 'run_inference_test')
    _save_test_model(self._saved_model_dir)
    self._model = inference_utils.load_model(self._saved_model_dir)

  def testRunInferenceWithModel(self):
    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    results = inference_utils.run_inference(
        self._model, {'x': constant_op.constant(input_data)})
    self.assertIn('output', results)
    np.testing.assert_allclose(results['output'], input_data * 2.0)

  def testRunInferenceWithCallable(self):
    def double_fn(**kwargs):
      x = kwargs['x']
      import tensorflow as tf  # pylint: disable=g-import-not-at-top
      return {'output': tf.constant(x.numpy() * 2.0)}

    input_data = np.array([1.0, 2.0], dtype=np.float32)
    results = inference_utils.run_inference(
        double_fn, {'x': constant_op.constant(input_data)})
    self.assertIn('output', results)

  def testRunInferenceOutputsAreNumpy(self):
    input_data = np.array([5.0], dtype=np.float32)
    results = inference_utils.run_inference(
        self._model, {'x': constant_op.constant(input_data)})
    for value in results.values():
      self.assertIsInstance(value, np.ndarray)


if __name__ == '__main__':
  test.main()
