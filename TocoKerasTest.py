# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lite.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
import tempfile
import numpy as np

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.lite.python import lite_constants
from tensorflow.contrib.lite.python.interpreter import Interpreter
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph


class FromKerasFile(test_util.TensorFlowTestCase):

    def setUp(self):
        keras.backend.clear_session()

    def _getSequentialModel(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
        model.compile(
            loss=keras.losses.MSE,
            optimizer=keras.optimizers.RMSprop(),
            metrics=[keras.metrics.categorical_accuracy],
            sample_weight_mode='temporal')
        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 3))
        model.train_on_batch(x, y)
        model.predict(x)

        try:
            fd, keras_file = tempfile.mkstemp('.h5')
            keras.models.save_model(model, keras_file)
        finally:
            os.close(fd)
        return keras_file

    def testSequentialModel(self):
        """Test a Sequential tf.keras model with default inputs."""
        keras_file = self._getSequentialModel()

        converter = lite.TocoConverter.from_keras_model_file(keras_file)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

        os.remove(keras_file)

        # Check values from converted model.
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(1, len(input_details))
        self.assertEqual('dense_input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertTrue(([1, 3] == input_details[0]['shape']).all())
        self.assertEqual((0., 0.), input_details[0]['quantization'])

        output_details = interpreter.get_output_details()
        self.assertEqual(1, len(output_details))
        self.assertEqual('time_distributed/Reshape_1',
                         output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
        self.assertEqual((0., 0.), output_details[0]['quantization'])

    def testSequentialModelInputArray(self):
        """Test a Sequential tf.keras model testing input arrays argument."""
        keras_file = self._getSequentialModel()

        # Invalid input array raises error.
        with self.assertRaises(ValueError) as error:
            lite.TocoConverter.from_keras_model_file(
                keras_file, input_arrays=['invalid-input'])
        self.assertEqual("Invalid tensors 'invalid-input' were found.",
                         str(error.exception))

        # Valid input array.
        converter = lite.TocoConverter.from_keras_model_file(
            keras_file, input_arrays=['dense_input'])
        tflite_model = converter.convert()
        os.remove(keras_file)
        self.assertTrue(tflite_model)

    def testSequentialModelInputShape(self):
        """Test a Sequential tf.keras model testing input shapes argument."""
        keras_file = self._getSequentialModel()

        # Passing in shape of invalid input array has no impact as long as all input
        # arrays have a shape.
        converter = lite.TocoConverter.from_keras_model_file(
            keras_file, input_shapes={'invalid-input': [2, 3]})
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

        # Passing in shape of valid input array.
        converter = lite.TocoConverter.from_keras_model_file(
            keras_file, input_shapes={'dense_input': [2, 3]})
        tflite_model = converter.convert()
        os.remove(keras_file)
        self.assertTrue(tflite_model)

        # Check input shape from converted model.
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(1, len(input_details))
        self.assertEqual('dense_input', input_details[0]['name'])
        self.assertTrue(([2, 3] == input_details[0]['shape']).all())

    def testSequentialModelOutputArray(self):
        """Test a Sequential tf.keras model testing output arrays argument."""
        keras_file = self._getSequentialModel()

        # Invalid output array raises error.
        with self.assertRaises(ValueError) as error:
            lite.TocoConverter.from_keras_model_file(
                keras_file, output_arrays=['invalid-output'])
        self.assertEqual("Invalid tensors 'invalid-output' were found.",
                         str(error.exception))

        # Valid output array.
        converter = lite.TocoConverter.from_keras_model_file(
            keras_file, output_arrays=['time_distributed/Reshape_1'])
        tflite_model = converter.convert()
        os.remove(keras_file)
        self.assertTrue(tflite_model)

    def testFunctionalModel(self):
        """Test a Functional tf.keras model with default inputs."""
        inputs = keras.layers.Input(shape=(3,), name='input')
        x = keras.layers.Dense(2)(inputs)
        output = keras.layers.Dense(3)(x)

        model = keras.models.Model(inputs, output)
        model.compile(
            loss=keras.losses.MSE,
            optimizer=keras.optimizers.RMSprop(),
            metrics=[keras.metrics.categorical_accuracy])
        x = np.random.random((1, 3))
        y = np.random.random((1, 3))
        model.train_on_batch(x, y)

        model.predict(x)
        fd, keras_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, keras_file)

        # Convert to TFLite model.
        converter = lite.TocoConverter.from_keras_model_file(keras_file)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

        os.close(fd)
        os.remove(keras_file)

        # Check values from converted model.
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(1, len(input_details))
        self.assertEqual('input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertTrue(([1, 3] == input_details[0]['shape']).all())
        self.assertEqual((0., 0.), input_details[0]['quantization'])

        output_details = interpreter.get_output_details()
        self.assertEqual(1, len(output_details))
        self.assertEqual('dense_1/BiasAdd', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertTrue(([1, 3] == output_details[0]['shape']).all())
        self.assertEqual((0., 0.), output_details[0]['quantization'])

    def testFunctionalModelMultipleInputs(self):
        """Test a Functional tf.keras model with multiple inputs and outputs."""
        a = keras.layers.Input(shape=(3,), name='input_a')
        b = keras.layers.Input(shape=(3,), name='input_b')
        dense = keras.layers.Dense(4, name='dense')
        c = dense(a)
        d = dense(b)
        e = keras.layers.Dropout(0.5, name='dropout')(c)

        model = keras.models.Model([a, b], [d, e])
        model.compile(
            loss=keras.losses.MSE,
            optimizer=keras.optimizers.RMSprop(),
            metrics=[keras.metrics.mae],
            loss_weights=[1., 0.5])

        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))
        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))
        model.train_on_batch([input_a_np, input_b_np], [
                             output_d_np, output_e_np])

        model.predict([input_a_np, input_b_np], batch_size=5)
        fd, keras_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, keras_file)

        # Convert to TFLite model.
        converter = lite.TocoConverter.from_keras_model_file(keras_file)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

        os.close(fd)
        os.remove(keras_file)

        # Check values from converted model.
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(2, len(input_details))
        self.assertEqual('input_a', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertTrue(([1, 3] == input_details[0]['shape']).all())
        self.assertEqual((0., 0.), input_details[0]['quantization'])

        self.assertEqual('input_b', input_details[1]['name'])
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertTrue(([1, 3] == input_details[1]['shape']).all())
        self.assertEqual((0., 0.), input_details[1]['quantization'])

        output_details = interpreter.get_output_details()
        self.assertEqual(2, len(output_details))
        self.assertEqual('dense_1/BiasAdd', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertTrue(([1, 4] == output_details[0]['shape']).all())
        self.assertEqual((0., 0.), output_details[0]['quantization'])

        self.assertEqual('dropout/Identity', output_details[1]['name'])
        self.assertEqual(np.float32, output_details[1]['dtype'])
        self.assertTrue(([1, 4] == output_details[1]['shape']).all())
        self.assertEqual((0., 0.), output_details[1]['quantization'])

    def testFunctionalSequentialModel(self):
        """Test a Functional tf.keras model containing a Sequential model."""
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
        model = keras.models.Model(model.input, model.output)

        model.compile(
            loss=keras.losses.MSE,
            optimizer=keras.optimizers.RMSprop(),
            metrics=[keras.metrics.categorical_accuracy],
            sample_weight_mode='temporal')
        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 3))
        model.train_on_batch(x, y)
        model.predict(x)

        model.predict(x)
        fd, keras_file = tempfile.mkstemp('.h5')
        keras.models.save_model(model, keras_file)

        # Convert to TFLite model.
        converter = lite.TocoConverter.from_keras_model_file(keras_file)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

        os.close(fd)
        os.remove(keras_file)

        # Check values from converted model.
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        self.assertEqual(1, len(input_details))
        self.assertEqual('dense_input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertTrue(([1, 3] == input_details[0]['shape']).all())
        self.assertEqual((0., 0.), input_details[0]['quantization'])

        output_details = interpreter.get_output_details()
        self.assertEqual(1, len(output_details))
        self.assertEqual('time_distributed/Reshape_1',
                         output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertTrue(([1, 3, 3] == output_details[0]['shape']).all())
        self.assertEqual((0., 0.), output_details[0]['quantization'])


if __name__ == '__main__':
    test.main()
