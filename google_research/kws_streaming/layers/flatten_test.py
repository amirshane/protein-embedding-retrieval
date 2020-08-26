# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for kws_streaming.layers.flaten."""

import numpy as np
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
from kws_streaming.layers.modes import Modes
from kws_streaming.layers.stream import Stream
from kws_streaming.models import utils
tf1.disable_eager_execution()


class FlattenTest(tf.test.TestCase):

  def init(self, shape=(8, 2), flat_dim="time"):
    self.batch_size = 1
    # input data placeholder
    input_tf = tf.keras.layers.Input(
        shape=shape, batch_size=self.batch_size, name="inp1")

    # input test data
    self.inputs = np.random.uniform(size=(self.batch_size,) + shape)

    # create non streamable trainable model
    mode = Modes.TRAINING
    if flat_dim == "time":
      flat_tf = Stream(cell=tf.keras.layers.Flatten(), mode=mode)(input_tf)
    else:
      flat_tf = tf.reshape(
          input_tf,
          (-1, input_tf.shape[1], input_tf.shape[2] * input_tf.shape[3]))
    # flat_tf = flatten.Flatten(mode=mode, flat_dim=flat_dim)(input_tf)
    self.model_train = tf.keras.Model(input_tf, flat_tf)
    self.model_train.summary()

    # output data, generated by non streaming model
    self.outputs = self.model_train.predict(self.inputs)
    return self.outputs

  def test_streaming_inference_internal_state(self):
    # test on input with [batch, time, feature1, feature2]
    self.init(shape=(8, 2, 1))
    # convert non streamable trainable model to streamable with internal state
    mode = Modes.STREAM_INTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(
                1,
                2,
                1,
            ), batch_size=self.batch_size, name="inp1")
    ]
    model_stream = utils.convert_to_inference_model(self.model_train,
                                                    input_tensors, mode)
    # run streaming inference
    for i in range(self.inputs.shape[1]):
      input_stream = np.expand_dims(self.inputs[0][i], 0)
      input_stream = np.expand_dims(input_stream, 1)
      output_stream = model_stream.predict(input_stream)
    self.assertAllEqual(output_stream, self.outputs)

  def test_streaming_inference_external_state(self):
    # test on input with [batch, time, feature1]
    self.init(shape=(8, 2))
    # convert non streamable trainable model to streamable with external state
    mode = Modes.STREAM_EXTERNAL_STATE_INFERENCE
    input_tensors = [
        tf.keras.layers.Input(
            shape=(
                1,
                2,
            ), batch_size=self.batch_size, name="inp1")
    ]
    model_stream = utils.convert_to_inference_model(self.model_train,
                                                    input_tensors, mode)

    input_state = np.zeros(model_stream.inputs[1].shape, dtype=np.float32)

    # run streaming inference
    for i in range(self.inputs.shape[1]):
      input_stream = np.expand_dims(self.inputs[0][i], 0)
      input_stream = np.expand_dims(input_stream, 1)
      output_stream, output_state = model_stream.predict(
          [input_stream, input_state])
      input_state = output_state  # update input state
    self.assertAllEqual(output_stream, self.outputs)

  def test_dims_flattening(self):
    # test on input with [batch, time, feature1, feature2]
    shape = (8, 2, 3)
    # test flattening of all dims including time, batch is excluded
    flat_time_output = self.init(shape, "time")
    self.assertAllEqual(flat_time_output.shape,
                        [self.batch_size, shape[0] * shape[1] * shape[2]])

    # test flattening of all dims including feature, batch and time are excluded
    flat_feature_output = self.init(shape, "feature")
    self.assertAllEqual(flat_feature_output.shape,
                        [self.batch_size, 8, shape[1] * shape[2]])


if __name__ == "__main__":
  tf.test.main()
