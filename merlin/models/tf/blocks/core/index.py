#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
#
from typing import Optional

import merlin.io
import tensorflow as tf

from merlin.models.tf.core import Block

"""
This would be useful for instance to convert the item-tower.
We could integrate this into the Block-class.

two_tower_block = ...
topk_index = TopKIndex.from_block(two_tower_block.item_block(), item_dataset)

recommender = two_tower_block.query_block().connect(topk_index)



"""


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class IndexBlock(Block):
    def __init__(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None, **kwargs):
        super(IndexBlock, self).__init__(**kwargs)
        self.values = values
        self.ids = ids

    @classmethod
    def from_dataset(cls, data: merlin.io.Dataset, id_column: str, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_block(cls, block: Block, data: merlin.io.Dataset, id_column: str, **kwargs):
        raise NotImplementedError()

    def update(self, values: tf.Tensor, ids: Optional[tf.Tensor] = None):
        if len(tf.shape(values)) != 2:
            raise ValueError(f"The candidates embeddings tensor must be 2D (got {values.shape}).")
        if not ids:
            ids = tf.range(values.shape[0])

        self.ids.assign(ids)
        self.values.assign(values)
        return self

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.values[inputs]

    def to_dataset(self, **kwargs) -> merlin.io.Dataset:
        raise NotImplementedError()


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class TopKIndex(IndexBlock):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.matmul(inputs, self.values, transpose_b=True)