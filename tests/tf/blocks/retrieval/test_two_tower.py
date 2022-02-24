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
import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_two_tower_block(testing_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(testing_data.schema, query_tower=ml.MLPBlock([64, 128]))
    outputs = two_tower(testing_data.tf_tensor_dict)

    assert len(outputs) == 2
    for key in ["item", "query"]:
        assert list(outputs[key].shape) == [100, 128]


def test_two_tower_block_tower_save(testing_data: SyntheticData, tmp_path):
    two_tower = ml.TwoTowerBlock(testing_data.schema, query_tower=ml.MLPBlock([64, 128]))
    two_tower(testing_data.tf_tensor_dict)

    query_tower = two_tower.query_block()
    query_tower.save(str(tmp_path / "query_tower"))
    query_tower_copy = tf.keras.models.load_model(str(tmp_path / "query_tower"))
    weights = zip(query_tower.get_weights(), query_tower_copy.get_weights())
    assert all([np.array_equal(w1, w2) for w1, w2 in weights])

    item_tower = two_tower.item_block()
    item_tower.save(str(tmp_path / "item_tower"))
    item_tower_copy = tf.keras.models.load_model(str(tmp_path / "item_tower"))
    weights = zip(item_tower.get_weights(), item_tower_copy.get_weights())
    assert all([np.array_equal(w1, w2) for w1, w2 in weights])


def test_two_tower_block_serialization(testing_data: SyntheticData):
    two_tower = ml.TwoTowerBlock(testing_data.schema, query_tower=ml.MLPBlock([64, 128]))
    copy_two_tower = testing_utils.assert_serialization(two_tower)

    outputs = copy_two_tower(testing_data.tf_tensor_dict)

    assert len(outputs) == 2
    for key in ["item", "query"]:
        assert list(outputs[key].shape) == [100, 128]


# TODO: Fix this test
# def test_two_tower_block_saving(ecommerce_data: SyntheticData):
#     two_tower = ml.TwoTowerBlock(ecommerce_data.schema, query_tower=ml.MLPBlock([64, 128]))
#
#     model = two_tower.connect(
#         ml.ItemRetrievalTask(ecommerce_data.schema, target_name="click", metrics=[])
#     )
#
#     dataset = ecommerce_data.tf_dataloader(batch_size=50)
#     copy_two_tower = testing_utils.assert_model_is_retrainable(model, dataset)
#
#     outputs = copy_two_tower(ecommerce_data.tf_tensor_dict)
#     assert list(outputs.shape) == [100, 1]


def test_two_tower_block_no_item_features(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        schema = testing_data.schema.remove_by_tag(Tags.ITEM)
        ml.TwoTowerBlock(schema, query_tower=ml.MLPBlock([64]))
        assert "The schema should contain features with the tag `item`" in str(excinfo.value)


def test_two_tower_block_no_user_features(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        schema = testing_data.schema.remove_by_tag(Tags.USER)
        ml.TwoTowerBlock(schema, query_tower=ml.MLPBlock([64]))
        assert "The schema should contain features with the tag `user`" in str(excinfo.value)


def test_two_tower_block_no_schema():
    with pytest.raises(ValueError) as excinfo:
        ml.TwoTowerBlock(schema=None, query_tower=ml.MLPBlock([64]))
    assert "The schema is required by TwoTower" in str(excinfo.value)


def test_two_tower_block_no_bottom_block(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        ml.TwoTowerBlock(schema=testing_data.schema, query_tower=None)
    assert "The query_tower is required by TwoTower" in str(excinfo.value)