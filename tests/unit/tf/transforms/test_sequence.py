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

import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.loader import Loader
from merlin.schema import Tags


@pytest.mark.parametrize("use_loader", [False, True])
def test_seq_predict_masked(sequence_testing_data: Dataset, use_loader: bool):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_masked = mm.PredictMasked(schema=seq_schema, target=target, masking_prob=0.3)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    if use_loader:
        dataset_transformed = Loader(
            sequence_testing_data, batch_size=8, shuffle=False, transform=predict_masked
        )
        output = next(iter(dataset_transformed))
    else:
        output = predict_masked(batch)
    output_x, output_y = output

    target_mask = output_y._keras_mask

    # Checking if there is no sequence with no elements masked as target
    tf.assert_equal(
        tf.reduce_all(tf.reduce_any(target_mask, axis=1)),
        True,
        message=f"There are sequences with no targets masked {target_mask.numpy()}",
    )
    # Checking if there is no sequence with all elements masked
    tf.assert_equal(
        tf.reduce_any(tf.reduce_all(target_mask, axis=1)),
        False,
        message=f"There are sequences with all targets masked {target_mask.numpy()}",
    )

    as_ragged = mm.ListToRagged()
    batch = as_ragged(batch)

    for k, v in batch.items():
        # Checking if inputs values didn't change
        tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

        # Checks if for sequential input columns the mask has been assigned
        # (opposite of the target mask)
        if k in seq_schema.column_names:
            tf.Assert(
                tf.reduce_all(output_x[k]._keras_mask == tf.logical_not(target_mask)),
                [],
            )


def test_seq_predict_masked_after_embedings(
    sequence_testing_data: Dataset, use_loader: bool = False
):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)

    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_masked = mm.PredictMasked(schema=seq_schema, target=target, masking_prob=0.3)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    if use_loader:
        dataset_transformed = Loader(
            sequence_testing_data, batch_size=8, shuffle=False, transform=predict_masked
        )
        output = next(iter(dataset_transformed))
    else:
        output = predict_masked(batch)
    output_x, output_y = output

    embedding = tf.keras.layers.Embedding(100, 5, mask_zero=True)
    embedded_item_ids = embedding(output_x["item_id_seq"])

    tf.Assert(
        tf.reduce_all(embedded_item_ids._keras_mask == output_x["item_id_seq"]._keras_mask), []
    )
