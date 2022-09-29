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
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.backend import random_bernoulli

from merlin.models.tf.core.base import Block, BlockType, PredictionOutput
from merlin.models.tf.core.combinators import TabularBlock
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.transforms.tensor import ListToRagged
from merlin.models.tf.typing import TabularData
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.schema_utils import tensorflow_metadata_json_to_schema
from merlin.schema import ColumnSchema, Schema, Tags


@Block.registry.register_with_multiple_names("remove_pad_3d")
@tf.keras.utils.register_keras_serializable(package="merlin_models")
class RemovePad3D(Block):
    """
    Flatten the sequence of labels and filter out non-targets positions

    Parameters
    ----------
        padding_idx: int
            The padding index value.
            Defaults to 0.
    Returns
    -------
        targets: tf.Tensor
            The flattened vector of true targets positions
        flatten_predictions: tf.Tensor
            If the predictions are 3-D vectors (sequential task),
            flatten the predictions vectors to keep only the ones related to target positions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding_idx = 0

    def compute_output_shape(self, input_shape):
        return input_shape

    def call_outputs(
        self, outputs: PredictionOutput, training=True, **kwargs
    ) -> "PredictionOutput":
        targets, predictions = outputs.targets, outputs.predictions
        targets = tf.reshape(targets, (-1,))
        non_pad_mask = targets != self.padding_idx
        targets = tf.boolean_mask(targets, non_pad_mask)

        assert isinstance(predictions, tf.Tensor), "Predictions must be a tensor"

        if len(tuple(predictions.get_shape())) == 3:
            predictions = tf.reshape(predictions, (-1, predictions.shape[-1]))
            predictions = tf.boolean_mask(
                predictions, tf.broadcast_to(tf.expand_dims(non_pad_mask, 1), tf.shape(predictions))
            )

        return outputs.copy_with_updates(predictions=predictions, targets=targets)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class SequenceTransform(TabularBlock):
    """Prepares sequential inputs and targets for next-item prediction.
    The target is extracted from the shifted sequence of item ids and
    the sequential input features are truncated in the last position.
    Parameters
    ----------
    schema : Schema
        The schema with the sequential columns to be truncated
    target : Union[str, Tags, ColumnSchema]
        The sequential input column that will be used to extract the target
    pre : Optional[BlockType], optional
        A block that is called before this method call().
        If not set, the ListToRagged() block is applied to convert
        the tuple representation of sequential features to RaggedTensors,
        so that the tensors sequences can be shifted/truncated
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        pre: Optional[BlockType] = None,
        **kwargs,
    ):
        _pre = ListToRagged()
        if pre:
            _pre = _pre.connect(pre)
        super().__init__(pre=_pre, schema=schema, **kwargs)

        self.target = target
        self.target_name = self._get_target(target)

    def _get_target(self, target):
        if (
            (isinstance(target, str) and target not in self.schema.column_names)
            or (isinstance(target, Tags) and len(self.schema.select_by_tag(target)) > 0)
            or (isinstance(target, ColumnSchema) and target not in self.schema)
        ):
            raise ValueError("The target column needs to be part of the sequential schema")

        target_name = target
        if isinstance(target, ColumnSchema):
            target_name = target.name
        if isinstance(target, Tags):
            if len(self.schema.select_by_tag(target)) > 1:
                raise ValueError(
                    "Only 1 column should the Tag ({target}) provided for target, but"
                    f"the following columns have that tag: "
                    f"{self.schema.select_by_tag(target).column_names}"
                )
            target_name = self.schema.select_by_tag(target).column_names[0]
        return target_name

    def call(
        self, inputs: TabularData, targets=None, training=False, testing=False, **kwargs
    ) -> Union[Prediction, Tuple]:
        raise NotImplementedError()

    def _check_seq_inputs_targets(self, inputs: TabularData):
        target_shape = inputs[self.target_name].get_shape().as_list()
        if len(target_shape) != 2:
            raise ValueError(
                f"The target column ({self.target_name}) is expected to be a 2D tensor,"
                f" but the shape is {target_shape}"
            )
        if target_shape[-1] == 1:
            raise ValueError(
                "The 2nd dim of the target column ({self.target_name}) should be greater"
                " than 1, so that the sequential input can be shifted as target"
            )

        seq_inputs_shapes = {
            col: inputs[col].get_shape().as_list() for col in self.schema.column_names
        }

        seq_shapes = list(seq_inputs_shapes.values())
        if not all(x == seq_shapes[0] for x in seq_shapes):
            raise ValueError(
                "The sequential inputs must have the same shape, but the shapes"
                f"are different: {seq_inputs_shapes}"
            )

    def compute_output_shape(self, input_shape):
        new_input_shapes = dict()
        for k, v in input_shape.items():
            new_input_shapes[k] = v
            if k in self.schema.column_names:
                # If it is a list/sparse feature (in tuple representation), uses the offset as shape
                if isinstance(v, tuple) and isinstance(v[1], tf.TensorShape):
                    new_input_shapes[k] = tf.TensorShape([v[1][0], None])
                else:
                    # Reducing 1 position of the seq length
                    new_input_shapes[k] = tf.TensorShape([v[0], v[1] - 1])

        return new_input_shapes

    def get_config(self):
        """Returns the config of the layer as a Python dictionary."""
        config = super().get_config()
        config["target"] = self.target

        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config. Returning the instance."""
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        config["schema"] = tensorflow_metadata_json_to_schema(config["schema"])
        schema = config.pop("schema")
        target = config.pop("target")
        return cls(schema, target, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PredictMasked(SequenceTransform):
    """This block implements the Masked Language Modeling (MLM) training approach
    introduced in BERT (NLP) and later adapted to RecSys by BERT4Rec [1].
    Given an input tf.RaggedTensor with sequences of embeddings
    and the corresponding sequence of item ids, some positions are randomly selected (masked)
    to be the targets for prediction.
    The input embeddings at the masked positions are
    replaced by a common trainable embedding, to avoid leakage of the targets information.
    The targets are output being the same as the input id sequence. The masked targets
    can be discovered later by checking the outputs._keras_mask.
    This transformation is only applied during training, as during inference you want
    to use all available information of the sequence for prediction.

    References
    ----------
    .. [1] Sun, Fei, et al. "BERT4Rec: Sequential recommendation with bidirectional encoder
           representations from transformer." Proceedings of the 28th ACM international
           conference on information and knowledge management. 2019.

    Parameters
    ----------
    schema : Schema
        The input schema, that will be used to discover the name
        of the item id column
    masking_prob : float, optional
        Probability of an item to be selected (masked) as a label of the given sequence.
        Note: We enforce that at least one item is masked for each sequence, so that it
        is useful for training, by default 0.2
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[str, Tags, ColumnSchema],
        masking_prob: float = 0.2,
        **kwargs,
    ):
        self.masking_prob = masking_prob
        super().__init__(schema, target, **kwargs)

    def call(self, inputs: TabularBlock, targets=None) -> Prediction:
        """Masks some items from the input sequence features to be the targets
        and output the input sequence features with replaced embeddings for masked
        elements and also the targets (copy of the items ids sequence)

        Parameters
        ----------
        inputs : TabularBlock
            A dict with the input features
        training : bool, optional
            A flag indicating whether model is being trained or not, by default False

        Returns
        -------
        Prediction
            Returns a Prediction(replaced_inputs, targets)
        """
        self._check_seq_inputs_targets(inputs)

        if self.target_name not in self.target_name:
            raise ValueError(
                f"The inputs provided does contain the target column ({self.target_name})"
            )

        new_target = tf.identity(inputs[self.target_name])
        if targets is None:
            targets = new_target
        else:
            if not isinstance(targets, dict):
                targets = dict()
            targets[self.target_name] = new_target

        return (inputs, targets)

    def compute_mask(self, inputs, mask=None):
        """Is called by Keras and returns the targets mask that will
        be assigned to the input tensors and targets, being accessible
        by inputs._keras_mask
        """
        item_id_seq = inputs[self.target_name]
        self.target_mask = self._generate_target_mask(item_id_seq)

        inputs_mask = dict()
        for k, v in inputs.items():
            if k in self.schema.column_names:
                inputs_mask[k] = tf.logical_not(self.target_mask)
            else:
                inputs_mask[k] = None

        return (inputs_mask, self.target_mask)

    def _generate_target_mask(self, ids_seq: tf.RaggedTensor):
        row_lengths = ids_seq.row_lengths(1)

        assertion_min_seq_length = tf.Assert(tf.reduce_all(row_lengths > 1), [row_lengths])

        with tf.control_dependencies([assertion_min_seq_length]):
            # Targets are masked according to a probability
            target_mask_by_prob = self._get_masked_by_prob(row_lengths, prob=self.masking_prob)
            # Exactly one target is masked per row
            one_target_mask = self._get_one_masked(row_lengths)

            # For sequences (rows) with either all or none elements sampled (masked) as targets
            # as those sequences would be invalid for training
            # the row mask is replaced by a row mask that contains exactly one masked target
            replacement_cond = tf.logical_or(
                tf.logical_not(tf.reduce_any(target_mask_by_prob, axis=1)),
                tf.reduce_all(target_mask_by_prob, axis=1),
            )
            target_mask = tf.where(
                tf.expand_dims(replacement_cond, -1), one_target_mask, target_mask_by_prob
            )
            padding_mask = tf.sequence_mask(row_lengths)
            target_mask_ragged = tf.ragged.boolean_mask(target_mask, padding_mask)

            return target_mask_ragged

    @staticmethod
    def _get_masked_by_prob(row_lengths: tf.Tensor, prob: float) -> tf.Tensor:
        """Generates a dense mask boolean tensor with True values
        for randomly selected targets
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        batch_size = tf.shape(row_lengths)[0]
        output = tf.cast(random_bernoulli([batch_size, max_seq_length], p=prob), tf.bool)
        padding_mask = tf.sequence_mask(row_lengths)
        # Ignoring masked items in the padding positions
        output = tf.logical_and(output, padding_mask)
        return output

    @staticmethod
    def _get_one_masked(row_lengths: tf.Tensor):
        """Generates a dense mask boolean tensor where for each tensor (row)
        there is exactly one True value (selected target)
        """
        max_seq_length = tf.cast(tf.reduce_max(row_lengths), tf.int32)
        random_targets_indices = tf.cast(
            tf.math.floor(
                (
                    tf.random.uniform(
                        shape=tf.shape(row_lengths), minval=0.0, maxval=1.0, dtype=tf.float32
                    )
                    * tf.cast(row_lengths, tf.float32)
                )
            ),
            tf.int32,
        )

        one_target_mask = tf.cast(tf.one_hot(random_targets_indices, max_seq_length), tf.bool)
        return one_target_mask

    def get_config(self):
        config = super().get_config()
        config["masking_prob"] = self.masking_prob
        return config

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        schema = schema_utils.tensorflow_metadata_json_to_schema(config.pop("schema"))
        target = config.pop("target")
        masking_prob = config.pop("masking_prob")
        return cls(schema, target, masking_prob, **config)
