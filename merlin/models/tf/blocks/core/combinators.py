import copy
import sys
from functools import reduce
from typing import Dict, List, Optional, Union

import six
import tensorflow as tf

from merlin.models.tf.blocks.core.base import (
    Block,
    BlockType,
    NoOp,
    PredictionOutput,
    is_input_block,
    right_shift_layer,
)
from merlin.models.tf.blocks.core.tabular import (
    AsTabular,
    Filter,
    TabularAggregationType,
    TabularBlock,
)
from merlin.models.tf.blocks.core.transformations import AsDenseFeatures
from merlin.models.tf.utils import tf_utils
from merlin.models.utils import schema_utils
from merlin.models.utils.misc_utils import filter_kwargs
from merlin.schema import Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class SequentialBlock(Block):
    """The SequentialLayer represents a sequence of Keras layers.
    It is a Keras Layer that can be used instead of tf.keras.layers.Sequential,
    which is actually a Keras Model.  In contrast to keras Sequential, this
    layer can be used as a pure Layer in tf.functions and when exporting
    SavedModels, without having to pre-declare input and output shapes.  In turn,
    this layer is usable as a preprocessing layer for TF Agents Networks, and
    can be exported via PolicySaver.
    Usage::

        c = SequentialLayer([layer1, layer2, layer3])
        output = c(inputs)    # Equivalent to: output = layer3(layer2(layer1(inputs)))
    """

    def __init__(
        self,
        *layers,
        filter: Optional[Union[Schema, Tags, List[str], "Filter"]] = None,
        pre_aggregation: Optional["TabularAggregationType"] = None,
        block_name: Optional[str] = None,
        copy_layers: bool = False,
        **kwargs,
    ):
        """Create a composition.

        Parameters
        ----------
        layers:
            A list or tuple of layers to compose.
        **kwargs:
            Arguments to pass to `Keras` layer initializer, including `name`.

        Raises
        ------
        TypeError:
            If any of the layers are not instances of keras `Layer`.
        """
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]  # type: ignore

        self.block_name = block_name

        if pre_aggregation:
            layers = [TabularBlock(aggregation=pre_aggregation), *layers]  # type: ignore

        for layer in layers:
            if not isinstance(layer, tf.keras.layers.Layer):
                raise TypeError(
                    "Expected all layers to be instances of keras Layer, but saw: '{}'".format(
                        layer
                    )
                )

        super(SequentialBlock, self).__init__(**kwargs)

        if getattr(layers[0], "has_schema", None):
            super().set_schema(layers[0].schema)

        layers = copy.copy(layers) if copy_layers else layers
        if filter:
            if not isinstance(filter, Filter):
                filter = Filter(filter)
            self.layers = [filter, *layers]
        else:
            self.layers = list(layers)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = input_signature
        for layer in self.layers:
            output_signature = layer.compute_output_signature(output_signature)
        return output_signature

    def build(self, input_shape=None):
        self._maybe_propagate_context(input_shape)
        last_layer = None
        for layer in self.layers:
            try:
                layer.build(input_shape)
            except TypeError:
                t, v, tb = sys.exc_info()
                if isinstance(input_shape, dict) and isinstance(last_layer, TabularBlock):
                    v = TypeError(
                        f"Couldn't build {layer}, "
                        f"did you forget to add aggregation to {last_layer}?"
                    )
                six.reraise(t, v, tb)
            input_shape = layer.compute_output_shape(input_shape)
            last_layer = layer
        self.built = True

    def set_schema(self, schema=None):
        for layer in self.layers:
            self._maybe_set_schema(layer, schema)

        return super().set_schema(schema)

    def _get_name(self):
        return self.block_name if self.block_name else f"{self.__class__.__name__}"

    @property
    def inputs(self):
        first = list(self)[0]
        if isinstance(first, SequentialBlock):
            return first.inputs
        if is_input_block(first):
            return first

    @property
    def first(self):
        return self.layers[0]

    @property
    def last(self):
        return self.layers[-1]

    @property
    def filter_features(self) -> List[str]:
        if isinstance(self.layers[0], Filter):
            return self.layers[0].feature_names
        elif isinstance(self.layers[0], SequentialBlock):
            return self.layers[0].filter_features

        return []

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = {}
        for layer in self.layers:
            for v in layer.trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def non_trainable_weights(self):
        weights = {}
        for layer in self.layers:
            for v in layer.non_trainable_weights:
                weights[id(v)] = v
        return list(weights.values())

    @property
    def trainable(self):
        return all(layer.trainable for layer in self.layers)

    @trainable.setter
    def trainable(self, value):
        for layer in self.layers:
            layer.trainable = value

    @property
    def losses(self):
        values = set()
        for layer in self.layers:
            values.update(layer.losses)
        return list(values)

    @property
    def regularizers(self):
        values = set()
        for layer in self.layers:
            values.update(layer.regularizers)
        return list(values)

    def call(self, inputs, training=False, **kwargs):
        if getattr(self, "_need_to_call_context", False):
            # convert sparse inputs to dense before storing them to the context?
            self.context(AsDenseFeatures()(inputs))

        outputs = inputs
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                filtered_kwargs = filter_kwargs(kwargs, layer, filter_positional_or_keyword=False)
                filtered_kwargs.update(
                    filter_kwargs(
                        {**dict(training=training), **kwargs},
                        layer.call,
                        filter_positional_or_keyword=False,
                    )
                )
            else:
                filtered_kwargs = filter_kwargs(
                    dict(training=training), layer, filter_positional_or_keyword=False
                )
            outputs = layer(outputs, **filtered_kwargs)

        return outputs

    def compute_loss(self, inputs, targets, **kwargs):
        outputs, targets = inputs, targets
        for layer in self.layers:
            outputs, targets = layer.compute_loss(outputs, targets=targets, **kwargs)

        return outputs, targets

    def call_outputs(
        self, outputs: PredictionOutput, training=False, **kwargs
    ) -> "PredictionOutput":
        for layer in self.layers:
            outputs = layer.call_outputs(outputs, training=training, **kwargs)
        return outputs

    def get_config(self):
        config = {}
        for i, layer in enumerate(self.layers):
            config[i] = tf.keras.utils.serialize_keras_object(layer)

        return config

    def __getitem__(self, key):
        return self.layers[key]

    @property
    def is_tabular(self):
        return getattr(self.layers[-1], "is_tabular", False)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layers = [
            tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for conf in config.values()
        ]

        return SequentialBlock(layers)

    def __rrshift__(self, other):
        return right_shift_layer(self, other)

    def __rshift__(self, other):
        # pylint: disable=arguments-out-of-order
        return right_shift_layer(other, self)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ParallelBlock(TabularBlock):
    """Merge multiple layers or TabularModule's into a single output of TabularData.

    Parameters
    ----------
    blocks_to_merge: Union[TabularModule, Dict[str, TabularBlock]]
        TabularBlocks to merge into, this can also be one or multiple dictionaries keyed by the
        name the module should have.
    {tabular_module_parameters}
    """

    def __init__(
        self,
        *inputs: Union[tf.keras.layers.Layer, Dict[str, tf.keras.layers.Layer]],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(
            pre=pre, post=post, aggregation=aggregation, schema=schema, name=name, **kwargs
        )
        self.strict = strict
        self.parallel_layers: Union[List[TabularBlock], Dict[str, TabularBlock]]
        if all(isinstance(x, dict) for x in inputs):
            to_merge: Dict[str, tf.keras.layers.Layer] = reduce(
                lambda a, b: dict(a, **b), inputs
            )  # type: ignore
            parsed_to_merge: Dict[str, TabularBlock] = {}
            for key, val in to_merge.items():
                parsed_to_merge[key] = val
            self.parallel_layers = parsed_to_merge
        elif all(isinstance(x, tf.keras.layers.Layer) for x in inputs):
            parsed: List[TabularBlock] = []
            for i, inp in enumerate(inputs):
                parsed.append(inp)  # type: ignore
            self.parallel_layers = parsed
        else:
            raise ValueError(
                "Please provide one or multiple layer's to merge or "
                f"dictionaries of layer. got: {inputs}"
            )

        # Merge schemas if necessary.
        if not schema and all(getattr(m, "_schema", False) for m in self.parallel_values):
            if len(self.parallel_values) == 1:
                self.set_schema(self.parallel_values[0].schema)
            else:
                s = reduce(
                    lambda a, b: a + b, [m.schema for m in self.parallel_values]
                )  # type: ignore
                self.set_schema(s)

    @property
    def parallel_values(self) -> List[tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return list(self.parallel_layers.values())

        return self.parallel_layers

    @property
    def parallel_dict(self) -> Dict[str, tf.keras.layers.Layer]:
        if isinstance(self.parallel_layers, dict):
            return self.parallel_layers

        return {str(i): m for i, m in enumerate(self.parallel_layers)}

    def select_by_name(self, name: str) -> Optional["Block"]:
        return self.parallel_dict.get(name)

    def __getitem__(self, key) -> "Block":
        return self.parallel_dict[key]

    def __setitem__(self, key: str, item: "Block"):
        self.parallel_dict[key] = item

    def add_branch(self, name: str, block: "Block") -> "ParallelBlock":
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[name] = block

        return self

    def apply_to_branch(self, branch_name: str, *block: "Block"):
        if isinstance(self.parallel_layers, dict):
            self.parallel_layers[branch_name] = self.parallel_layers[branch_name].apply(*block)

    def call(self, inputs, **kwargs):
        if self.strict:
            assert isinstance(inputs, dict), "Inputs needs to be a dict"

        if getattr(self, "_need_to_call_context", False):
            self.context(inputs)

        outputs = {}
        if isinstance(inputs, dict) and all(
            name in inputs for name in list(self.parallel_dict.keys())
        ):
            for name, block in self.parallel_dict.items():
                out = block(inputs[name])
                if not isinstance(out, dict):
                    out = {name: out}
                outputs.update(out)
        else:
            for name, layer in self.parallel_dict.items():
                out = layer(inputs)
                if not isinstance(out, dict):
                    out = {name: out}
                outputs.update(out)

        return outputs

    def compute_call_output_shape(self, input_shape):
        output_shapes = {}

        for name, layer in self.parallel_dict.items():
            if isinstance(input_shape, dict) and all(
                key in input_shape for key in list(self.parallel_dict.keys())
            ):
                out = layer.compute_output_shape(input_shape[name])
            else:
                out = layer.compute_output_shape(input_shape)
            if isinstance(out, dict):
                output_shapes.update(out)
            else:
                output_shapes[name] = out

        return output_shapes

    def build(self, input_shape):
        if isinstance(input_shape, dict) and all(
            name in input_shape for name in list(self.parallel_dict.keys())
        ):
            for key, block in self.parallel_dict.items():
                block.build(input_shape[key])
        else:
            for layer in self.parallel_values:
                layer.build(input_shape)

        return super().build(input_shape)

    def get_config(self):
        return tf_utils.maybe_serialize_keras_objects(
            self, super(ParallelBlock, self).get_config(), ["parallel_layers"]
        )

    @classmethod
    def parse_config(cls, config, custom_objects=None):
        config = tf_utils.maybe_deserialize_keras_objects(config, ["pre", "post", "aggregation"])
        if "schema" in config:
            config["schema"] = schema_utils.tensorflow_metadata_json_to_schema(config["schema"])

        parallel_layers = config.pop("parallel_layers")
        inputs = {
            name: tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
            for name, conf in parallel_layers.items()
        }

        return inputs, config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inputs, config = cls.parse_config(config, custom_objects)

        return cls(inputs, **config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class WithShortcut(ParallelBlock):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        shortcut_filter: Optional[Filter] = None,
        aggregation=None,
        post: Optional[BlockType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        block_outputs_name: Optional[str] = None,
        **kwargs,
    ):
        block_outputs_name = block_outputs_name or block.name
        shortcut = shortcut_filter if shortcut_filter else NoOp()
        inputs = {block_outputs_name: block, "shortcut": shortcut}
        super().__init__(
            inputs,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        output = ParallelBlock.from_config(config, **kwargs)
        output.__class__ = cls

        return output


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ResidualBlock(WithShortcut):
    def __init__(
        self,
        block: Union[tf.keras.layers.Layer, Block],
        activation=None,
        post: Optional[BlockType] = None,
        schema: Optional[Schema] = None,
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        from merlin.models.tf.blocks.core.aggregation import SumResidual

        super().__init__(
            block,
            post=post,
            aggregation=SumResidual(activation=activation),
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DualEncoderBlock(ParallelBlock):
    def __init__(
        self,
        left: Union[TabularBlock, tf.keras.layers.Layer],
        right: Union[TabularBlock, tf.keras.layers.Layer],
        pre: Optional[BlockType] = None,
        post: Optional[BlockType] = None,
        aggregation: Optional[TabularAggregationType] = None,
        schema: Optional[Schema] = None,
        left_name: str = "left",
        right_name: str = "right",
        name: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        if not getattr(left, "is_tabular", False):
            left = SequentialBlock([left, AsTabular(left_name)])
        if not getattr(right, "is_tabular", False):
            right = SequentialBlock([right, AsTabular(right_name)])

        towers = {left_name: left, right_name: right}

        super().__init__(
            towers,
            pre=pre,
            post=post,
            aggregation=aggregation,
            schema=schema,
            name=name,
            strict=strict,
            **kwargs,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        output = ParallelBlock.from_config(config, **kwargs)
        output.__class__ = cls

        return output