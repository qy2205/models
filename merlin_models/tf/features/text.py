import tensorflow as tf

from merlin_models.tf.tabular import TabularLayer


class ParseTokenizedText(TabularLayer):
    def __init__(
        self,
        max_text_length=None,
        aggregation=None,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(aggregation, trainable, name, dtype, dynamic, **kwargs)
        self.max_text_length = max_text_length

    def call(self, inputs, **kwargs):
        outputs, text_tensors, text_column_names = {}, {}, []
        for name, val in inputs.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                values = val[0][:, 0]
                row_lengths = val[1][:, 0]
                text_tensors[name] = tf.RaggedTensor.from_row_lengths(
                    values, row_lengths
                ).to_tensor()
                text_column_names.append("/".join(name.split("/")[:-1]))
            # else:
            #     outputs[name] = val

        for text_col in set(text_column_names):
            outputs[text_col] = dict(
                input_ids=tf.cast(text_tensors[text_col + "/tokens"], tf.int32),
                attention_mask=tf.cast(text_tensors[text_col + "/attention_mask"], tf.int32),
            )

        return outputs

    def compute_output_shape(self, input_shapes):
        assert self.max_text_length is not None

        output_shapes, text_column_names = {}, []
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)
        for name, val in input_shapes.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                text_column_names.append("/".join(name.split("/")[:-1]))

        for text_col in set(text_column_names):
            output_shapes[text_col] = dict(
                input_ids=tf.TensorShape([batch_size, self.max_text_length]),
                attention_mask=tf.TensorShape([batch_size, self.max_text_length]),
            )

        return output_shapes


class TextEmbeddingFeaturesWithTransformers(TabularLayer):
    def __init__(
        self,
        transformer_model,
        max_text_length=None,
        output="pooler_output",
        trainable=False,
        **kwargs
    ):
        super().__init__(trainable=trainable, **kwargs)
        self.parse_tokens = ParseTokenizedText(max_text_length=max_text_length)
        self.transformer_model = transformer_model
        self.transformer_output = output

    def call(self, inputs, **kwargs):
        tokenized = self.parse_tokens(inputs)
        outputs = {}
        for key, val in tokenized.items():
            if self.transformer_output == "pooler_output":
                outputs[key] = self.transformer_model(**val).pooler_output
            elif self.transformer_output == "last_hidden_state":
                outputs[key] = self.transformer_model(**val).last_hidden_state
            else:
                outputs[key] = self.transformer_model(**val)

        return outputs

    def compute_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        # TODO: Handle all transformer output modes

        output_shapes, text_column_names = {}, []
        for name, val in input_shapes.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                text_column_names.append("/".join(name.split("/")[:-1]))

        for text_col in set(text_column_names):
            output_shapes[text_col] = tf.TensorShape(
                [batch_size, self.transformer_model.config.hidden_size]
            )

        return super().compute_output_shape(output_shapes)
