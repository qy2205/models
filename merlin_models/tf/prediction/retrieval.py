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
from typing import Optional, Sequence

import tensorflow as tf
from merlin.schema import Schema, Tags
from tensorflow.python.layers.base import Layer

from merlin_models.tf.losses import LossType, loss_registry

from ..blocks.item_prediction import PredictionsScaler
from ..blocks.retrieval import ItemRetrievalScorer
from ..blocks.transformations import L2Norm
from ..core import Block, MetricOrMetricClass
from ..metrics.ranking import ranking_metrics
from ..prediction.sampling import InBatchSampler, ItemSampler
from .classification import MultiClassClassificationTask
from .evaluation import BruteForceTopK


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class ItemRetrieval(MultiClassClassificationTask):
    DEFAULT_LOSS = "sparse_categorical_crossentropy"
    DEFAULT_METRICS = ranking_metrics(top_ks=[10, 20])

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[Layer] = None,
        loss: Optional[LossType] = DEFAULT_LOSS,
        metrics: Sequence[MetricOrMetricClass] = DEFAULT_METRICS,
        train_metrics: Sequence[MetricOrMetricClass] = None,
        pre: Optional[Block] = None,
        pre_metrics: Optional[Block] = None,
        **kwargs,
    ):

        super().__init__(
            metrics=list(metrics),
            target_name=target_name,
            task_name=task_name,
            task_block=task_block,
            pre=pre,
            **kwargs,
        )
        self.loss = loss_registry.parse(loss)

        if len(self.metrics) > 0:
            max_k = tf.reduce_max(sum([metric.top_ks for metric in metrics], []))
            pre_metrics = BruteForceTopK(k=max_k - 1)
        self.pre_metrics = pre_metrics

    def load_candidates(self, candidates):
        """Load candidates to use for evaluation/prediction"""
        if not self.pre_metrics:
            raise ValueError("The evaluation block is not set for metrics evaluation")
        self.pre_metrics.load_index(candidates)


def ItemRetrievalTask(
    schema: Schema,
    loss: Optional[LossType] = "categorical_crossentropy",
    metrics: Sequence[MetricOrMetricClass] = ranking_metrics(top_ks=[10, 20]),
    samplers: Sequence[ItemSampler] = (),
    extra_pre_call: Optional[Block] = None,
    target_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_block: Optional[Layer] = None,
    softmax_temperature: float = 1,
    normalize: bool = True,
) -> ItemRetrieval:
    """
    Function to create the ItemRetrieval task with the right parameters.

    Parameters
    ----------
        schema: Schema
            The schema object including features to use and their properties.
        loss: Optional[LossType]
            Loss function.
            Defaults to `categorical_crossentropy`.
        samplers: List[ItemSampler]
            List of samplers for negative sampling, by default `[InBatchSampler()]`
        metrics: Sequence[MetricOrMetricClass]
            List of top-k ranking metrics.
            Defaults to MultiClassClassificationTask.DEFAULT_METRICS["ranking"].
        extra_pre_call: Optional[PredictionBlock]
            Optional extra pre-call block. Defaults to None.
        target_name: Optional[str]
            If specified, name of the target tensor to retrieve from dataloader.
            Defaults to None.
        task_name: Optional[str]
            name of the task.
            Defaults to None.
        task_block: Block
            The `Block` that applies additional layers op to inputs.
            Defaults to None.
        softmax_temperature: float
            Parameter used to reduce model overconfidence, so that softmax(logits / T).
            Defaults to 1.
        normalize: bool
            Apply L2 normalization before computing dot interactions.
            Defaults to True.

    Returns
    -------
        PredictionTask
            The item retrieval prediction task
    """
    item_id_feature_name = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    if samplers is None or len(samplers) == 0:
        samplers = (InBatchSampler(),)

    prediction_call = ItemRetrievalScorer(
        samplers=samplers, item_id_feature_name=item_id_feature_name
    )

    if normalize:
        prediction_call = L2Norm().connect(prediction_call)

    if softmax_temperature != 1:
        prediction_call = prediction_call.connect(PredictionsScaler(1.0 / softmax_temperature))

    if extra_pre_call is not None:
        prediction_call = prediction_call.connect(extra_pre_call)

    return ItemRetrieval(
        target_name,
        task_name,
        task_block,
        loss=loss,
        metrics=metrics,
        pre=prediction_call,
    )