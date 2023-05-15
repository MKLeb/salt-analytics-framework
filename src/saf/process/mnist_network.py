# Copyright 2021-2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Mask data based on provided regex rules.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Type
from typing import TypeVar

import joblib
import numpy

from saf.models import CollectedEvent
from saf.models import PipelineRunContext
from saf.models import ProcessConfigBase

log = logging.getLogger(__name__)

RegexProcessObject = TypeVar("RegexProcessObject")


class MNISTNetworkConfig(ProcessConfigBase):
    """
    Configuration schema for the jupyter notebook processor plugin.
    """

    model: str


def get_config_schema() -> Type[ProcessConfigBase]:
    """
    Get the jupyter notebook processor plugin configuration schema.
    """
    return MNISTNetworkConfig


async def process(
    *,
    ctx: PipelineRunContext[MNISTNetworkConfig],
    event: CollectedEvent,
) -> CollectedEvent:
    """
    Run the jupyter notebook, doing papermill parameterizing using the event data given.
    """
    if "mnist_model" not in ctx.cache:
        model_path = pathlib.Path(ctx.config.model)
        log.debug(f"Loading the mnist model from {model_path}")
        ctx.cache["mnist_model"] = joblib.load(model_path)
        ctx.cache["mnist_model_evaluations"] = []
    else:
        log.debug("Did not load the model, already cached it")

    model = ctx.cache["mnist_model"]
    new_event = event.copy()
    x, y = new_event.data["test_mnist_digit"]
    evaluate = model.evaluate(numpy.asarray([x]), numpy.asarray([y]))
    log.debug(f"Evaluate result: {evaluate}")
    ctx.cache["mnist_model_evaluations"].append(evaluate)
    avg_accuracy = sum([res[1] for res in ctx.cache["mnist_model_evaluations"]]) / len(
        ctx.cache["mnist_model_evaluations"]
    )
    avg_loss = sum([res[0] for res in ctx.cache["mnist_model_evaluations"]]) / len(
        ctx.cache["mnist_model_evaluations"]
    )
    log.debug(f"Average accuracy: {avg_accuracy}, average loss: {avg_loss}")
    new_event.data = {
        "evaluation": evaluate,
        "accuracy": avg_accuracy,
        "loss": avg_loss,
    }

    return new_event
