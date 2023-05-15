# Copyright 2021-2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
A plugin which downloads (if not already downloaded) the mnist digits dataset and feeds images from it one by one down the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import random
from typing import AsyncIterator
from typing import Type

import numpy as np
from tensorflow import keras

from saf.models import CollectConfigBase
from saf.models import CollectedEvent
from saf.models import PipelineRunContext

log = logging.getLogger(__name__)


class MNISTDigitsConfig(CollectConfigBase):
    """
    Configuration schema for the mnist_digits collect plugin.
    """

    interval: float = 5
    path: str = "/tmp/mnist_digits"
    x_path: str = "/tmp/mnist_test_x"
    y_path: str = "/tmp/mnist_test_y"


def get_config_schema() -> Type[MNISTDigitsConfig]:
    """
    Get the mnist_digits plugin configuration schema.
    """
    return MNISTDigitsConfig


async def collect(*, ctx: PipelineRunContext[MNISTDigitsConfig]) -> AsyncIterator[CollectedEvent]:
    """
    Method called to collect events.
    """
    file_path = pathlib.Path(ctx.config.path)
    log.debug(f"Downloading the MNIST digits dataset to {file_path}")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=file_path)
    x_test = x_test / 255  # Normalize
    x_test_flattened = x_test.reshape(len(x_test), 28 * 28)  # Flatten

    while True:
        idx = random.choice(range(len(x_test_flattened)))
        rand_test = (x_test_flattened[idx], y_test[idx])
        with open(ctx.config.x_path, "wb") as x_out:
            np.save(x_out, rand_test[0])
        with open(ctx.config.y_path, "wb") as y_out:
            np.save(y_out, rand_test[1])
        event = CollectedEvent(data={"x_path": ctx.config.x_path, "y_path": ctx.config.y_path})
        yield event
        await asyncio.sleep(ctx.config.interval)
