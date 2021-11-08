# Copyright 2021 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
The NOOP process plugins exists as an implementation example.

It doesn't really do anything to the collected event
"""
import logging
from typing import Type

from saf.models import CollectedEvent
from saf.models import ProcessConfigBase


log = logging.getLogger(__name__)


def get_config_schema() -> Type[ProcessConfigBase]:
    """
    Get the noop plugin configuration schema.
    """
    return ProcessConfigBase


async def process(  # pylint: disable=unused-argument
    *,
    config: ProcessConfigBase,
    event: CollectedEvent,
) -> CollectedEvent:
    """
    Method called to process the event.
    """
    log.info("Processing: %s", event)
    return event