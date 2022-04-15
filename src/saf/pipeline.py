# Copyright 2021-2022 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Salt Analytics Framework Pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from types import ModuleType
from typing import Any

import backoff

from saf.models import CollectConfigBase
from saf.models import CollectedEvent
from saf.models import ForwardConfigBase
from saf.models import PipelineConfig
from saf.models import PipelineRunContext
from saf.models import ProcessConfigBase

log = logging.getLogger(__name__)


def _check_backoff_exception(exc: Exception) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return True
    return False


class Pipeline:
    """
    Salt Analytics Pipeline.
    """

    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
        self.collect_config: CollectConfigBase = config.parent.collectors[config.collect]
        self.process_configs: list[ProcessConfigBase] = []
        for config_name in config.process:
            self.process_configs.append(config.parent.processors[config_name])
        self.forward_configs: list[ForwardConfigBase] = []
        for config_name in config.forward:
            self.forward_configs.append(config.parent.forwarders[config_name])

    async def run(self) -> None:
        """
        Run the pipeline.
        """
        log.info("Pipeline %r started", self.name)
        while True:
            try:
                await self._run()
            except asyncio.CancelledError:
                log.info("Pipeline %r canceled", self.name)
                break
            except Exception as exc:  # pylint: disable=broad-except
                log.error(
                    "Failed to start the pipeline %s due to an error: %s",
                    self.name,
                    exc,
                    exc_info=True,
                )
                break

    @backoff.on_exception(
        backoff.expo,
        Exception,
        jitter=backoff.full_jitter,
        max_tries=5,
        giveup=_check_backoff_exception,
    )
    async def _run(self) -> None:
        shared_cache: dict[str, Any] = {}
        process_ctxs: dict[str, PipelineRunContext[ProcessConfigBase]] = {}
        forward_ctxs: dict[str, PipelineRunContext[ForwardConfigBase]] = {}
        collect_ctx: PipelineRunContext[CollectConfigBase] = PipelineRunContext.construct(
            config=self.collect_config,
            shared_cache=shared_cache,
        )
        try:
            collect_plugin = self.collect_config.loaded_plugin
            async for event in collect_plugin.collect(ctx=collect_ctx):
                # Process the event
                for process_config in self.process_configs:
                    if process_config.name not in process_ctxs:
                        process_ctxs[process_config.name] = PipelineRunContext.construct(
                            config=process_config,
                            shared_cache=shared_cache,
                        )
                    # We pass copies of the event so that, in case an exception occurs while
                    # the event is being processed, and the event has already been modified,
                    # the next processor to run will get an unmodified copy of the event, not
                    # the partially processed event
                    original_event = event.copy()
                    process_plugin = process_config.loaded_plugin
                    try:
                        event = await process_plugin.process(
                            ctx=process_ctxs[process_config.name],
                            event=event,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        log.error(
                            "An exception occurred while processing the event: %s",
                            exc,
                            exc_info=True,
                        )
                        # Restore the original event
                        event = original_event
                # Forward the event
                coros = []
                for forward_config in self.forward_configs:
                    if forward_config.name not in forward_ctxs:
                        forward_ctxs[forward_config.name] = PipelineRunContext.construct(
                            config=forward_config,
                            shared_cache=shared_cache,
                        )
                    forward_plugin = forward_config.loaded_plugin
                    coros.append(
                        self._wrap_forwarder_plugin_call(
                            forward_plugin,
                            forward_ctxs[forward_config.name],
                            event.copy(),
                        ),
                    )
                if self.config.concurrent_forwarders:
                    await asyncio.gather(*coros)
                else:
                    for coro in coros:
                        await coro
        finally:
            shared_cache.clear()
            process_ctxs.clear()
            forward_ctxs.clear()

    async def _wrap_forwarder_plugin_call(
        self, plugin: ModuleType, ctx: PipelineRunContext[ForwardConfigBase], event: CollectedEvent
    ) -> None:
        try:
            await plugin.forward(ctx=ctx, event=event)
        except Exception as exc:  # pylint: disable=broad-except
            log.error(
                "An exception occurred while forwarding the event through config %r: %s",
                ctx.config,
                exc,
                exc_info=True,
            )
