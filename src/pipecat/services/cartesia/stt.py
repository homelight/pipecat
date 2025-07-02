#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import urllib.parse
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


class CartesiaLiveOptions:
    def __init__(
        self,
        *,
        model: str = "ink-whisper",
        language: str = Language.EN.value,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        **kwargs,
    ):
        self.model = model
        self.language = language
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.additional_params = kwargs

    def to_dict(self):
        params = {
            "model": self.model,
            "language": self.language if isinstance(self.language, str) else self.language.value,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
        }

        return params

    def items(self):
        return self.to_dict().items()

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_params.get(key, default)

    @classmethod
    def from_json(cls, json_str: str) -> "CartesiaLiveOptions":
        return cls(**json.loads(json_str))


class CartesiaSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        sample_rate: int = 16000,
        live_options: Optional[CartesiaLiveOptions] = None,
        **kwargs,
    ):
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = CartesiaLiveOptions(
            model="ink-whisper",
            language=Language.EN.value,
            encoding="pcm_s16le",
            sample_rate=sample_rate,
        )

        merged_options = default_options
        if live_options:
            merged_options_dict = default_options.to_dict()
            merged_options_dict.update(live_options.to_dict())
            merged_options = CartesiaLiveOptions(
                **{
                    k: v
                    for k, v in merged_options_dict.items()
                    if not isinstance(v, str) or v != "None"
                }
            )

        self._settings = merged_options
        self.set_model_name(merged_options.model)
        self._api_key = api_key
        self._base_url = base_url or "api.cartesia.ai"
        self._connection = None
        self._receiver_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send an audio chunk for transcription, maintaining a resilient connection.

        The Cartesia WebSocket can be closed by the remote side at any time. In that
        scenario, ``websockets`` raises a ``ConnectionClosed`` (or
        ``ConnectionClosedOK``) exception on ``send``.  Propagating that
        exception up the stack aborts the whole Pipecat pipeline, resulting in
        noisy logs like:

            websockets.exceptions.ConnectionClosedOK: sent 1000 (OK); then received 1000 (OK)

        Instead, we catch the closure here, perform a best-effort reconnect, and
        silently drop the current audio buffer.  The next audio buffer will be
        delivered over the fresh connection.
        """

        try:
            # If the processor is cancelling/end-of-pipeline we skip sending
            # altogether because the connection is expected to be closed.
            if self._cancelling:
                # Consume the audio frame silently and exit early.
                yield None
                return

            # (Re)connect only when we are still active.
            if not self._connection or self._connection.closed:
                await self._connect()

            # Attempt to send the audio payload
            await self._connection.send(audio)
        except (
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosed,
        ) as e:
            # Connection closed gracefully or with an error – attempt to
            # reconnect so we can continue processing without failing the
            # entire pipeline.
            logger.warning(f"{self}: WebSocket closed during send: {e}. Reconnecting.")
            try:
                await self._disconnect()
            except Exception:
                # Ignore additional errors while cleaning up
                pass

            await self._connect()
            # We purposefully *do not* resend the audio buffer: by the time the
            # reconnection succeeds the buffer would be outdated and resending
            # could confuse the transcription stream.
        except Exception as e:
            # Catch-all to prevent the exception from bubbling up and killing
            # the pipeline.  We log the error and try to keep going.
            logger.error(f"{self}: Unexpected error while sending audio: {e}")

        # Always yield once so the caller can keep iterating over the generator
        yield None

    async def _connect(self):
        params = self._settings.to_dict()
        ws_url = f"wss://{self._base_url}/stt/websocket?{urllib.parse.urlencode(params)}"
        logger.debug(f"Connecting to Cartesia: {ws_url}")
        headers = {"Cartesia-Version": "2025-04-16", "X-API-Key": self._api_key}

        try:
            self._connection = await websockets.connect(ws_url, extra_headers=headers)
            # Setup the receiver task to handle the incoming messages from the Cartesia server
            if self._receiver_task is None or self._receiver_task.done():
                self._receiver_task = asyncio.create_task(self._receive_messages())
            logger.debug(f"Connected to Cartesia")
        except Exception as e:
            logger.error(f"{self}: unable to connect to Cartesia: {e}")

    async def _receive_messages(self):
        try:
            while True:
                if not self._connection or self._connection.closed:
                    break

                message = await self._connection.recv()
                try:
                    data = json.loads(message)
                    await self._process_response(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.debug(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in message receiver: {e}")

    async def _process_response(self, data):
        if "type" in data:
            if data["type"] == "transcript":
                await self._on_transcript(data)

            elif data["type"] == "error":
                logger.error(f"Cartesia error: {data.get('message', 'Unknown error')}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_transcript(self, data):
        if "text" not in data:
            return

        transcript = data.get("text", "")
        is_final = data.get("is_final", False)
        language = None

        if "language" in data:
            try:
                language = Language(data["language"])
            except (ValueError, KeyError):
                pass

        if len(transcript) > 0:
            await self.stop_ttfb_metrics()
            if is_final:
                await self.push_frame(
                    TranscriptionFrame(transcript, "", time_now_iso8601(), language)
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions, just push the frame without tracing
                await self.push_frame(
                    InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
                )

    async def _disconnect(self):
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"Unexpected exception while cancelling task: {e}")
            self._receiver_task = None

        if self._connection and self._connection.open:
            logger.debug("Disconnecting from Cartesia")

            await self._connection.close()
            self._connection = None

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Send finalize command to flush the transcription session
            if self._connection and self._connection.open:
                await self._connection.send("finalize")
