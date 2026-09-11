import logging
import time
from io import BytesIO
from typing import Optional

from google import genai
from google.genai import types

from music_arena import (
    Audio,
    DetailedTextToMusicPrompt,
    PromptSupport,
    TextToMusicResponse,
)
from music_arena.secret import get_secret
from music_arena.system import TextToMusicAPISystem

_LOGGER = logging.getLogger(__name__)


class Lyria(TextToMusicAPISystem):
    def __init__(
        self,
        *args,
        model_id: Optional[str] = None,
        model_id_secret_name: Optional[str] = None,
        fixed_duration: float = 30.0,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (model_id is None) == (model_id_secret_name is None):
            raise ValueError(
                "Exactly one of model_id or model_id_secret_name must be set."
            )
        self._model_id_literal = model_id
        self._model_id_secret_name = model_id_secret_name
        self._fixed_duration = fixed_duration
        self._timeout = timeout
        self._model_id: Optional[str] = None
        self._client: Optional[genai.Client] = None

    def _prepare(self):
        # Without an explicit timeout, the SDK passes timeout=None through to
        # httpx, which disables request timeouts entirely and can hang the
        # server forever on a stalled call (observed: an 11-day hang in
        # production). `timeout` (seconds) is opt-in per variant.
        http_options = (
            types.HttpOptions(timeout=int(self._timeout * 1000))
            if self._timeout is not None
            else None
        )
        self._client = genai.Client(
            api_key=get_secret("GEMINI_API_KEY"),
            http_options=http_options,
        )
        # Model IDs are public, documented strings (e.g. "lyria-3.5"), not
        # secrets. `model_id_secret_name` remains for older variants already
        # wired up that way.
        self._model_id = (
            self._model_id_literal
            if self._model_id_literal is not None
            else get_secret(self._model_id_secret_name).strip()
        )

    def _release(self):
        if self._client is not None:
            del self._client
            self._client = None
        self._model_id = None

    def prompt_support(self, prompt: DetailedTextToMusicPrompt) -> PromptSupport:
        # Some Lyria variants (e.g. the clip model) always return a
        # fixed-length generation; others treat this as an approximate cap.
        if prompt.duration is not None and prompt.duration > self._fixed_duration:
            return PromptSupport.PARTIAL
        return PromptSupport.SUPPORTED

    async def _generate_single(
        self, prompt: DetailedTextToMusicPrompt, seed: int
    ) -> TextToMusicResponse:
        assert self._client is not None
        assert self._model_id is not None
        timings: list[tuple[str, float]] = []

        _LOGGER.info("Calling Lyria model='%s'", self._model_id)
        s = time.time()
        timings.append(("call", s))
        text_prompt = prompt.overall_prompt
        if prompt.instrumental and "instrumental" not in text_prompt.lower():
            text_prompt = f"{text_prompt} (instrumental only)"
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=text_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["Audio", "Text"],
            ),
        )
        timings.append(("decode", time.time()))

        audio_bytes = None
        text_parts: list[str] = []
        for part in (response.parts or []):
            if part.text:
                text_parts.append(part.text)
            if part.inline_data and part.inline_data.data:
                audio_bytes = part.inline_data.data

        if audio_bytes is None:
            raise RuntimeError("Lyria 3 response did not contain audio inline_data.")

        audio = Audio.from_file(BytesIO(audio_bytes))
        if prompt.duration is not None:
            audio = audio.crop(duration=min(prompt.duration, self._fixed_duration))
        timings.append(("done", time.time()))

        # Lyria returns text parts containing lyric/timing metadata.
        lyrics = (
            prompt.lyrics
            if prompt.lyrics is not None
            else (text_parts[0] if text_parts else None)
        )
        return TextToMusicResponse(audio=audio, lyrics=lyrics, custom_timings=timings)
