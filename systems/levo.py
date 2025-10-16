import asyncio
import gc
import os
import time

import numpy as np
import torch
from codeclm.models import CodecLM, builders
from omegaconf import OmegaConf

from music_arena import (
    Audio,
    DetailedTextToMusicPrompt,
    PromptSupport,
    TextToMusicResponse,
)
from music_arena.chat.lyrics import generate_lyrics
from music_arena.system import TextToMusicGPUSystem


class SongGeneration(TextToMusicGPUSystem):
    def __init__(
        self,
        ckpt_path: str = "./songgeneration_base_new",
        max_duration: float = 30.0,
        lyrics_config: str = "4o-v00",
        use_flash_attn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ckpt_path = ckpt_path
        self._max_duration = max_duration
        self._lyrics_config = lyrics_config
        self._use_flash_attn = use_flash_attn

        self._cfg = None
        self._audio_tokenizer = None
        self._model = None

    def _prepare(self):
        """Initialize model and components based on Tencent SongGeneration implementation"""
        # Load configuration
        cfg_path = os.path.join(self._ckpt_path, "config.yaml")
        ckpt_path = os.path.join(self._ckpt_path, "model.pt")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        self._cfg = OmegaConf.load(cfg_path)
        self._cfg.lm.use_flash_attn_2 = self._use_flash_attn
        self._cfg.mode = "inference"

        # Load audio tokenizer
        self._audio_tokenizer = builders.get_audio_tokenizer_model(
            self._cfg.audio_tokenizer_checkpoint, self._cfg
        )
        self._audio_tokenizer = self._audio_tokenizer.eval().cuda()

        # Load AudioLM
        audiolm = builders.get_lm_model(self._cfg)
        checkpoint = torch.load(self._ckpt_path, map_location="cpu")
        audiolm_state_dict = {
            k.replace("audiolm.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("audiolm")
        }
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval().cuda().to(torch.float16)

        # Load CodecLM
        self._model = CodecLM(
            name="tmp",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=self._max_duration,
            seperate_tokenizer=None,
        )

    def _release(self):
        """Clean up model and components"""
        if self._cfg is not None:
            del self._cfg
        if self._audio_tokenizer is not None:
            del self._audio_tokenizer
        if self._model is not None:
            del self._model

        torch.cuda.empty_cache()
        gc.collect()

        self._cfg = None
        self._audio_tokenizer = None
        self._model = None

    def prompt_support(self, prompt: DetailedTextToMusicPrompt) -> PromptSupport:
        if prompt.duration is not None and prompt.duration > self._max_duration:
            return PromptSupport.UNSUPPORTED
        return PromptSupport.SUPPORTED

    def _generate_single(
        self, prompt: DetailedTextToMusicPrompt, seed: int
    ) -> TextToMusicResponse:
        timings = []
        assert self._model is not None
        duration = (
            prompt.duration if prompt.duration is not None else self._max_duration
        )
        duration = min(duration, self._max_duration)

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Extract lyrics
        if prompt.instrumental:
            lyrics = ""
        else:
            lyrics = prompt.lyrics
            if lyrics is None:
                timings.append(("lyrics", time.time()))
                lyrics = asyncio.run(
                    generate_lyrics(
                        prompt=prompt.copy(duration=30.0),
                        config=self._lyrics_config,
                    )
                )

        # Prepare generation input for text-to-music
        timings.append(("preprocess", time.time()))

        # Create generation input matching Tencent SongGeneration format
        generate_inp = {
            "lyrics": [lyrics.replace(" ", " ")],
            "descriptions": [prompt.overall_prompt] if prompt.overall_prompt else None,
            "melody_wavs": None,  # No melody provided
            "vocal_wavs": None,  # No vocal provided
            "bgm_wavs": None,  # No BGM provided
            "melody_is_wav": True,  # No melody provided
        }

        # Generate tokens using the model
        timings.append(("generate", time.time()))

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = self._model.generate(**generate_inp, return_tokens=True)

        # Generate audio from tokens using separate tokenizer
        with torch.no_grad():
            wav_separate = self._separate_tokenizer.generate_audio(
                tokens, chunked=True, gen_type="mixed"
            )

        timings.append(("done", time.time()))

        # Convert to numpy and create Audio object
        audio_arr = wav_separate[0].cpu().numpy().squeeze()
        audio = Audio(
            samples=audio_arr,
            sample_rate=self._cfg.sample_rate,
        )
        audio = audio.crop(duration=duration)

        return TextToMusicResponse(
            audio=audio,
            lyrics=lyrics if not prompt.instrumental else None,
            custom_timings=timings,
        )
