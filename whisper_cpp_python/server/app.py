import os
import io
import json
from threading import Lock
from typing import List, Optional, Union, Any
from typing_extensions import TypedDict, Literal, Annotated

import whisper_cpp_python

from fastapi import Depends, FastAPI, APIRouter, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings, Field
from sse_starlette.sse import EventSourceResponse


class Settings(BaseSettings):
    model: str
    strategy: int = 0
    n_threads: int = max((os.cpu_count() or 2) // 2, 1)


router = APIRouter()

whisper: Optional[whisper_cpp_python.Whisper] = None


def create_app(settings: Optional[Settings] = None):
    if settings is None:
        settings = Settings()
    app = FastAPI(
        title="whisper.cpp Python API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    global whisper
    whisper = whisper_cpp_python.Whisper(
        model_path=settings.model,
        strategy=settings.strategy,
        n_threads=settings.n_threads,
    )
    return app


whisper_lock = Lock()


def get_whisper():
    with whisper_lock:
        yield whisper


@router.post("/v1/audio/transcriptions")
def transcription(
        file: Annotated[bytes, File()],
        model: Annotated[str, Body()],
        prompt: Annotated[str, Body()] = None,
        response_format: Annotated[Literal["json", "text", "srt", "verbose_json", "vtt"], Body()] = "json",
        temperature: Annotated[float, Body()] = 0.8,
        language: Annotated[str, Body()] = 'en',
        whisper: whisper_cpp_python.Whisper = Depends(get_whisper)) -> Any:
    return whisper.transcribe(io.BytesIO(file), prompt, response_format, temperature, language)


@router.post("/v1/audio/translations")
def translation(
        file: Annotated[bytes, File()],
        model: Annotated[str, Body()],
        prompt: Annotated[str, Body()] = None,
        response_format: Annotated[Literal["json", "text", "srt", "verbose_json", "vtt"], Body()] = "json",
        temperature: Annotated[float, Body()] = 0.8,
        whisper: whisper_cpp_python.Whisper = Depends(get_whisper)) -> Any:
    return whisper.translate(io.BytesIO(file), prompt, response_format, temperature)

