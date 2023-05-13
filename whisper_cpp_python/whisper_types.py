from typing import List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class WhisperChunk(TypedDict):
    start: int
    end: int
    text: str

