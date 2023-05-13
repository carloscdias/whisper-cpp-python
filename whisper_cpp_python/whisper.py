from . import whisper_cpp
from .whisper_types import WhisperChunk
from typing import List, Literal, Any
import ctypes
import librosa


class Whisper():
    WHISPER_SR = 16000

    def __init__(self, model_path, strategy = 0, n_threads = 1):
        self.context = whisper_cpp.whisper_init_from_file(model_path.encode('utf-8'))
        self.params  = whisper_cpp.whisper_full_default_params(strategy)
        self.params.n_threads = n_threads
        self.params.print_special = False
        self.params.print_progress = False
        self.params.print_realtime = False
        self.params.print_timestamps = False

    def transcribe(self, file, prompt = None, response_format = 'json', temperature = 0.8, language = 'en') -> Any:
        data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
        self.params.language = language.encode('utf-8')
        if prompt:
            self.params.initial_prompt = prompt.encode('utf-8')
        self.params.temperature = temperature
        result = self._full(data)
        return self._parse_format(result, response_format)

    def translate(self, file, prompt = None, response_format = 'json', temperature = 0.8) -> Any:
        data, sr = librosa.load(file, sr=Whisper.WHISPER_SR)
        self.params.translate = True
        self.params.initial_prompt = prompt.encode('utf-8')
        self.params.temperature = temperature
        result = self._full(data)
        return self._parse_format(result, response_format)

    def _full(self, data) -> List[WhisperChunk]:
        chunks: List[WhisperChunk] = []
        # run the inference
        result = whisper_cpp.whisper_full(ctypes.c_void_p(self.context), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))

        if result != 0:
            raise "Error: {}".format(result)

        n_segments = whisper_cpp.whisper_full_n_segments(ctypes.c_void_p(self.context))
        for i in range(n_segments):
            t0  = whisper_cpp.whisper_full_get_segment_t0(ctypes.c_void_p(self.context), i)/100.0
            t1  = whisper_cpp.whisper_full_get_segment_t1(ctypes.c_void_p(self.context), i)/100.0
            txt = whisper_cpp.whisper_full_get_segment_text(ctypes.c_void_p(self.context), i).decode('utf-8')
            chunks.append({
                "start": t0,
                "end": t1,
                "text": txt,
            })
        return chunks

    def _parse_format(self, chunks: List[WhisperChunk], response_format: Literal["json", "text", "srt", "verbose_json", "vtt"]):
        return {
            "json": self._parse_format_json,
            "text": self._parse_format_text,
            "srt": self._parse_format_srt,
            "verbose_json": self._parse_format_verbose_json,
            "vtt": self._parse_format_vtt,
        }[response_format](chunks)

    def _parse_format_json(self, chunks: List[WhisperChunk]):
        return {
            "text": ''.join([c['text'] for c in chunks]),
        }

    def _parse_format_text(self, chunks: List[WhisperChunk]):
        return ''.join([c['text'] for c in chunks])

    def _parse_format_srt(self, chunks: List[WhisperChunk]):
        return '\n'.join([f'{i + 1}\n{Whisper.format_time(c["start"])} --> {Whisper.format_time(c["end"])}\n{c["text"]}\n' for i, c in enumerate(chunks)])

    def _parse_format_verbose_json(self, chunks: List[WhisperChunk]):
        segments = [{
                "id": 0,
                "seek": 0,
                "start": s['start'],
                "end": s['end'],
                "text": s['text'],
                "tokens": [],
                "temperature": 0,
                "avg_logprob": -0.12,
                "compression_ratio": 0.84,
                "no_speech_prob": 0.12,
                "transient": False,
            } for s in chunks]
        return {
            "task": "",
            "language": "english",
            "duration": 9.28,
            "segments": segments,
            "text": ''.join([c['text'] for c in chunks]),
        }

    def _parse_format_vtt(self, chunks: List[WhisperChunk]):
        return '\n'.join([f'{i + 1}\n{Whisper.format_time(c["start"])} --> {Whisper.format_time(c["end"])} align:middle\n{c["text"]}\n' for i, c in enumerate(chunks)])

    def __dealloc__(self):
        # free the memory
        whisper_cpp.whisper_free(ctypes.c_void_p(self.context))

    @staticmethod
    def format_time(t: int):
        msec = t * 10
        hr = msec / (1000 * 60 * 60)
        msec = msec - hr * (1000 * 60 * 60)
        minu = msec / (1000 * 60)
        msec = msec - minu * (1000 * 60)
        sec = msec / 1000
        msec = msec - sec * 1000
        return f'{int(hr):02}:{int(minu):02}:{int(sec):02}.{int(msec):03}'

