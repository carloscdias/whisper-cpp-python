# auto-generated file
import sys
import os
import ctypes
import pathlib


# Load the library
def _load_shared_library(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    if "WHISPER_CPP_LIB" in os.environ:
        lib_base_name = os.environ["WHISPER_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "whisper"

# Load the library
_lib = _load_shared_library(_lib_base_name)



whisper_context_p = ctypes.c_void_p

whisper_state_p = ctypes.c_void_p

whisper_token = ctypes.c_int

class whisper_token_data(ctypes.Structure):
    _fields_ = [
        ("id", whisper_token),
        ("tid", whisper_token),
        ("p", ctypes.c_float),
        ("plog", ctypes.c_float),
        ("pt", ctypes.c_float),
        ("ptsum", ctypes.c_float),
        ("t0", ctypes.c_int64),
        ("t1", ctypes.c_int64),
        ("vlen", ctypes.c_float),
    ]

class whisper_model_loader(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("read", ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t))),
        ("eof", ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_void_p))),
        ("close", ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_void_p))),
    ]

def whisper_init_from_file(path_model: ctypes.c_char_p) -> whisper_context_p:
    return _lib.whisper_init_from_file(path_model)

_lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
_lib.whisper_init_from_file.restype = whisper_context_p


def whisper_init_from_buffer(buffer: ctypes.c_void_p,
    buffer_size: ctypes.c_size_t) -> whisper_context_p:
    return _lib.whisper_init_from_buffer(buffer, buffer_size)

_lib.whisper_init_from_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.whisper_init_from_buffer.restype = whisper_context_p


def whisper_init(loader: ctypes.POINTER(whisper_model_loader)) -> whisper_context_p:
    return _lib.whisper_init(loader)

_lib.whisper_init.argtypes = [ctypes.POINTER(whisper_model_loader)]
_lib.whisper_init.restype = whisper_context_p


def whisper_init_from_file_no_state(path_model: ctypes.c_char_p) -> whisper_context_p:
    return _lib.whisper_init_from_file_no_state(path_model)

_lib.whisper_init_from_file_no_state.argtypes = [ctypes.c_char_p]
_lib.whisper_init_from_file_no_state.restype = whisper_context_p


def whisper_init_from_buffer_no_state(buffer: ctypes.c_void_p,
    buffer_size: ctypes.c_size_t) -> whisper_context_p:
    return _lib.whisper_init_from_buffer_no_state(buffer, buffer_size)

_lib.whisper_init_from_buffer_no_state.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.whisper_init_from_buffer_no_state.restype = whisper_context_p


def whisper_init_no_state(loader: ctypes.POINTER(whisper_model_loader)) -> whisper_context_p:
    return _lib.whisper_init_no_state(loader)

_lib.whisper_init_no_state.argtypes = [ctypes.POINTER(whisper_model_loader)]
_lib.whisper_init_no_state.restype = whisper_context_p


def whisper_init_state(ctx: whisper_context_p) -> whisper_state_p:
    return _lib.whisper_init_state(ctx)

_lib.whisper_init_state.argtypes = [whisper_context_p]
_lib.whisper_init_state.restype = whisper_state_p


def whisper_free(ctx: whisper_context_p):
    _lib.whisper_free(ctx)

_lib.whisper_free.argtypes = [whisper_context_p]
_lib.whisper_free.restype = None


def whisper_free_state(state: whisper_state_p):
    _lib.whisper_free_state(state)

_lib.whisper_free_state.argtypes = [whisper_state_p]
_lib.whisper_free_state.restype = None


def whisper_pcm_to_mel(ctx: whisper_context_p,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_pcm_to_mel(ctx, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel.argtypes = [whisper_context_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_pcm_to_mel.restype = ctypes.c_int


def whisper_pcm_to_mel_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_pcm_to_mel_with_state.restype = ctypes.c_int


def whisper_pcm_to_mel_phase_vocoder(ctx: whisper_context_p,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_pcm_to_mel_phase_vocoder(ctx, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel_phase_vocoder.argtypes = [whisper_context_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_pcm_to_mel_phase_vocoder.restype = ctypes.c_int


def whisper_pcm_to_mel_phase_vocoder_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_pcm_to_mel_phase_vocoder_with_state(ctx, state, samples, n_samples, n_threads)

_lib.whisper_pcm_to_mel_phase_vocoder_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_pcm_to_mel_phase_vocoder_with_state.restype = ctypes.c_int


def whisper_set_mel(ctx: whisper_context_p,
    data: ctypes.POINTER(ctypes.c_float),
    n_len: ctypes.c_int,
    n_mel: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_set_mel(ctx, data, n_len, n_mel)

_lib.whisper_set_mel.argtypes = [whisper_context_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_set_mel.restype = ctypes.c_int


def whisper_set_mel_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    data: ctypes.POINTER(ctypes.c_float),
    n_len: ctypes.c_int,
    n_mel: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_set_mel_with_state(ctx, state, data, n_len, n_mel)

_lib.whisper_set_mel_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_set_mel_with_state.restype = ctypes.c_int


def whisper_encode(ctx: whisper_context_p,
    offset: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_encode(ctx, offset, n_threads)

_lib.whisper_encode.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_encode.restype = ctypes.c_int


def whisper_encode_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    offset: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_encode_with_state(ctx, state, offset, n_threads)

_lib.whisper_encode_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_encode_with_state.restype = ctypes.c_int


def whisper_decode(ctx: whisper_context_p,
    tokens: ctypes.POINTER(whisper_token),
    n_tokens: ctypes.c_int,
    n_past: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_decode(ctx, tokens, n_tokens, n_past, n_threads)

_lib.whisper_decode.argtypes = [whisper_context_p, ctypes.POINTER(whisper_token), ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.whisper_decode.restype = ctypes.c_int


def whisper_decode_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    tokens: ctypes.POINTER(whisper_token),
    n_tokens: ctypes.c_int,
    n_past: ctypes.c_int,
    n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_decode_with_state(ctx, state, tokens, n_tokens, n_past, n_threads)

_lib.whisper_decode_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.POINTER(whisper_token), ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.whisper_decode_with_state.restype = ctypes.c_int


def whisper_tokenize(ctx: whisper_context_p,
    text: ctypes.c_char_p,
    tokens: ctypes.POINTER(whisper_token),
    n_max_tokens: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_tokenize(ctx, text, tokens, n_max_tokens)

_lib.whisper_tokenize.argtypes = [whisper_context_p, ctypes.c_char_p, ctypes.POINTER(whisper_token), ctypes.c_int]
_lib.whisper_tokenize.restype = ctypes.c_int


def whisper_lang_max_id() -> ctypes.c_int:
    return _lib.whisper_lang_max_id()

_lib.whisper_lang_max_id.argtypes = []
_lib.whisper_lang_max_id.restype = ctypes.c_int


def whisper_lang_id(lang: ctypes.c_char_p) -> ctypes.c_int:
    return _lib.whisper_lang_id(lang)

_lib.whisper_lang_id.argtypes = [ctypes.c_char_p]
_lib.whisper_lang_id.restype = ctypes.c_int


def whisper_lang_str(id: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_lang_str(id)

_lib.whisper_lang_str.argtypes = [ctypes.c_int]
_lib.whisper_lang_str.restype = ctypes.c_char_p


def whisper_lang_auto_detect(ctx: whisper_context_p,
    offset_ms: ctypes.c_int,
    n_threads: ctypes.c_int,
    lang_probs: ctypes.POINTER(ctypes.c_float)) -> ctypes.c_int:
    return _lib.whisper_lang_auto_detect(ctx, offset_ms, n_threads, lang_probs)

_lib.whisper_lang_auto_detect.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_lib.whisper_lang_auto_detect.restype = ctypes.c_int


def whisper_lang_auto_detect_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    offset_ms: ctypes.c_int,
    n_threads: ctypes.c_int,
    lang_probs: ctypes.POINTER(ctypes.c_float)) -> ctypes.c_int:
    return _lib.whisper_lang_auto_detect_with_state(ctx, state, offset_ms, n_threads, lang_probs)

_lib.whisper_lang_auto_detect_with_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_lib.whisper_lang_auto_detect_with_state.restype = ctypes.c_int


def whisper_n_len(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_n_len(ctx)

_lib.whisper_n_len.argtypes = [whisper_context_p]
_lib.whisper_n_len.restype = ctypes.c_int


def whisper_n_len_from_state(state: whisper_state_p) -> ctypes.c_int:
    return _lib.whisper_n_len_from_state(state)

_lib.whisper_n_len_from_state.argtypes = [whisper_state_p]
_lib.whisper_n_len_from_state.restype = ctypes.c_int


def whisper_n_vocab(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_n_vocab(ctx)

_lib.whisper_n_vocab.argtypes = [whisper_context_p]
_lib.whisper_n_vocab.restype = ctypes.c_int


def whisper_n_text_ctx(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_n_text_ctx(ctx)

_lib.whisper_n_text_ctx.argtypes = [whisper_context_p]
_lib.whisper_n_text_ctx.restype = ctypes.c_int


def whisper_n_audio_ctx(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_n_audio_ctx(ctx)

_lib.whisper_n_audio_ctx.argtypes = [whisper_context_p]
_lib.whisper_n_audio_ctx.restype = ctypes.c_int


def whisper_is_multilingual(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_is_multilingual(ctx)

_lib.whisper_is_multilingual.argtypes = [whisper_context_p]
_lib.whisper_is_multilingual.restype = ctypes.c_int


def whisper_model_n_vocab(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_vocab(ctx)

_lib.whisper_model_n_vocab.argtypes = [whisper_context_p]
_lib.whisper_model_n_vocab.restype = ctypes.c_int


def whisper_model_n_audio_ctx(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_audio_ctx(ctx)

_lib.whisper_model_n_audio_ctx.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_ctx.restype = ctypes.c_int


def whisper_model_n_audio_state(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_audio_state(ctx)

_lib.whisper_model_n_audio_state.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_state.restype = ctypes.c_int


def whisper_model_n_audio_head(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_audio_head(ctx)

_lib.whisper_model_n_audio_head.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_head.restype = ctypes.c_int


def whisper_model_n_audio_layer(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_audio_layer(ctx)

_lib.whisper_model_n_audio_layer.argtypes = [whisper_context_p]
_lib.whisper_model_n_audio_layer.restype = ctypes.c_int


def whisper_model_n_text_ctx(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_text_ctx(ctx)

_lib.whisper_model_n_text_ctx.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_ctx.restype = ctypes.c_int


def whisper_model_n_text_state(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_text_state(ctx)

_lib.whisper_model_n_text_state.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_state.restype = ctypes.c_int


def whisper_model_n_text_head(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_text_head(ctx)

_lib.whisper_model_n_text_head.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_head.restype = ctypes.c_int


def whisper_model_n_text_layer(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_text_layer(ctx)

_lib.whisper_model_n_text_layer.argtypes = [whisper_context_p]
_lib.whisper_model_n_text_layer.restype = ctypes.c_int


def whisper_model_n_mels(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_n_mels(ctx)

_lib.whisper_model_n_mels.argtypes = [whisper_context_p]
_lib.whisper_model_n_mels.restype = ctypes.c_int


def whisper_model_ftype(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_ftype(ctx)

_lib.whisper_model_ftype.argtypes = [whisper_context_p]
_lib.whisper_model_ftype.restype = ctypes.c_int


def whisper_model_type(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_model_type(ctx)

_lib.whisper_model_type.argtypes = [whisper_context_p]
_lib.whisper_model_type.restype = ctypes.c_int


def whisper_get_logits(ctx: whisper_context_p) -> ctypes.POINTER(ctypes.c_float):
    return _lib.whisper_get_logits(ctx)

_lib.whisper_get_logits.argtypes = [whisper_context_p]
_lib.whisper_get_logits.restype = ctypes.POINTER(ctypes.c_float)


def whisper_get_logits_from_state(state: whisper_state_p) -> ctypes.POINTER(ctypes.c_float):
    return _lib.whisper_get_logits_from_state(state)

_lib.whisper_get_logits_from_state.argtypes = [whisper_state_p]
_lib.whisper_get_logits_from_state.restype = ctypes.POINTER(ctypes.c_float)


def whisper_token_to_str(ctx: whisper_context_p,
    token: whisper_token) -> ctypes.c_char_p:
    return _lib.whisper_token_to_str(ctx, token)

_lib.whisper_token_to_str.argtypes = [whisper_context_p, whisper_token]
_lib.whisper_token_to_str.restype = ctypes.c_char_p


def whisper_model_type_readable(ctx: whisper_context_p) -> ctypes.c_char_p:
    return _lib.whisper_model_type_readable(ctx)

_lib.whisper_model_type_readable.argtypes = [whisper_context_p]
_lib.whisper_model_type_readable.restype = ctypes.c_char_p


def whisper_token_eot(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_eot(ctx)

_lib.whisper_token_eot.argtypes = [whisper_context_p]
_lib.whisper_token_eot.restype = whisper_token


def whisper_token_sot(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_sot(ctx)

_lib.whisper_token_sot.argtypes = [whisper_context_p]
_lib.whisper_token_sot.restype = whisper_token


def whisper_token_prev(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_prev(ctx)

_lib.whisper_token_prev.argtypes = [whisper_context_p]
_lib.whisper_token_prev.restype = whisper_token


def whisper_token_solm(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_solm(ctx)

_lib.whisper_token_solm.argtypes = [whisper_context_p]
_lib.whisper_token_solm.restype = whisper_token


def whisper_token_not(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_not(ctx)

_lib.whisper_token_not.argtypes = [whisper_context_p]
_lib.whisper_token_not.restype = whisper_token


def whisper_token_beg(ctx: whisper_context_p) -> whisper_token:
    return _lib.whisper_token_beg(ctx)

_lib.whisper_token_beg.argtypes = [whisper_context_p]
_lib.whisper_token_beg.restype = whisper_token


def whisper_token_lang(ctx: whisper_context_p,
    lang_id: ctypes.c_int) -> whisper_token:
    return _lib.whisper_token_lang(ctx, lang_id)

_lib.whisper_token_lang.argtypes = [whisper_context_p, ctypes.c_int]
_lib.whisper_token_lang.restype = whisper_token


def whisper_token_translate() -> whisper_token:
    return _lib.whisper_token_translate()

_lib.whisper_token_translate.argtypes = []
_lib.whisper_token_translate.restype = whisper_token


def whisper_token_transcribe() -> whisper_token:
    return _lib.whisper_token_transcribe()

_lib.whisper_token_transcribe.argtypes = []
_lib.whisper_token_transcribe.restype = whisper_token


def whisper_print_timings(ctx: whisper_context_p):
    _lib.whisper_print_timings(ctx)

_lib.whisper_print_timings.argtypes = [whisper_context_p]
_lib.whisper_print_timings.restype = None


def whisper_reset_timings(ctx: whisper_context_p):
    _lib.whisper_reset_timings(ctx)

_lib.whisper_reset_timings.argtypes = [whisper_context_p]
_lib.whisper_reset_timings.restype = None


def whisper_print_system_info() -> ctypes.c_char_p:
    return _lib.whisper_print_system_info()

_lib.whisper_print_system_info.argtypes = []
_lib.whisper_print_system_info.restype = ctypes.c_char_p


whisper_new_segment_callback = ctypes.POINTER(ctypes.CFUNCTYPE(whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_void_p))

whisper_progress_callback = ctypes.POINTER(ctypes.CFUNCTYPE(whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_void_p))

whisper_encoder_begin_callback = ctypes.POINTER(ctypes.CFUNCTYPE(whisper_context_p, whisper_state_p, ctypes.c_void_p))

whisper_logits_filter_callback = ctypes.POINTER(ctypes.CFUNCTYPE(whisper_context_p, whisper_state_p, ctypes.POINTER(whisper_token_data), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p))

class greedy(ctypes.Structure):
    _fields_ = [
        ("best_of", ctypes.c_int),
    ]

class beam_search(ctypes.Structure):
    _fields_ = [
        ("beam_size", ctypes.c_int),
        ("patience", ctypes.c_float),
    ]

class whisper_full_params(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("speed_up", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("initial_prompt", ctypes.c_char_p),
        ("prompt_tokens", ctypes.POINTER(whisper_token)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_non_speech_tokens", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        ("greedy", greedy),
        ("beam_search", beam_search),
        ("new_segment_callback", whisper_new_segment_callback),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", whisper_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", whisper_encoder_begin_callback),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", whisper_logits_filter_callback),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
    ]

def whisper_full_default_params(strategy: ctypes.c_int) -> whisper_full_params:
    return _lib.whisper_full_default_params(strategy)

_lib.whisper_full_default_params.argtypes = [ctypes.c_int]
_lib.whisper_full_default_params.restype = whisper_full_params


def whisper_full(ctx: whisper_context_p,
    params: whisper_full_params,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_full(ctx, params, samples, n_samples)

_lib.whisper_full.argtypes = [whisper_context_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.whisper_full.restype = ctypes.c_int


def whisper_full_with_state(ctx: whisper_context_p,
    state: whisper_state_p,
    params: whisper_full_params,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_full_with_state(ctx, state, params, samples, n_samples)

_lib.whisper_full_with_state.argtypes = [whisper_context_p, whisper_state_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.whisper_full_with_state.restype = ctypes.c_int


def whisper_full_parallel(ctx: whisper_context_p,
    params: whisper_full_params,
    samples: ctypes.POINTER(ctypes.c_float),
    n_samples: ctypes.c_int,
    n_processors: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_full_parallel(ctx, params, samples, n_samples, n_processors)

_lib.whisper_full_parallel.argtypes = [whisper_context_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.whisper_full_parallel.restype = ctypes.c_int


def whisper_full_n_segments(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_full_n_segments(ctx)

_lib.whisper_full_n_segments.argtypes = [whisper_context_p]
_lib.whisper_full_n_segments.restype = ctypes.c_int


def whisper_full_n_segments_from_state(state: whisper_state_p) -> ctypes.c_int:
    return _lib.whisper_full_n_segments_from_state(state)

_lib.whisper_full_n_segments_from_state.argtypes = [whisper_state_p]
_lib.whisper_full_n_segments_from_state.restype = ctypes.c_int


def whisper_full_lang_id(ctx: whisper_context_p) -> ctypes.c_int:
    return _lib.whisper_full_lang_id(ctx)

_lib.whisper_full_lang_id.argtypes = [whisper_context_p]
_lib.whisper_full_lang_id.restype = ctypes.c_int


def whisper_full_lang_id_from_state(state: whisper_state_p) -> ctypes.c_int:
    return _lib.whisper_full_lang_id_from_state(state)

_lib.whisper_full_lang_id_from_state.argtypes = [whisper_state_p]
_lib.whisper_full_lang_id_from_state.restype = ctypes.c_int


def whisper_full_get_segment_t0(ctx: whisper_context_p,
    i_segment: ctypes.c_int) -> ctypes.c_int64:
    return _lib.whisper_full_get_segment_t0(ctx, i_segment)

_lib.whisper_full_get_segment_t0.argtypes = [whisper_context_p, ctypes.c_int]
_lib.whisper_full_get_segment_t0.restype = ctypes.c_int64


def whisper_full_get_segment_t0_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int) -> ctypes.c_int64:
    return _lib.whisper_full_get_segment_t0_from_state(state, i_segment)

_lib.whisper_full_get_segment_t0_from_state.argtypes = [whisper_state_p, ctypes.c_int]
_lib.whisper_full_get_segment_t0_from_state.restype = ctypes.c_int64


def whisper_full_get_segment_t1(ctx: whisper_context_p,
    i_segment: ctypes.c_int) -> ctypes.c_int64:
    return _lib.whisper_full_get_segment_t1(ctx, i_segment)

_lib.whisper_full_get_segment_t1.argtypes = [whisper_context_p, ctypes.c_int]
_lib.whisper_full_get_segment_t1.restype = ctypes.c_int64


def whisper_full_get_segment_t1_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int) -> ctypes.c_int64:
    return _lib.whisper_full_get_segment_t1_from_state(state, i_segment)

_lib.whisper_full_get_segment_t1_from_state.argtypes = [whisper_state_p, ctypes.c_int]
_lib.whisper_full_get_segment_t1_from_state.restype = ctypes.c_int64


def whisper_full_get_segment_text(ctx: whisper_context_p,
    i_segment: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_full_get_segment_text(ctx, i_segment)

_lib.whisper_full_get_segment_text.argtypes = [whisper_context_p, ctypes.c_int]
_lib.whisper_full_get_segment_text.restype = ctypes.c_char_p


def whisper_full_get_segment_text_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_full_get_segment_text_from_state(state, i_segment)

_lib.whisper_full_get_segment_text_from_state.argtypes = [whisper_state_p, ctypes.c_int]
_lib.whisper_full_get_segment_text_from_state.restype = ctypes.c_char_p


def whisper_full_n_tokens(ctx: whisper_context_p,
    i_segment: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_full_n_tokens(ctx, i_segment)

_lib.whisper_full_n_tokens.argtypes = [whisper_context_p, ctypes.c_int]
_lib.whisper_full_n_tokens.restype = ctypes.c_int


def whisper_full_n_tokens_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_full_n_tokens_from_state(state, i_segment)

_lib.whisper_full_n_tokens_from_state.argtypes = [whisper_state_p, ctypes.c_int]
_lib.whisper_full_n_tokens_from_state.restype = ctypes.c_int


def whisper_full_get_token_text(ctx: whisper_context_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_full_get_token_text(ctx, i_segment, i_token)

_lib.whisper_full_get_token_text.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_text.restype = ctypes.c_char_p


def whisper_full_get_token_text_from_state(ctx: whisper_context_p,
    state: whisper_state_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_full_get_token_text_from_state(ctx, state, i_segment, i_token)

_lib.whisper_full_get_token_text_from_state.argtypes = [whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_text_from_state.restype = ctypes.c_char_p


def whisper_full_get_token_id(ctx: whisper_context_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> whisper_token:
    return _lib.whisper_full_get_token_id(ctx, i_segment, i_token)

_lib.whisper_full_get_token_id.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_id.restype = whisper_token


def whisper_full_get_token_id_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> whisper_token:
    return _lib.whisper_full_get_token_id_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_id_from_state.argtypes = [whisper_state_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_id_from_state.restype = whisper_token


def whisper_full_get_token_data(ctx: whisper_context_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> whisper_token_data:
    return _lib.whisper_full_get_token_data(ctx, i_segment, i_token)

_lib.whisper_full_get_token_data.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_data.restype = whisper_token_data


def whisper_full_get_token_data_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> whisper_token_data:
    return _lib.whisper_full_get_token_data_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_data_from_state.argtypes = [whisper_state_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_data_from_state.restype = whisper_token_data


def whisper_full_get_token_p(ctx: whisper_context_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> ctypes.c_float:
    return _lib.whisper_full_get_token_p(ctx, i_segment, i_token)

_lib.whisper_full_get_token_p.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_p.restype = ctypes.c_float


def whisper_full_get_token_p_from_state(state: whisper_state_p,
    i_segment: ctypes.c_int,
    i_token: ctypes.c_int) -> ctypes.c_float:
    return _lib.whisper_full_get_token_p_from_state(state, i_segment, i_token)

_lib.whisper_full_get_token_p_from_state.argtypes = [whisper_state_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_p_from_state.restype = ctypes.c_float


def whisper_bench_memcpy(n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_bench_memcpy(n_threads)

_lib.whisper_bench_memcpy.argtypes = [ctypes.c_int]
_lib.whisper_bench_memcpy.restype = ctypes.c_int


def whisper_bench_memcpy_str(n_threads: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_bench_memcpy_str(n_threads)

_lib.whisper_bench_memcpy_str.argtypes = [ctypes.c_int]
_lib.whisper_bench_memcpy_str.restype = ctypes.c_char_p


def whisper_bench_ggml_mul_mat(n_threads: ctypes.c_int) -> ctypes.c_int:
    return _lib.whisper_bench_ggml_mul_mat(n_threads)

_lib.whisper_bench_ggml_mul_mat.argtypes = [ctypes.c_int]
_lib.whisper_bench_ggml_mul_mat.restype = ctypes.c_int


def whisper_bench_ggml_mul_mat_str(n_threads: ctypes.c_int) -> ctypes.c_char_p:
    return _lib.whisper_bench_ggml_mul_mat_str(n_threads)

_lib.whisper_bench_ggml_mul_mat_str.argtypes = [ctypes.c_int]
_lib.whisper_bench_ggml_mul_mat_str.restype = ctypes.c_char_p

