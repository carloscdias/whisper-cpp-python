from whisper_cpp_python import Whisper
from whisper_cpp_python.whisper_cpp import whisper_progress_callback

#@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
def callback(ctx, state, i, p):
    print('hehehe')
    print(i)

model = Whisper('vendor/whisper.cpp/models/ggml-tiny.bin')
model.params.progress_callback = whisper_progress_callback(callback)
model.params.print_progress = False
model.params.print_special = False

print(model.transcribe('vendor/whisper.cpp/samples/jfk.wav'))
