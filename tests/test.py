from whisper_cpp_python import Whisper

model = Whisper('vendor/whisper.cpp/models/ggml-tiny.bin')
print(model.transcribe('vendor/whisper.cpp/samples/jfk.wav'))
