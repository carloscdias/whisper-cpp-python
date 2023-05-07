import src.whisper_cpp as whisper_cpp
import ctypes
import librosa

class Whisper():
    WHISPER_SR = 16000

    def __init__(self, model_path):
        self.context = whisper_cpp.whisper_init_from_file(model_path.encode('utf-8'))
        self.params  = whisper_cpp.whisper_full_default_params(0)

    def transcribe(self, audio_file):
        data, sr = librosa.load(audio_file, sr=Whisper.WHISPER_SR)

        # run the inference
        result = whisper_cpp.whisper_full(ctypes.c_void_p(self.context), self.params, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data))

        if result != 0:
            print("Error: {}".format(result))
            exit(1)

        # print results from Python
        print("\nResults from Python:\n")
        n_segments = whisper_cpp.whisper_full_n_segments(ctypes.c_void_p(self.context))
        result = []
        for i in range(n_segments):
            t0  = whisper_cpp.whisper_full_get_segment_t0(ctypes.c_void_p(self.context), i)/100.0
            t1  = whisper_cpp.whisper_full_get_segment_t1(ctypes.c_void_p(self.context), i)/100.0
            txt = whisper_cpp.whisper_full_get_segment_text(ctypes.c_void_p(self.context), i).decode('utf-8')
            result.append((t0, t1, txt))
        return result

    def free(self):
        # free the memory
        whisper_cpp.whisper_free(ctypes.c_void_p(self.context))


