# whisper-cpp-python

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/carloscdias/whisper-cpp-python/build_and_publish.yml)
![GitHub](https://img.shields.io/github/license/carloscdias/whisper-cpp-python)
![PyPI](https://img.shields.io/pypi/v/whisper-cpp-python)

whisper-cpp-python is a Python module inspired by [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) that provides a Python interface to the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) model.
This module automatically parses the C++ header file of the project during building time, generating the corresponding Python bindings.

## Installation

To install the module, you can use pip:

```bash
pip install whisper-cpp-python
```

## Usage

To use the module, you need to create an instance of the `Whisper` class, passing the path to the model file as a parameter. Then, you can call the `transcribe` method to transcribe a given audio file.

```python
from whisper_cpp_python import Whisper

model = Whisper('vendor/whisper.cpp/models/ggml-tiny.bin')
print(model.transcribe('vendor/whisper.cpp/samples/jfk.wav'))
```

In the example above, the module transcribes the `jfk.wav` audio file using the `ggml-tiny` model.


All interfaces provided by `whisper.h` are available in python. The following example
show how to pass a custom progress_callback function to the model.

```python
from whisper_cpp_python import Whisper
from whisper_cpp_python.whisper_cpp import whisper_progress_callback

def callback(ctx, state, i, p):
    print(i)

model = Whisper('../quantized_models/whisper/models/ggml-tiny.bin')
model.params.progress_callback = whisper_progress_callback(callback)

print(model.transcribe('vendor/whisper.cpp/samples/jfk.wav'))
```

## License

whisper-cpp-python is released under the MIT License. See [LICENSE](LICENSE) for details.
