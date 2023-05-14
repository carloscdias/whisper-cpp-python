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

To use the module, you need to create an instance of the `Whisper` class, passing the path to the model file as a parameter. Then, you can call the `transcribe` or `translate` method to transcribe or translate a given audio file.

### High-level API

The high-level API provides a simple managed interface through the `Wisper` class.

Below is a short example demonstrating how to use the high-level API to transcribe an mp3:

```python
>>> from whisper_cpp_python import Whisper
>>> whisper = Whisper(model_path="./models/ggml-tiny.bin")
>>> output = whisper.transcribe(open('samples/jfk.mp3'))
>>> print(output)
{'text': 'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.'}
>>> output = whisper.transcribe(open('samples/jfk.mp3'), response_format='verbose_json')
>>> print(output)
{
    'task': 'transcribe',
    'language': 'en',
    'duration': 11.0,
    'text': 'And so, my fellow Americans ask not what your country can do for you, ask what you can do for your country.',
    'segments': [{
        'id': 0,
        'seek': 0.0,
        'start': 0.0,
        'end': 10.98,
        'text': ' And so, my fellow Americans ask not what your country can do for you, ask what you can do for your country.',
        'tokens': [50364,
            400,
            370,
            11,
            452,
            7177,
            6280,
            1029,
            406,
            437,
            428,
            1941,
            393,
            360,
            337,
            291,
            11,
            1029,
            437,
            291,
            393,
            360,
            337,
            428,
            1941,
            13,
            50913],
       'temperature': 0.800000011920929,
       'avg_logprob': -0.3063158459133572,
       'compression_ratio': 2.4000000953674316,
       'no_speech_prob': 0.0,
       'transient': False
   }]
}
```

### Low-level API

All interfaces provided by `whisper.h` are available in python. The following example
show how to pass a custom `progress_callback` function to the model.

```python
from whisper_cpp_python import Whisper
from whisper_cpp_python.whisper_cpp import whisper_progress_callback

def callback(ctx, state, i, p):
    print(i)

model = Whisper('../quantized_models/whisper/models/ggml-tiny.bin')
model.params.progress_callback = whisper_progress_callback(callback)

print(model.transcribe('vendor/whisper.cpp/samples/jfk.wav'))
```

## Web Server

`whisper-cpp-python` offers a web server which aims to act as a drop-in replacement for the OpenAI API.
This allows you to use whisper.cpp compatible models with any OpenAI compatible client (language libraries, services, etc).

To install the server package and get started:

```bash
pip install whisper-cpp-python[server]
python3 -m whisper_cpp_python.server --model models/ggml-tiny.bin
```

Navigate to [http://localhost:8001/docs](http://localhost:8001/docs) to see the OpenAPI documentation.


## License

whisper-cpp-python is released under the MIT License. See [LICENSE](LICENSE) for details.
