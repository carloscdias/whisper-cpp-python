from skbuild import setup
import skbuild.constants
import shutil

from pathlib import Path

from pycparser import c_ast, parse_file

FILE = '''# auto-generated file
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
        lib_ext = ".dylib"
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

'''

DEFAULT_TYPE = 'ctypes.c_void_p'

class WhisperCppFileGen():
    T = '    '
    types = {
        'bool': 'ctypes.c_bool',
        'int': 'ctypes.c_int',
        'int64_t': 'ctypes.c_int64',
        'size_t': 'ctypes.c_size_t',
        'float': 'ctypes.c_float',
        'char': 'ctypes.c_char',
        'void': '',
    }

    replace = {
        'ctypes.POINTER(ctypes.c_char)': 'ctypes.c_char_p',
        'ctypes.POINTER()': 'ctypes.c_void_p',
    }

    @staticmethod
    def get_nested_type(node, ignore = {c_ast.Decl, c_ast.Typename}):
        typ = type(node)
        if typ in ignore:
            return WhisperCppFileGen.get_nested_type(node.type)
        return typ

    def __init__(self, filename, fake_libc = 'vendor/pycparser/utils/fake_libc_include'):
        self.ast = parse_file(filename, use_cpp=True, cpp_args=['-E', f'-I{fake_libc}'], cpp_path='gcc')
        self.blocks = []
        self._output = None
        self._process()

    def _process(self):
        for node in self.ast:
            to_pop = None
            typ = WhisperCppFileGen.get_nested_type(node)
            if typ == c_ast.FuncDecl:
                self.format_function(node)
            elif typ == c_ast.Struct:
                self.format_ctypes_structure(node)
            elif typ == c_ast.TypeDecl:
                self.format_ctypes_defs(node)
            elif typ == c_ast.Typedef:
                self.format_ctypes_defs(node)
        self._output = FILE
        while len(self.blocks) > 0:
            self._output += '\n\n' + self.blocks.pop(0)

    def print(self):
        print(self._output)

    def output(self, filename):
        with open(filename, 'w') as f:
            f.write(self._output)

    def format_ctypes_defs(self, node):
        typ = type(node)
        if typ != c_ast.Typedef or node.name in self.types or not node.name.startswith('whisper'):
            return
        t_type = self.get_ctypes_type(node.type)
        if node.name == t_type:
            return
        t_def = f'{node.name} = {t_type}'
        self.types[node.name] = node.name
        self.blocks.append(t_def)

    def format_ctypes_structure(self, node, cls_name = ''):
        typ = type(node)
        while typ in {c_ast.Decl, c_ast.TypeDecl, c_ast.Typedef}:
            node = node.type
            typ = type(node)
        if typ != c_ast.Struct:
            return
        cls_name = cls_name if cls_name else node.name
        if cls_name in self.types:
            return
        self.types[cls_name] = cls_name
        if not node.decls:
            cls = f'{cls_name}_p = ctypes.c_void_p'
            self.replace[f'ctypes.POINTER({cls_name})'] = f'{cls_name}_p'
        else:
            cls = f'class {cls_name}(ctypes.Structure):\n{self.T}_fields_ = [\n{self.T*2}'
            cls += f'\n{self.T*2}'.join(self.format_ctypes_structure_fields(node.decls))
            cls += f'\n{self.T}]'
        self.blocks.append(cls)

    def format_ctypes_structure_fields(self, fields):
        fields_txt = []
        for f in fields:
            typ = self.get_ctypes_type(f)
            typ = self.replace[typ] if typ in self.replace else typ
            fields_txt.append(f'("{f.name}", {typ}),')
        return fields_txt

    def get_ctypes_type(self, node, name=''):
        if node is None:
            return ''
        typ = type(node)
        if typ == c_ast.Typename or typ == c_ast.Decl:
            return self.get_ctypes_type(node.type, node.name)
        if typ == c_ast.TypeDecl:
            return self.get_ctypes_type(node.type, node.declname)
        if typ == c_ast.FuncDecl:
            params = [self.get_ctypes_type(t) for t in node.args.params]
            params = [self.replace[t] if t in self.replace else t for t in params]
            ret = self.get_ctypes_type(node.type)
            ret = ret if ret else 'None'
            return 'ctypes.CFUNCTYPE(' + ret + (', ' if len(params) > 0 else '') + ', '.join(params) + ')'
        if typ == c_ast.PtrDecl:
            if node.type and type(node.type) == c_ast.FuncDecl:
                return self.get_ctypes_type(node.type)
            return 'ctypes.POINTER(' + self.get_ctypes_type(node.type) + ')'
        if typ == c_ast.Struct:
            name = node.name if node.name else name
            self.format_ctypes_structure(node, name)
            return name
        if typ == c_ast.Enum:
            return 'ctypes.c_int'
        return self.types.get(''.join(node.names), DEFAULT_TYPE)

    def get_function_args(self, args):
        names = []
        types = []
        if args is None or args.params is None:
            return names, types
        for p in args.params:
            names.append(p.name)
            typ = self.get_ctypes_type(p)
            typ = self.replace[typ] if typ in self.replace else typ
            types.append(typ)
        return list(filter(lambda x: x, names)), types

    def format_function(self, node):
        name = node.name
        typ = self.get_ctypes_type(node.type.type)
        ret = self.replace[typ] if typ in self.replace else typ
        args, types = self.get_function_args(node.type.args)
        args_typed = f',\n{self.T}'.join([f'{n}: {t}' for n, t in zip(args, types)])
        all_args = ', '.join(args)
        all_types = ', '.join(types)
        ret_f = f' -> {ret}:\n{self.T}return ' if ret else f':\n{self.T}'
        pyfunction = f'def {name}({args_typed}){ret_f}_lib.{name}({all_args})'
        all_block = f'{pyfunction}\n\n_lib.{name}.argtypes = [{all_types}]\n_lib.{name}.restype = {ret if ret else "None"}\n'
        self.blocks.append(all_block)


if __name__ == '__main__':
    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")

    # Copy built C-extensions back to the project.
    setup(
        name="whisper_cpp_python",
        description="A Python wrapper for whisper.cpp",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version="0.2.1",
        author="Carlos Cardoso Dias",
        author_email="carlosdias.dev@gmail.com",
        license="MIT",
        package_dir={"whisper_cpp_python": "whisper_cpp_python", "whisper_cpp_python.server": "whisper_cpp_python/server"},
        packages=["whisper_cpp_python", "whisper_cpp_python.server"],
        install_requires=[
            "librosa>=0.10.0.post2",
            "typing-extensions>=4.5.0",
        ],
        extras_require={
            "server": ["uvicorn>=0.21.1", "fastapi>=0.95.0", "sse-starlette>=1.3.3", "python-multipart>=0.0.6"],
        },
        python_requires=">=3.9",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        include_package_data=True,
        cmake_process_manifest_hook=lambda x: list(filter(lambda y: not y.endswith('.h'), x)),
    )

    # generate whisper_cpp.py with whisper.h header file
    dest_dir = Path("whisper_cpp_python")
    c_header_file = "vendor/whisper.cpp/whisper.h"
    file_gen = WhisperCppFileGen(c_header_file)
    file_gen.output(dest_dir / "whisper_cpp.py")
