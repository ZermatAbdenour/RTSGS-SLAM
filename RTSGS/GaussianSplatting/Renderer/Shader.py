from pathlib import Path
from OpenGL.GL import *


class Shader:
    def __init__(self, vertex_shader_path: str, fragment_shader_path: str, geometry_shader_path: str | None = None):
        vertex_src = self._read_file(vertex_shader_path)
        fragment_src = self._read_file(fragment_shader_path)
        geometry_src = self._read_file(geometry_shader_path) if geometry_shader_path else None
        self.program = self._create_program(vertex_src, fragment_src, geometry_src)

    def use(self):
        glUseProgram(self.program)

    @staticmethod
    def _compile_shader(source: str, shader_type) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compile failed:\n{log}")
        return shader

    @classmethod
    def _create_program(cls, vertex_src: str, fragment_src: str, geometry_src: str | None = None) -> int:
        vs = cls._compile_shader(vertex_src, GL_VERTEX_SHADER)
        fs = cls._compile_shader(fragment_src, GL_FRAGMENT_SHADER)
        gs = cls._compile_shader(geometry_src, GL_GEOMETRY_SHADER) if geometry_src is not None else None

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        if gs is not None:
            glAttachShader(program, gs)
        glLinkProgram(program)

        status = glGetProgramiv(program, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            glDeleteProgram(program)
            raise RuntimeError(f"Program link failed:\n{log}")

        glDetachShader(program, vs)
        glDetachShader(program, fs)
        glDeleteShader(vs)
        glDeleteShader(fs)
        if gs is not None:
            glDetachShader(program, gs)
            glDeleteShader(gs)
        return program

    @staticmethod
    def _read_file(path: str) -> str:
        base_dir = Path(__file__).parent
        with open((base_dir/path).resolve(), "r", encoding="utf-8") as f:
            return f.read()
