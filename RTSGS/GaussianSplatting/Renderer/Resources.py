from .Shader import Shader
from .Mesh import Mesh
import numpy as np
#shaders
simple_shader = None
line_shader = None
camera_instance_shader = None
camera_point_shader = None


#primetives
quad = None

def init_resources():
    #shaders
    global simple_shader
    simple_shader = Shader('./Shaders/point_vertex.glsl','./Shaders/point_fragment.glsl')
    global line_shader
    line_shader = Shader('./Shaders/line_vertex.glsl','./Shaders/line_fragment.glsl')
    global camera_instance_shader
    camera_instance_shader = Shader('./Shaders/camera_instance_vertex.glsl','./Shaders/line_fragment.glsl')
    global camera_point_shader
    camera_point_shader = Shader(
        './Shaders/camera_point_vertex.glsl',
        './Shaders/line_fragment.glsl',
        geometry_shader_path='./Shaders/camera_point_geometry.glsl',
    )

    #primitives
    global quad
    quad = Mesh(
        np.array([
            -0.5,-0.5,0.0,
            -0.5,0.5,0.0,
            0.5,0.5,0.0,
            0.5,-0.5,0.0
        ]).astype(np.float32),
        np.array([
            0,1,2,
            0,2,3
        ]).astype(np.float32)
    )