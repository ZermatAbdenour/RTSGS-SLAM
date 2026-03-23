#version 460 core
layout(location = 0) in vec3 aPos;

layout(location = 2) in vec4 iModel0;
layout(location = 3) in vec4 iModel1;
layout(location = 4) in vec4 iModel2;
layout(location = 5) in vec4 iModel3;
layout(location = 6) in vec3 iColor;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 Color;

void main() {
    mat4 model = mat4(iModel0, iModel1, iModel2, iModel3);
    vec4 world = model * vec4(aPos, 1.0);
    gl_Position = u_projection * u_view * world;
    Color = iColor;
}
