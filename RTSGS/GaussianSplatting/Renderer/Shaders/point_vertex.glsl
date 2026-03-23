#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 u_view;
uniform mat4 u_projection;
out vec3 Color;
void main(){
    vec4 view_pos = u_view * vec4(aPos, 1.0);
    float distance = length(view_pos.xyz);
    
    gl_Position = u_projection * view_pos;
    gl_PointSize = 10.0 / distance;
    
    Color = aColor;
}
