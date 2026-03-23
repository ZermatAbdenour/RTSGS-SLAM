#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aForward;
layout(location = 3) in vec3 aUp;
layout(location = 4) in vec3 aRight;

out vec3 vWorldPos;
out vec3 vColor;
out vec3 vForward;
out vec3 vUp;
out vec3 vRight;

void main() {
    vWorldPos = aPos;
    vColor = aColor;
    vForward = aForward;
    vUp = aUp;
    vRight = aRight;
}
