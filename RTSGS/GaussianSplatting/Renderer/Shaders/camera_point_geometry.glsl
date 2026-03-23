#version 460 core
layout(points) in;
layout(line_strip, max_vertices = 64) out;

in vec3 vWorldPos[];
in vec3 vColor[];
in vec3 vForward[];
in vec3 vUp[];
in vec3 vRight[];

out vec3 Color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_cam_scale;

void emit_edge(vec3 a, vec3 b, vec3 c) {
    Color = c;
    gl_Position = u_projection * u_view * vec4(a, 1.0);
    EmitVertex();
    gl_Position = u_projection * u_view * vec4(b, 1.0);
    EmitVertex();
    EndPrimitive();
}

vec3 build_world_point(vec3 center, vec3 right, vec3 up, vec3 forward, vec3 local) {
    return center + right * local.x + up * local.y + forward * local.z;
}

void main() {
    vec3 center = vWorldPos[0];
    vec3 col = vColor[0];
    vec3 forward = normalize(vForward[0]);
    vec3 up = normalize(vUp[0]);
    vec3 right = normalize(vRight[0]);

    float s = u_cam_scale;
    float rearW = s * 0.55;
    float rearH = s * 0.42;
    float frontW = s * 1.00;
    float frontH = s * 0.75;
    float rearZ = s * 0.28;
    float frontZ = s * 1.45;

    // Rear image plane (camera body side)
    vec3 r0 = build_world_point(center, right, up, forward, vec3(-rearW, -rearH, rearZ));
    vec3 r1 = build_world_point(center, right, up, forward, vec3( rearW, -rearH, rearZ));
    vec3 r2 = build_world_point(center, right, up, forward, vec3( rearW,  rearH, rearZ));
    vec3 r3 = build_world_point(center, right, up, forward, vec3(-rearW,  rearH, rearZ));

    // Front plane (frustum opening)
    vec3 f0 = build_world_point(center, right, up, forward, vec3(-frontW, -frontH, frontZ));
    vec3 f1 = build_world_point(center, right, up, forward, vec3( frontW, -frontH, frontZ));
    vec3 f2 = build_world_point(center, right, up, forward, vec3( frontW,  frontH, frontZ));
    vec3 f3 = build_world_point(center, right, up, forward, vec3(-frontW,  frontH, frontZ));

    // Body center and a tiny top marker to indicate "up" visually.
    vec3 bodyCenter = build_world_point(center, right, up, forward, vec3(0.0, 0.0, rearZ * 0.6));
    vec3 topMark = build_world_point(center, right, up, forward, vec3(0.0, frontH * 1.25, rearZ + (frontZ - rearZ) * 0.35));

    // Rear rectangle.
    emit_edge(r0, r1, col);
    emit_edge(r1, r2, col);
    emit_edge(r2, r3, col);
    emit_edge(r3, r0, col);

    // Front rectangle.
    emit_edge(f0, f1, col);
    emit_edge(f1, f2, col);
    emit_edge(f2, f3, col);
    emit_edge(f3, f0, col);

    // Connect rear to front (camera rails).
    emit_edge(r0, f0, col);
    emit_edge(r1, f1, col);
    emit_edge(r2, f2, col);
    emit_edge(r3, f3, col);

    // Body and top marker accents.
    emit_edge(bodyCenter, r0, col);
    emit_edge(bodyCenter, r1, col);
    emit_edge(bodyCenter, r2, col);
    emit_edge(bodyCenter, r3, col);
    emit_edge(r3, topMark, col);
    emit_edge(r2, topMark, col);
}
