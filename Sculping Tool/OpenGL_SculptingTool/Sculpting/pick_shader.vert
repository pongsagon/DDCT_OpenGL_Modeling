#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 2) in vec3 pickingColor;

uniform mat4 MVP;
out vec3 passColor;

void main() {
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);
    passColor = pickingColor;
}
