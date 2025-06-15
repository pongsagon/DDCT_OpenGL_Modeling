#version 330 core
out vec4 FragColor;

void main() {
    int r = (gl_PrimitiveID & 0x000000FF) >>  0;
    int g = (gl_PrimitiveID & 0x0000FF00) >>  8;
    int b = (gl_PrimitiveID & 0x00FF0000) >> 16;
    FragColor = vec4(r, g, b, 255) / 255.0;
}