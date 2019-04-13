#version 330 core

in vec2 uvCoordinates;
out vec4 outputColor;

uniform sampler2D image;

void main() {
    outputColor = vec4(texture(image, uvCoordinates).rgb, 1.0f);
}