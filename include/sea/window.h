// Copyright (c) 2020, Danilo Peixoto. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SEA_WINDOW_H
#define SEA_WINDOW_H

#include <sea/global.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

SEA_NAMESPACE_BEGIN

struct Scene;
struct Camera;
struct Renderer;
struct Denoiser;

struct Window {
    const GLchar * title;
    GLuint width;
    GLuint height;
	GLuint device_id;

    GLFWwindow * handle;

    Scene * scene;
    Camera * camera;
    Renderer * renderer;

	Denoiser * denoiser;

    GLint mouse_button;
    glm::vec2 mouse_position;

    glm::vec3 translation;
    glm::vec3 rotation;

    GLboolean progressive_rendering;

    GLuint program;
    GLuint texture;
    GLuint vao;
    GLuint vbo;
};

__host__ GLboolean window_load_shader(const GLchar *, GLenum, GLuint &);
__host__ GLboolean window_create_shader(const GLchar *, GLuint &);
__host__ GLvoid window_delete_shader(GLuint &);

__host__ GLvoid window_initialize(Window *);
__host__ GLvoid window_render(Window *);

__host__ GLvoid window_reset_view(Window *);
__host__ GLvoid window_move_view(Window *, GLfloat, GLfloat, GLfloat);
__host__ GLvoid window_rotate_view(Window *, GLfloat, GLfloat, GLfloat);

__host__ GLvoid window_resize(Window *, GLuint, GLuint);
__host__ GLvoid window_keyboard(Window *, GLuint, GLuint);
__host__ GLvoid window_mouse_button(Window *, GLuint, GLuint);
__host__ GLvoid window_mouse_cursor(Window *, GLfloat, GLfloat);
__host__ GLvoid window_mouse_scroll(Window *, GLfloat, GLfloat);

__host__ Window * window_create(const GLchar *, GLuint, GLuint, GLuint);
__host__ GLvoid window_delete(Window *);

__host__ GLvoid window_show(Window *);
__host__ GLvoid window_close(Window *);

SEA_NAMESPACE_END

#endif