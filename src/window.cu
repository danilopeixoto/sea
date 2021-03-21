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

#include <sea/window.h>
#include <sea/bvh.h>
#include <sea/scene.h>
#include <sea/renderer.h>
#include <sea/denoiser.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <glm/trigonometric.hpp>

SEA_NAMESPACE_BEGIN

__host__ GLboolean window_load_shader(const GLchar * filename, GLenum type, GLuint & id) {
    std::ifstream file(filename);

    if (!file.is_open())
        return false;

    std::stringstream buffer;
    std::string source;

    buffer << file.rdbuf();
    source = buffer.str();

    file.close();

    const GLchar * code = source.c_str();
    GLuint shader_id = glCreateShader(type);

    glShaderSource(shader_id, 1, &code, nullptr);
    glCompileShader(shader_id);

    GLint status;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);

    if (!status)
        return false;

    id = shader_id;

    return true;
}
__host__ GLboolean window_create_shader(const GLchar * shader, GLuint & id) {
    std::string name(shader);
    GLuint vertex_shader, fragment_shader;

    if (!window_load_shader((name + ".vert").c_str(), GL_VERTEX_SHADER, vertex_shader) ||
        !window_load_shader((name + ".frag").c_str(), GL_FRAGMENT_SHADER, fragment_shader))
        return false;

    id = glCreateProgram();

    glAttachShader(id, vertex_shader);
    glAttachShader(id, fragment_shader);

    glLinkProgram(id);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint status;
    glGetProgramiv(id, GL_LINK_STATUS, &status);

    return status;
}
__host__ GLvoid window_delete_shader(GLuint & id) {
    glDeleteProgram(id);
}

__host__ GLvoid window_create_screen(Window * window) {
    if (window_create_shader("res/shaders/texture", window->program)) {
        glUseProgram(window->program);
        glUniform1i(glGetUniformLocation(window->program, "texture"), 0);

        glGenTextures(1, &window->texture);
        glBindTexture(GL_TEXTURE_2D, window->texture);

        glActiveTexture(GL_TEXTURE0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window->width, window->height, 0, GL_RGB, GL_FLOAT, 0);

        float vertices[] = {
            -1.0f, -1.0f, 0.0f, 1.0f,
             1.0f, -1.0f, 1.0f, 1.0f,
            -1.0f,  1.0f, 0.0f, 0.0f,

            -1.0f,  1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 0.0f
        };

        glGenVertexArrays(1, &window->vao);
        glBindVertexArray(window->vao);

        glGenBuffers(1, &window->vbo);
        glBindBuffer(GL_ARRAY_BUFFER, window->vbo);

        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, false, 4 * sizeof(GLfloat), nullptr);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, false, 4 * sizeof(GLfloat), (GLvoid *)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);
    }
}
__host__ GLvoid window_delete_screen(Window * window) {
    glDeleteBuffers(1, &window->vbo);
    glDeleteVertexArrays(1, &window->vao);
    glDeleteTextures(1, &window->texture);

    window_delete_shader(window->program);
}
__host__ GLvoid window_resize_screen(Window * window) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window->width, window->height, 0, GL_RGB, GL_FLOAT, 0);
}
__host__ GLvoid window_render_screen(Window * window) {
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

__host__ GLvoid window_initialize(Window * window) {
	window_create_screen(window);

	BSDF * white_diffuse_material;
	cudaMallocManaged(&white_diffuse_material, sizeof(BSDF));

	white_diffuse_material->type = BSDF_DIFFUSE;
	white_diffuse_material->intensity = 1.0f;
	white_diffuse_material->color = glm::vec3(1.0f, 1.0f, 1.0f);
	white_diffuse_material->index_of_refraction = 1.0f;

	size_t scene_version = 0;
	std::string scene_name("res/meshes/cornell_box");
	std::string scene_path = scene_name + std::to_string(scene_version) + ".obj";

	window->scene = scene_load(scene_path.c_str(), white_diffuse_material);

	if (window->scene == nullptr) {
		std::cerr << "Error: cannot load scene from " << scene_path << " path." << std::endl;

		cudaFree(white_diffuse_material);
		glfwSetWindowShouldClose(window->handle, GLFW_TRUE);

		return;
	}

	BSDF * light_material;
	cudaMallocManaged(&light_material, sizeof(BSDF));

	light_material->type = BSDF_LIGHT;
	light_material->intensity = 20.0f;
	light_material->color = glm::vec3(1.0f, 0.65f, 0.3f);
	light_material->index_of_refraction = 1.0f;

	BSDF * red_diffuse_material;
	cudaMallocManaged(&red_diffuse_material, sizeof(BSDF));

	red_diffuse_material->type = BSDF_DIFFUSE;
	red_diffuse_material->intensity = 1.0f;
	red_diffuse_material->color = glm::vec3(0.65f, 0.065f, 0.05f);
	red_diffuse_material->index_of_refraction = 1.0f;

	BSDF * green_diffuse_material;
	cudaMallocManaged(&green_diffuse_material, sizeof(BSDF));

	green_diffuse_material->type = BSDF_DIFFUSE;
	green_diffuse_material->intensity = 1.0f;
	green_diffuse_material->color = glm::vec3(0.14f, 0.45f, 0.2f);
	green_diffuse_material->index_of_refraction = 1.0f;

	BSDF * glass_material;
	cudaMallocManaged(&glass_material, sizeof(BSDF));

	glass_material->type = BSDF_GLASS;
	glass_material->intensity = 1.0f;
	glass_material->color = glm::vec3(1.0f, 1.0f, 1.0f);
	glass_material->index_of_refraction = 1.55f;

    Triangle * shape = nullptr;

    switch (scene_version) {
    case 0:
        light_material->intensity = 20.0f;

        shape = window->scene->triangle_list[10];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[11];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[8];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[9];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[6];
        shape->bsdf = green_diffuse_material;

        shape = window->scene->triangle_list[7];
        shape->bsdf = green_diffuse_material;

        break;

    case 1:
        light_material->intensity = 20.0f;

        shape = window->scene->triangle_list[528];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[529];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[530];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[531];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[532];
        shape->bsdf = green_diffuse_material;

        shape = window->scene->triangle_list[533];
        shape->bsdf = green_diffuse_material;

        for (size_t i = 0; i <= 263; i++) {
            shape = window->scene->triangle_list[i];
            shape->bsdf = glass_material;
        }

        break;

    case 2:
        light_material->intensity = 1.0f;

        shape = window->scene->triangle_list[8];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[9];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[0];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[1];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[2];
        shape->bsdf = green_diffuse_material;

        shape = window->scene->triangle_list[3];
        shape->bsdf = green_diffuse_material;

        break;

    case 3:
        light_material->intensity = 1.0f;

        shape = window->scene->triangle_list[1886];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[1887];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[1894];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[1895];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[1892];
        shape->bsdf = green_diffuse_material;

        shape = window->scene->triangle_list[1893];
        shape->bsdf = green_diffuse_material;

        break;

    case 4:
        light_material->intensity = 1.0f;

        shape = window->scene->triangle_list[761];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[762];
        shape->bsdf = light_material;

        shape = window->scene->triangle_list[769];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[770];
        shape->bsdf = red_diffuse_material;

        shape = window->scene->triangle_list[768];
        shape->bsdf = green_diffuse_material;

        shape = window->scene->triangle_list[767];
        shape->bsdf = green_diffuse_material;

        break;

    default:
        break;
    }

    bvh_tree_build(window->scene, 1);
    scene_group_lights(window->scene);

    window->camera = camera_create(
        glm::radians(45.0f), 15.0f, 0.0f, (GLfloat)window->width, (GLfloat)window->height);

	window->renderer = renderer_create(window->width, window->height, 4, 1, 1, 2.2f);
	window->denoiser = denoiser_create(LDR);

	denoiser_update(window->denoiser, window->renderer);
}
__host__ GLvoid window_render(Window * window) {
    camera_update_view_matrix(window->camera, window->translation, window->rotation);

    if (!window->progressive_rendering) {
		window->renderer->progressive_time = 0;
        window->progressive_rendering = true;
    }

	curandState * random_state;
	cudaMallocManaged(&random_state, window->width * window->height * sizeof(curandState));

	GLfloat t = (GLfloat)window->renderer->progressive_time;
    window->renderer->accumulated = t / (t + 1.0f);

	dim3 block_size, grid_size;

	renderer_compute_kernel_parameters(window->renderer, renderer_initialize, block_size, grid_size);
	renderer_initialize<<<grid_size, block_size>>>(window->renderer, random_state);
	cudaDeviceSynchronize();

	renderer_compute_kernel_parameters(window->renderer, renderer_render_passes, block_size, grid_size);
	renderer_render_passes<<<grid_size, block_size>>>(window->scene, window->camera, window->renderer, random_state);
	cudaDeviceSynchronize();

	renderer_compute_kernel_parameters(window->renderer, renderer_render, block_size, grid_size);
    renderer_render<<<grid_size, block_size>>>(window->scene, window->camera, window->renderer, random_state);
	cudaDeviceSynchronize();

	cudaFree(random_state);

	denoiser_denoise(window->denoiser, window->renderer, 1.0f);

	renderer_compute_kernel_parameters(window->renderer, renderer_accumulate, block_size, grid_size);
	renderer_accumulate<<<grid_size, block_size>>>(window->renderer);
	cudaDeviceSynchronize();

    window->renderer->time++;
    window->renderer->progressive_time++;

    GLfloat * data = window->renderer->image->data;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window->width, window->height, 0, GL_RGB, GL_FLOAT, data);

    window_render_screen(window);
}

__host__ GLvoid window_reset_view(Window * window) {
    window->translation.x = 0.0f;
    window->translation.y = 0.0f;
    window->translation.z = 0.0f;

    window->rotation.x = 0.0f;
    window->rotation.y = 0.0f;
    window->rotation.z = 0.0f;
}
__host__ GLvoid window_move_view(Window * window, GLfloat dx, GLfloat dy, GLfloat dz) {
    window->translation.x += dx;
    window->translation.y += dy;
    window->translation.z += dz;
}
__host__ GLvoid window_rotate_view(Window * window, GLfloat ax, GLfloat ay, GLfloat az) {
    window->rotation.x += ax;
    window->rotation.y += ay;
    window->rotation.z += az;
}

__host__ GLvoid window_resize(Window * window, GLuint width, GLuint height) {
    window->width = width;
    window->height = height;

    glViewport(0, 0, width, height);

    window_resize_screen(window);

	renderer_update(window->renderer, window->camera, width, height);
	denoiser_update(window->denoiser, window->renderer);

    window->progressive_rendering = false;
}
__host__ GLvoid window_keyboard(Window * window, GLuint key, GLuint action) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_D:
            window_reset_view(window);
            window->progressive_rendering = false;
            break;
        case GLFW_KEY_G:
            window->renderer->maximum_depth ^= 9;
            window->progressive_rendering = false;
            break;
		case GLFW_KEY_A:
			window->renderer->pass = 1;
			window->renderer->image = window->renderer->albedo_pass;
			window->progressive_rendering = false;
			break;
		case GLFW_KEY_N:
			window->renderer->pass = 2;
			window->renderer->image = window->renderer->normal_pass;
			window->progressive_rendering = false;
			break;
		case GLFW_KEY_R:
			window->renderer->pass = 0;
			window->renderer->image = window->renderer->radiance_pass;
			window->progressive_rendering = false;
			break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window->handle, GLFW_TRUE);
            break;
        default:
            break;
        }
    }
}
__host__ GLvoid window_mouse_button(Window * window, GLuint button, GLuint action) {
    window->mouse_button = action == GLFW_PRESS ? (GLint)button : -1;
}
__host__ GLvoid window_mouse_cursor(Window * window, GLfloat dx, GLfloat dy) {
    switch (window->mouse_button) {
    case GLFW_MOUSE_BUTTON_LEFT:
        window_rotate_view(window, -0.005f * dx, -0.005f * dy, 0.0f);
        window->progressive_rendering = false;
        break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
        window_move_view(window, -0.05f * dx, 0.05f * dy, 0.0f);
        window->progressive_rendering = false;
        break;
    case GLFW_MOUSE_BUTTON_RIGHT:
        window_move_view(window, 0.0f, 0.0f, -0.25f * (dx + dy));
        window->progressive_rendering = false;
        break;
    default:
        break;
    }
}
__host__ GLvoid window_mouse_scroll(Window * window, GLfloat dx, GLfloat dy) {
    window_move_view(window, 0.0f, 0.0f, -5.0f * (dx + dy));
    window->progressive_rendering = false;
}

__host__ Window * window_create(const GLchar * title, GLuint width, GLuint height, GLuint device_id) {
    Window * window = new Window;

    window->handle = nullptr;
	
    window->scene = nullptr;
    window->camera = nullptr;
    window->renderer = nullptr;

    window->title = title;
    window->width = width;
    window->height = height;
	window->device_id = device_id;

    window->mouse_button = -1;
    window->mouse_position.x = 0.0f;
    window->mouse_position.y = 0.0f;

    window_reset_view(window);

    window->progressive_rendering = true;

    return window;
}
__host__ GLvoid window_delete(Window * window) {
    if (window) {
		scene_delete(window->scene);
        camera_delete(window->camera);
        renderer_delete(window->renderer);
		denoiser_delete(window->denoiser);

        delete window;
        window = nullptr;
    }
}

__host__ GLvoid window_show(Window * window) {
	if (!glfwInit())
        return;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window->handle = glfwCreateWindow(window->width, window->height, window->title, nullptr, nullptr);
    glfwSetWindowSizeLimits(window->handle, 100, 100, GLFW_DONT_CARE, GLFW_DONT_CARE);

    if (window->handle) {
        glfwSetFramebufferSizeCallback(window->handle,
            [](GLFWwindow *  handle, int width, int height) {
            window_resize((Window *)glfwGetWindowUserPointer(handle), width, height);
        });

        glfwSetKeyCallback(window->handle,
            [](GLFWwindow * handle, int key, int code, int action, int modifier) {
            window_keyboard((Window *)glfwGetWindowUserPointer(handle), key, action);
        });

        glfwSetMouseButtonCallback(window->handle,
            [](GLFWwindow * handle, int button, int action, int modifier) {
            window_mouse_button((Window *)glfwGetWindowUserPointer(handle), button, action);
        });

        glfwSetCursorPosCallback(window->handle,
            [](GLFWwindow * handle, double x, double y) {
            Window * window = (Window *)glfwGetWindowUserPointer(handle);

            window_mouse_cursor(window,
                (GLfloat)x - window->mouse_position.x,
                (GLfloat)y - window->mouse_position.y);

            window->mouse_position.x = (GLfloat)x;
            window->mouse_position.y = (GLfloat)y;
        });

        glfwSetScrollCallback(window->handle,
            [](GLFWwindow * handle, double dx, double dy) {
            window_mouse_scroll((Window *)glfwGetWindowUserPointer(handle), (GLfloat)dx, (GLfloat)dy);
        });

        glfwSetWindowUserPointer(window->handle, window);
        glfwMakeContextCurrent(window->handle);
        glfwSwapInterval(1);

        GLFWmonitor * monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode * mode = glfwGetVideoMode(monitor);

        GLuint x = (mode->width - window->width) / 2;
        GLuint y = (mode->height - window->height) / 2;

        glfwSetWindowPos(window->handle, x, y);

        glewExperimental = GL_TRUE;
		
		if (glewInit() == GLEW_OK) {
			window_initialize(window);

			while (!glfwWindowShouldClose(window->handle)) {
				window_render(window);

                glfwSwapBuffers(window->handle);
                glfwPollEvents();
            }
        }
    }

	window_close(window);
}
__host__ GLvoid window_close(Window * window) {
    if (window->handle)
		glfwDestroyWindow(window->handle);

    glfwTerminate();
}

SEA_NAMESPACE_END