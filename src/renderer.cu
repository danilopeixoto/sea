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

#include <sea/renderer.h>
#include <sea/utility.h>
#include <sea/scene.h>
#include <sea/math.h>

#include <glm/trigonometric.hpp>
#include <glm/vec2.hpp>
#include <glm/common.hpp>

#include <cmath>

SEA_NAMESPACE_BEGIN

__host__ Image * image_create(size_t width, size_t height) {
    Image * image;
    cudaMallocManaged(&image, sizeof(Image));

    image->width = width;
    image->height = height;

    cudaMallocManaged(&image->data, width * height * 3 * sizeof(float));

    return image;
}
__host__ void image_delete(Image * image) {
    if (image) {
        cudaFree(image->data);
        cudaFree(image);

        image = nullptr;
    }
}
__host__ __device__ void image_set(
    Image * image, size_t x, size_t y, const glm::vec3 & color) {
    size_t index = (x + y * image->width) * 3;

    image->data[index] = color.r;
    image->data[index + 1] = color.g;
    image->data[index + 2] = color.b;
}
__host__ __device__ void image_get(
    const Image * image, size_t x, size_t y, glm::vec3 & color) {
    size_t index = (x + y * image->width) * 3;

    color.r = image->data[index];
    color.g = image->data[index + 1];
    color.b = image->data[index + 2];
}

__host__ Renderer * renderer_create(size_t width, size_t height,
    size_t camera_samples, size_t light_samples, size_t maximum_depth, float gamma) {
    Renderer * renderer;
    cudaMallocManaged(&renderer, sizeof(Renderer));

    renderer->camera_samples = camera_samples;
    renderer->light_samples = light_samples;
    renderer->maximum_depth = maximum_depth;
	renderer->pass = 0;
    renderer->time = 0;
    renderer->progressive_time = 0;
    renderer->accumulated = 0.0f;

    renderer->gamma = gamma;

	renderer->frame_radiance_pass = image_create(width, height);
    renderer->radiance_pass = image_create(width, height);
	renderer->albedo_pass = image_create(width, height);
	renderer->normal_pass = image_create(width, height);

	renderer->image = renderer->radiance_pass;

    return renderer;
}
__host__ void renderer_update(Renderer * renderer, Camera * camera, size_t width, size_t height) {
	camera->film.width = (float)width;
    camera->film.height = (float)height;

	image_delete(renderer->frame_radiance_pass);
    image_delete(renderer->radiance_pass);
	image_delete(renderer->albedo_pass);
	image_delete(renderer->normal_pass);

	renderer->frame_radiance_pass = image_create(width, height);
	renderer->radiance_pass = image_create(width, height);
	renderer->albedo_pass = image_create(width, height);
	renderer->normal_pass = image_create(width, height);

	Image * passes[3] = {
		renderer->radiance_pass,
		renderer->albedo_pass,
		renderer->normal_pass };

	renderer->image = passes[renderer->pass];
}
__host__ void renderer_delete(Renderer * renderer) {
    if (renderer) {
		image_delete(renderer->frame_radiance_pass);
        image_delete(renderer->radiance_pass);
		image_delete(renderer->albedo_pass);
		image_delete(renderer->normal_pass);

        cudaFree(renderer);

        renderer = nullptr;
    }
}
__global__ void renderer_initialize(const Renderer * renderer, curandState * random_state) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < renderer->image->width && j < renderer->image->height) {
		size_t index = i + j * renderer->image->width;
		curand_init(renderer->time, index, 0, &random_state[index]);
	}
}
__global__ void renderer_render_passes(const Scene * scene, Camera * camera, Renderer * renderer, curandState * random_state) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < renderer->image->width && j < renderer->image->height) {
		size_t index = i + j * renderer->image->width;
		curandState local_random_state = random_state[index];

		glm::vec3 output_albedo(0.0f, 0.0f, 0.0f);
		glm::vec3 output_normal(0.0f, 0.0f, 0.0f);

		for (size_t k = 0; k < renderer->camera_samples; k++) {
			Ray camera_ray;
			camera_generate_ray(camera, (float)i, (float)j, local_random_state, camera_ray);

			glm::vec3 albedo, normal;
			renderer_trace_passes(scene, camera, renderer, camera_ray, local_random_state, albedo, normal);

			output_albedo += albedo;
			output_normal += normal;
		}

		output_albedo /= (float)renderer->camera_samples;
		output_normal /= (float)renderer->camera_samples;

		color_gamma(output_albedo, renderer->gamma);
		color_saturate(output_albedo);

		glm::vec3 previous_albedo, previous_normal;

		image_get(renderer->albedo_pass, i, j, previous_albedo);
		image_get(renderer->normal_pass, i, j, previous_normal);

		output_albedo = color_mix(output_albedo, previous_albedo, renderer->accumulated);
		output_normal = color_mix(output_normal, previous_normal, renderer->accumulated);

		image_set(renderer->albedo_pass, i, j, output_albedo);
		image_set(renderer->normal_pass, i, j, output_normal);
	}
}
__device__ void renderer_trace_passes(
	const Scene * scene, const Camera * camera, const Renderer * renderer, const Ray & ray, curandState & state,
	glm::vec3 & albedo, glm::vec3 & normal) {
	Intersection intersection;

	if (!scene_intersects_ray(scene, ray, intersection)) {
		albedo = glm::vec3(0.0f, 0.0f, 0.0f);
		normal = glm::vec3(0.0f, 0.0f, 0.0f);
	}
	else {
		Triangle * triangle = scene->triangle_list[intersection.index];
		BSDF * bsdf = triangle->bsdf;

		ShaderGlobals shader_globals;
		triangle_shader_globals(triangle, ray, intersection, shader_globals);

		albedo = SEA_INV_PI * bsdf->intensity * bsdf->color;
		normal = glm::inverse(camera->view_matrix) * glm::vec4(shader_globals.normal, 0.0f);
	}
}
__global__ void renderer_render(const Scene * scene, Camera * camera, Renderer * renderer, curandState * random_state) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < renderer->image->width && j < renderer->image->height) {
		size_t index = i + j * renderer->image->width;
		curandState local_random_state = random_state[index];

		Ray camera_ray;
		camera_generate_ray(camera, (float)i, (float)j, local_random_state, camera_ray);

		glm::vec3 radiance = renderer_trace(scene, camera, renderer, camera_ray, local_random_state);

		color_gamma(radiance, renderer->gamma);
		color_saturate(radiance);

		image_set(renderer->frame_radiance_pass, i, j, radiance);
	}
}
__device__ glm::vec3 renderer_trace(
	const Scene * scene, const Camera * camera, const Renderer * renderer, const Ray & ray, curandState & state) {
    glm::vec3 radiance(0.0f, 0.0f, 0.0f);
    glm::vec3 weight(1.0f, 1.0f, 1.0f);

    Ray temp_ray = ray;

    for (size_t depth = 0; depth < renderer->maximum_depth; depth++) {
        Intersection intersection;

        if (!scene_intersects_ray(scene, temp_ray, intersection))
            break;

        Triangle * triangle = scene->triangle_list[intersection.index];
        BSDF * bsdf = triangle->bsdf;

        if (bsdf->type == BSDF_LIGHT) {
            radiance += color_multiply(weight, bsdf->intensity * bsdf->color);
            break;
        }
        else {
			ShaderGlobals shader_globals;
			triangle_shader_globals(triangle, temp_ray, intersection, shader_globals);

            glm::vec3 direct_light(0.0f, 0.0f, 0.0f);

            for (size_t j = 0; j < renderer->light_samples; j++) {
				size_t light_index = random_uniform_uinteger(state) % scene->light_count;
				Triangle * light = scene->light_group[light_index];

                const glm::vec3 & v0 = light->vertices.v0.position;
                const glm::vec3 & v1 = light->vertices.v1.position;
                const glm::vec3 & v2 = light->vertices.v2.position;

                glm::vec3 uvw = random_uniform_triangle(random_uniform_2D(state));
                glm::vec3 point = uvw.x * v0 + uvw.y * v1 + uvw.z * v2;

                Ray shadow_ray;
                shadow_ray.origin = shader_globals.point + SEA_BIAS * shader_globals.normal;
                shadow_ray.direction = point - shader_globals.point;

                float squared_light_distance = glm::dot(shadow_ray.direction, shadow_ray.direction);
                shadow_ray.direction /= glm::sqrt(squared_light_distance);

                Intersection shadow_intersection;
                scene_intersects_ray(scene, shadow_ray, shadow_intersection);

                if (shadow_intersection.hit && scene->triangle_list[shadow_intersection.index] == light) {
					ShaderGlobals light_shader_globals;
                    triangle_shader_globals(light, shadow_ray, shadow_intersection, light_shader_globals);

                    glm::vec3 diffuse_bsdf = SEA_INV_PI * bsdf->intensity * bsdf->color;
                    float cosine = glm::max(0.0f, glm::dot(shader_globals.normal, shadow_ray.direction));

                    glm::vec3 light_emission = light->bsdf->intensity * light->bsdf->color;
                    float light_inverse_pdf = light->surface_area;
                    float light_cosine = glm::max(0.0f, glm::dot(light_shader_globals.normal, -shadow_ray.direction));

                    glm::vec3 estimated_light = light_emission * light_inverse_pdf * light_cosine / squared_light_distance;

                    direct_light += color_multiply(diffuse_bsdf, estimated_light) * cosine;
                }
            }

            direct_light *= (float)scene->light_count / (float)renderer->light_samples;
            radiance += color_multiply(weight, direct_light);

            temp_ray.origin = shader_globals.point + SEA_BIAS * shader_globals.normal;

            glm::vec3 sample = random_uniform_cosine_weighted_hemisphere(random_uniform_2D(state));
            world_coordinate_system(sample, shader_globals, temp_ray.direction);

            weight = color_multiply(weight, bsdf->intensity * bsdf->color);

            float p = glm::max(weight.r, glm::max(weight.g, weight.b));

            if (random_uniform_1D(state) > p)
                break;

            weight /= p;
        }
    }

	return radiance;
}
__global__ void renderer_accumulate(Renderer * renderer) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < renderer->image->width && j < renderer->image->height) {
		glm::vec3 previous_radiance;
		image_get(renderer->radiance_pass, i, j, previous_radiance);

		glm::vec3 radiance;
		image_get(renderer->frame_radiance_pass, i, j, radiance);

		radiance = color_mix(radiance, previous_radiance, renderer->accumulated);
		image_set(renderer->radiance_pass, i, j, radiance);
	}
}

SEA_NAMESPACE_END