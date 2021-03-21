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

#ifndef SEA_RENDERER_H
#define SEA_RENDERER_H

#include <sea/global.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/vec3.hpp>

SEA_NAMESPACE_BEGIN

struct Ray;
struct Scene;
struct Camera;
struct BSDF;
struct ShaderGlobals;

struct Image {
    size_t width;
    size_t height;

    float * data;
};

struct Renderer {
	size_t camera_samples;
    size_t light_samples;
    size_t maximum_depth;
	size_t pass;
    size_t time;
    size_t progressive_time;
    float accumulated;

    float gamma;

    Image * image;

	Image * frame_radiance_pass;
	Image * radiance_pass;
	Image * albedo_pass;
	Image * normal_pass;
};

__host__ Image * image_create(size_t, size_t);
__host__ void image_delete(Image *);
__host__ __device__ void image_set(Image *, size_t, size_t, const glm::vec3 &);
__host__ __device__ void image_get(const Image *, size_t, size_t, glm::vec3 &);

template<typename Kernel>
__host__ void renderer_compute_kernel_parameters(const Renderer * renderer, Kernel kernel, dim3 & block_size, dim3 & grid_size) {
	int minimum_grid_size, maximum_block_size;

	cudaOccupancyMaxPotentialBlockSize(
		&minimum_grid_size, &maximum_block_size, kernel, 0, renderer->image->width * renderer->image->height);

	block_size.x = (unsigned int)std::floorf(std::sqrtf((float)maximum_block_size));
	block_size.y = block_size.x;
	block_size.z = 1;

	grid_size.x = (unsigned int)std::ceilf((float)renderer->image->width / (float)block_size.x);
	grid_size.y = (unsigned int)std::ceilf((float)renderer->image->height / (float)block_size.y);
	grid_size.z = 1;
}

__host__ Renderer * renderer_create(size_t, size_t, size_t, size_t, size_t, float);
__host__ void renderer_update(Renderer *, Camera *, size_t, size_t);
__host__ void renderer_delete(Renderer *);
__global__ void renderer_initialize(const Renderer *, curandState *);
__global__ void renderer_render_passes(const Scene *, Camera *, Renderer *, curandState *);
__device__ void renderer_trace_passes(
	const Scene *, const Camera *, const Renderer *, const Ray &, curandState &, glm::vec3 &, glm::vec3 &);
__global__ void renderer_render(const Scene *, Camera *, Renderer *, curandState *);
__device__ glm::vec3 renderer_trace(const Scene *, const Camera *, const Renderer *, const Ray &, curandState &);
__global__ void renderer_accumulate(Renderer *);

SEA_NAMESPACE_END

#endif