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

#ifndef SEA_UTILITY_H
#define SEA_UTILITY_H

#include <sea/global.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

SEA_NAMESPACE_BEGIN

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Intersection {
    bool hit;
    float distance;
    float u;
    float v;
    size_t index;
};

struct ShaderGlobals {
    glm::vec3 point;
    glm::vec3 geometric_normal;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Box {
    glm::vec3 minimum;
    glm::vec3 maximum;
};

__device__ size_t random_uniform_uinteger(curandState &);
__device__ float random_uniform_1D(curandState &);
__device__ glm::vec2 random_uniform_2D(curandState &);
__host__ __device__ glm::vec2 random_uniform_disk(const glm::vec2 &);
__host__ __device__ glm::vec3 random_uniform_triangle(const glm::vec2 &);
__host__ __device__ glm::vec3 random_uniform_cosine_weighted_hemisphere(const glm::vec2 &);

__host__ __device__ void world_coordinate_system(const glm::vec3 &, const ShaderGlobals &, glm::vec3 &);

__host__ __device__ glm::vec3 reflect(const glm::vec3 &, const glm::vec3 &);
__host__ __device__ glm::vec3 refract(const glm::vec3 &, const glm::vec3 &, float);
__host__ __device__ float fresnel(const glm::vec3 &, const glm::vec3 &, float);

__host__ __device__ void color_gamma(glm::vec3 &, float);
__host__ __device__ void color_saturate(glm::vec3 &);
__host__ __device__ glm::vec3 color_multiply(const glm::vec3 &, const glm::vec3 &);
__host__ __device__ glm::vec3 color_mix(const glm::vec3 &, const glm::vec3 &, float);

__host__ __device__ void ray_point(const Ray &, float, glm::vec3 &);

__host__ __device__ void intersection_make_empty(Intersection &);

__host__ __device__ void box_make_empty(Box &);
__host__ __device__ glm::vec3 box_size(const Box &);
__host__ __device__ void box_extend_point(Box &, const glm::vec3 &);
__host__ __device__ void box_extend_box(Box &, const Box &);
__host__ __device__ bool box_intersects_ray(const Box &, const Ray &, Intersection &);

SEA_NAMESPACE_END

#endif