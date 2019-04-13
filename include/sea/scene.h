// Copyright (c) 2019, Danilo Peixoto and Débora Bacelar. All rights reserved.
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

#ifndef SEA_SCENE_H
#define SEA_SCENE_H

#include <sea/global.h>
#include <sea/utility.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

SEA_NAMESPACE_BEGIN

struct BVHTree;

enum BSDFType {
    BSDF_NONE = 0x0,
    BSDF_LIGHT = 0x1,
    BSDF_DIFFUSE = 0x2,
    BSDF_GLASS = 0x4
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct BSDF {
    BSDFType type;
    float intensity;
    glm::vec3 color;
    float index_of_refraction;
};

struct Triangle {
    union {
        struct {
            Vertex v0;
            Vertex v1;
            Vertex v2;
        };

        Vertex v[3];
    } vertices;

    BSDF * bsdf;
    Box bounds;
    float surface_area;
};

struct Film {
    float width;
    float height;
};

struct Camera {
    float field_of_view;
    float focal_length;
    float aperture;
    Film film;
    glm::mat4x4 view_matrix;
};

struct Scene {
    size_t triangle_count;
    size_t light_count;
    Triangle ** triangle_list;
    Triangle ** light_group;
    BVHTree * bvh_tree;
};

__host__ __device__ bool triangle_intersects_ray(const Triangle *, const Ray &, Intersection &);
__host__ void triangle_compute_bounds(Triangle *);
__host__ void triangle_compute_uv(Triangle *);
__host__ __device__ void triangle_shader_globals(
    const Triangle *, const Ray &, const Intersection &, ShaderGlobals &);

__host__ __device__ float film_aspect_ratio(const Film &);

__host__ Camera * camera_create(float, float, float, float, float);
__host__ void camera_delete(Camera *);
__host__ void camera_update_view_matrix(Camera *, const glm::vec3 &, const glm::vec3 &);
__host__ __device__ void camera_generate_ray(
    const Camera *, float, float, size_t &, Ray &);

__host__ Scene * scene_load(const char *, BSDF *);
__host__ void scene_group_lights(Scene *);
__host__ void scene_delete(Scene *);
__host__ void scene_compute_bounds(const Scene *, size_t, size_t, Box &);
__host__ __device__ bool scene_intersects_ray(const Scene *, const Ray &, Intersection &);

SEA_NAMESPACE_END

#endif