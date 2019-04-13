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

#include <sea/utility.h>
#include <sea/math.h>

#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>

#include <cmath>
#include <cstdint>

SEA_NAMESPACE_BEGIN

__host__ __device__ size_t random_uniform_uinteger(size_t & seed) {
    uint32_t s = (uint32_t)seed;

    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s = s ^ (s >> 4u);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15);

    seed = (size_t)s;

    return seed;
}
__host__ __device__ float random_uniform_1D(size_t & seed) {
    return (float)random_uniform_uinteger(seed) / ((float)UINT32_MAX + 1.0f);
}
__host__ __device__ glm::vec2 random_uniform_2D(size_t & seed) {
    return glm::vec2(random_uniform_1D(seed), random_uniform_1D(seed));
}
__host__ __device__ glm::vec2 random_uniform_disk(const glm::vec2 & sample) {
    float radius = glm::sqrt(sample.x);
    float theta = 2.0f * SEA_PI * sample.y;

    return glm::vec2(radius * glm::cos(theta), radius * glm::sin(theta));
}
__host__ __device__ glm::vec3 random_uniform_triangle(const glm::vec2 & sample) {
    float s = glm::sqrt(sample.x);
    float u = 1.0f - s;
    float v = s * sample.y;

    return glm::vec3(u, v, 1.0 - u - v);
}
__host__ __device__ glm::vec3 random_uniform_cosine_weighted_hemisphere(const glm::vec2 & sample) {
    float s = glm::sqrt(sample.x);
    float phi = 2.0f * SEA_PI * sample.y;

    return glm::vec3(s * glm::cos(phi), s * glm::sin(phi), glm::sqrt(1.0f - sample.x));
}

__host__ __device__ void world_coordinate_system(
    const glm::vec3 & sample, const ShaderGlobals & shader_globals, glm::vec3 & direction) {
    const glm::vec3 & w = shader_globals.normal;
    glm::vec3 axis = glm::abs(w.x) > SEA_BIAS ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(axis, w));
    glm::vec3 v = glm::cross(w, u);

    direction.x = sample.x * u.x + sample.y * v.x + sample.z * w.x;
    direction.y = sample.x * u.y + sample.y * v.y + sample.z * w.y;
    direction.z = sample.x * u.z + sample.y * v.z + sample.z * w.z;
}

__host__ __device__ glm::vec3 reflect(const glm::vec3 & incident, const glm::vec3 & normal) {
    return glm::normalize(incident - 2.0f * glm::dot(incident, normal) * normal);
}
__host__ __device__ glm::vec3 refract(const glm::vec3 & incident, const glm::vec3 & normal, float eta) {
    float d = glm::abs(glm::dot(incident, normal));
    float k = 1.0f - eta * eta * (1.0f - d * d);

    if (k < 0.0f)
        return glm::vec3(0.0f);

    return glm::normalize(incident * eta - normal * (eta * d + glm::sqrt(k)));
}
__host__ __device__ float fresnel(const glm::vec3 & incident, const glm::vec3 & normal, float eta) {
    float d = glm::abs(glm::dot(incident, normal));
    float g = 1.0f / (eta * eta) - 1.0f + d * d;

    if (g < 0.0f)
        return 1.0f;

    g = glm::sqrt(g);

    float b = g - d;
    float f0 = b / (g + d);
    float f1 = (d * (g + d) - 1.0f) / (d * b + 1.0f);

    return 0.5f * f0 * f0 * (1.0f + f1 * f1);
}

__host__ __device__ void color_gamma(glm::vec3 & color, float gamma) {
    float t = 1.0f / gamma;

    color.r = glm::pow(color.r, t);
    color.g = glm::pow(color.g, t);
    color.b = glm::pow(color.b, t);
}
__host__ __device__ void color_saturate(glm::vec3 & color) {
    color.r = glm::clamp(color.r, 0.0f, 1.0f);
    color.g = glm::clamp(color.g, 0.0f, 1.0f);
    color.b = glm::clamp(color.b, 0.0f, 1.0f);
}
__host__ __device__ glm::vec3 color_multiply(const glm::vec3 & lhs, const glm::vec3 & rhs) {
    return glm::vec3(lhs.r * rhs.r, lhs.g * rhs.g, lhs.b * rhs.b);
}
__host__ __device__ glm::vec3 color_mix(const glm::vec3 & a, const glm::vec3 & b, float t) {
    return a + t * (b - a);
}

__host__ __device__ void ray_point(const Ray & ray, float distance, glm::vec3 & point) {
    point = ray.origin + ray.direction * distance;
}

__host__ __device__ void intersection_make_empty(Intersection & intersection) {
    intersection.hit = false;
    intersection.distance = SEA_INFINITY;
    intersection.index = 0;
}

__host__ __device__ void box_make_empty(Box & box) {
    box.minimum.x = SEA_INFINITY;
    box.minimum.y = SEA_INFINITY;
    box.minimum.z = SEA_INFINITY;

    box.maximum.x = -SEA_INFINITY;
    box.maximum.y = -SEA_INFINITY;
    box.maximum.z = -SEA_INFINITY;
}
__host__ __device__ glm::vec3 box_size(const Box & box) {
    return box.maximum - box.minimum;
}
__host__ __device__ void box_extend_point(Box & box, const glm::vec3 & point) {
    box.minimum.x = glm::min(box.minimum.x, point.x);
    box.minimum.y = glm::min(box.minimum.y, point.y);
    box.minimum.z = glm::min(box.minimum.z, point.z);

    box.maximum.x = glm::max(box.maximum.x, point.x);
    box.maximum.y = glm::max(box.maximum.y, point.y);
    box.maximum.z = glm::max(box.maximum.z, point.z);
}
__host__ __device__ void box_extend_box(Box & box, const Box & other_box) {
    box.minimum.x = glm::min(box.minimum.x, other_box.minimum.x);
    box.minimum.y = glm::min(box.minimum.y, other_box.minimum.y);
    box.minimum.z = glm::min(box.minimum.z, other_box.minimum.z);

    box.maximum.x = glm::max(box.maximum.x, other_box.maximum.x);
    box.maximum.y = glm::max(box.maximum.y, other_box.maximum.y);
    box.maximum.z = glm::max(box.maximum.z, other_box.maximum.z);
}
__host__ __device__ bool box_intersects_ray(
    const Box & box, const Ray & ray, Intersection & intersection) {
    intersection_make_empty(intersection);

    float tmin = (box.minimum.x - ray.origin.x) / ray.direction.x;
    float tmax = (box.maximum.x - ray.origin.x) / ray.direction.x;

    if (tmin > tmax) {
        float t = tmin;

        tmin = tmax;
        tmax = t;
    }

    float tymin = (box.minimum.y - ray.origin.y) / ray.direction.y;
    float tymax = (box.maximum.y - ray.origin.y) / ray.direction.y;

    if (tymin > tymax) {
        float t = tymin;

        tymin = tymax;
        tymax = t;
    }

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (box.minimum.z - ray.origin.z) / ray.direction.z;
    float tzmax = (box.maximum.z - ray.origin.z) / ray.direction.z;

    if (tzmin > tzmax) {
        float t = tzmin;

        tzmin = tzmax;
        tzmax = t;
    }

    if (tmin > tzmax || tzmin > tmax)
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    intersection.hit = true;
    intersection.distance = tmin;

    return true;
}

SEA_NAMESPACE_END