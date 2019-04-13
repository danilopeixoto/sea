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

#include <sea/scene.h>
#include <sea/bvh.h>
#include <sea/math.h>

#include <glm/vec4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

SEA_NAMESPACE_BEGIN

__host__ __device__ bool triangle_intersects_ray(
    const Triangle * triangle, const Ray & ray, Intersection & intersection) {
    intersection_make_empty(intersection);

    const glm::vec3 & v0 = triangle->vertices.v0.position;
    const glm::vec3 & v1 = triangle->vertices.v1.position;
    const glm::vec3 & v2 = triangle->vertices.v2.position;

    glm::vec3 u = v1 - v0;
    glm::vec3 v = v2 - v0;

    glm::vec3 p = glm::cross(ray.direction, v);
    float d = glm::dot(u, p);

    if (glm::abs(d) < SEA_EPSILON)
        return false;

    glm::vec3 t = ray.origin - v0;
    float inverse_det = 1.0f / d;

    float u_coordinate = glm::dot(t, p) * inverse_det;

    if (u_coordinate < 0.0f || u_coordinate > 1.0f)
        return false;

    glm::vec3 q = glm::cross(t, u);
    float v_coordinate = glm::dot(ray.direction, q) * inverse_det;

    if (v_coordinate < 0.0f || u_coordinate + v_coordinate > 1.0f)
        return false;

    float distance = glm::dot(v, q) * inverse_det;

    if (distance < SEA_EPSILON)
        return false;

    intersection.hit = true;
    intersection.distance = distance;
    intersection.u = u_coordinate;
    intersection.v = v_coordinate;

    return true;
}
__host__ void triangle_compute_bounds(Triangle * triangle) {
    static const glm::vec3 margin(SEA_BIAS, SEA_BIAS, SEA_BIAS);

    box_make_empty(triangle->bounds);

    box_extend_point(triangle->bounds, triangle->vertices.v0.position);
    box_extend_point(triangle->bounds, triangle->vertices.v1.position);
    box_extend_point(triangle->bounds, triangle->vertices.v2.position);

    triangle->bounds.minimum -= margin;
    triangle->bounds.maximum += margin;
}
__host__ void triangle_compute_uv(Triangle * triangle) {
    triangle->vertices.v0.uv.x = 0.0f;
    triangle->vertices.v0.uv.y = 0.0f;

    triangle->vertices.v1.uv.x = 1.0f;
    triangle->vertices.v1.uv.y = 0.0f;

    triangle->vertices.v2.uv.x = 0.0f;
    triangle->vertices.v2.uv.y = 1.0f;
}
__host__ __device__ void triangle_shader_globals(const Triangle * triangle,
    const Ray & ray, const Intersection & intersection, ShaderGlobals & shader_globals) {
    ray_point(ray, intersection.distance, shader_globals.point);

    float w = 1.0f - intersection.u - intersection.v;

    shader_globals.geometric_normal = glm::normalize(
        triangle->vertices.v0.normal * w +
        triangle->vertices.v1.normal * intersection.u +
        triangle->vertices.v2.normal * intersection.v);

    float d = glm::dot(ray.direction, shader_globals.geometric_normal);
    shader_globals.normal = d < 0.0f ? shader_globals.geometric_normal : -shader_globals.geometric_normal;

    shader_globals.uv =
        triangle->vertices.v0.uv * w +
        triangle->vertices.v1.uv * intersection.u +
        triangle->vertices.v2.uv * intersection.v;
}

__host__ __device__ float film_aspect_ratio(const Film & film) {
    return film.width / film.height;
}

__host__ Camera * camera_create(
    float field_of_view, float focal_length, float aperture, float width, float height) {
    static const glm::mat4 identity(1.0f);

    Camera * camera;
    cudaMallocManaged(&camera, sizeof(Camera));

    camera->field_of_view = field_of_view;
    camera->focal_length = focal_length;
    camera->aperture = aperture;

    camera->film.width = width;
    camera->film.height = height;

    camera->view_matrix = identity;

    return camera;
}
__host__ void camera_delete(Camera * camera) {
    if (camera) {
        cudaFree(camera);
        camera = nullptr;
    }
}
__host__ void camera_update_view_matrix(
    Camera * camera, const glm::vec3 & translation, const glm::vec3 & rotation) {
    static const glm::mat4 identity(1.0f);
    static const glm::vec3 y_axis(0.0f, 1.0f, 0.0f);
    static const glm::vec3 eye(0.0f, 0.0f, 25.0f);

    glm::mat4 translation_matrix = glm::translate(identity, translation);
    glm::mat4 default_view_matrix = glm::translate(identity, eye);
    glm::mat4 y_rotation_matrix = glm::rotate(identity, rotation.x, y_axis);

    camera->view_matrix = y_rotation_matrix * default_view_matrix * translation_matrix;

    glm::vec3 axis = glm::cross(y_axis, glm::vec3(
        camera->view_matrix[2][0], camera->view_matrix[2][1], camera->view_matrix[2][2]));
    glm::mat4 rotation_around_axis_matrix = glm::rotate(identity, rotation.y, axis);

    camera->view_matrix = rotation_around_axis_matrix * camera->view_matrix;
}
__host__ __device__ void camera_generate_ray(
    const Camera * camera, float x, float y, size_t & seed, Ray & ray) {
    glm::vec4 origin(0.0f, 0.0f, 0.0f, 1.0f);

    if (camera->aperture > SEA_BIAS) {
        float radius = camera->focal_length / camera->aperture * 0.5f;
        glm::vec2 sample = random_uniform_2D(seed);

        origin.x = radius * sample.x;
        origin.y = radius * sample.y;
    }

    glm::vec2 sample = random_uniform_2D(seed);
    float scale = camera->focal_length * glm::tan(camera->field_of_view * 0.5f);

    glm::vec4 pixel;

    pixel.x = (2.0f * (x + sample.x) / camera->film.width - 1.0f) * scale * film_aspect_ratio(camera->film);
    pixel.y = (1.0f - 2.0f * (y + sample.y) / camera->film.height) * scale;
    pixel.z = -camera->focal_length;
    pixel.w = 1.0f;

    origin = camera->view_matrix * origin;
    pixel = camera->view_matrix * pixel;

    ray.origin = origin;
    ray.direction = glm::normalize(pixel - origin);
}

__host__ Scene * scene_load(const char * filename, BSDF * bsdf) {
    std::ifstream file(filename);

    if (!file.is_open())
        return nullptr;

    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> textureCoordinates;
    std::vector<size_t> vertex_indices, normal_indices, texture_indices;

    std::string line;

    while (std::getline(file, line)) {
        std::istringstream attributes(line);

        std::string type;
        attributes >> type;

        if (type == "v") {
            glm::vec3 vertex;
            attributes >> vertex.x >> vertex.y >> vertex.z;

            vertices.push_back(vertex);
        }
        else if (type == "vt") {
            glm::vec2 uvCoordinates;
            attributes >> uvCoordinates.x >> uvCoordinates.y;

            textureCoordinates.push_back(uvCoordinates);
        }
        else if (type == "vn") {
            glm::vec3 normal;
            attributes >> normal.x >> normal.y >> normal.z;

            normals.push_back(glm::normalize(normal));
        }
        else if (type == "f") {
            for (size_t i = 0; i < 3; i++) {
                std::string tokens;
                attributes >> tokens;

                std::replace(tokens.begin(), tokens.end(), '/', ' ');

                std::istringstream indices(tokens);
                size_t index;

                indices >> index;
                vertex_indices.push_back(index - 1);

                if (indices.peek() == ' ') {
                    indices.ignore();

                    if (indices.peek() == ' ') {
                        indices.ignore();

                        indices >> index;
                        normal_indices.push_back(index - 1);
                    }
                    else {
                        indices >> index;
                        texture_indices.push_back(index - 1);

                        if (indices.peek() == ' ') {
                            indices.ignore();

                            indices >> index;
                            normal_indices.push_back(index - 1);
                        }
                    }
                }
            }
        }
    }

    file.close();

    size_t triangle_count = vertex_indices.size() / 3;

    if (triangle_count == 0)
        return nullptr;

    Scene * scene;
    cudaMallocManaged(&scene, sizeof(Scene));

    scene->bvh_tree = nullptr;
    scene->triangle_count = triangle_count;

    cudaMallocManaged(&scene->triangle_list, triangle_count * sizeof(Triangle *));

    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int i = 0; i < triangle_count; i++) {
        Triangle * triangle;
        cudaMallocManaged(&triangle, sizeof(Triangle));

        size_t vertex_index0 = vertex_indices[i * 3];
        size_t vertex_index1 = vertex_indices[i * 3 + 1];
        size_t vertex_index2 = vertex_indices[i * 3 + 2];

        triangle->vertices.v0.position = vertices[vertex_index0];
        triangle->vertices.v1.position = vertices[vertex_index1];
        triangle->vertices.v2.position = vertices[vertex_index2];

        glm::vec3 e0 = triangle->vertices.v1.position - triangle->vertices.v0.position;
        glm::vec3 e1 = triangle->vertices.v2.position - triangle->vertices.v0.position;

        glm::vec3 n = glm::cross(e0, e1);
        triangle->surface_area = glm::length(n);

        try {
            size_t normal_index0 = normal_indices.at(i * 3);
            size_t normal_index1 = normal_indices.at(i * 3 + 1);
            size_t normal_index2 = normal_indices.at(i * 3 + 2);

            triangle->vertices.v0.normal = normals.at(normal_index0);
            triangle->vertices.v1.normal = normals.at(normal_index1);
            triangle->vertices.v2.normal = normals.at(normal_index2);
        }
        catch (const std::out_of_range & exception) {
            n /= triangle->surface_area;

            triangle->vertices.v0.normal = n;
            triangle->vertices.v1.normal = n;
            triangle->vertices.v2.normal = n;
        }

        triangle->surface_area /= 2.0f;

        try {
            size_t texture_index0 = texture_indices.at(i * 3);
            size_t texture_index1 = texture_indices.at(i * 3 + 1);
            size_t texture_index2 = texture_indices.at(i * 3 + 2);

            triangle->vertices.v0.uv = textureCoordinates.at(texture_index0);
            triangle->vertices.v1.uv = textureCoordinates.at(texture_index1);
            triangle->vertices.v2.uv = textureCoordinates.at(texture_index2);
        }
        catch (const std::out_of_range & exception) {
            triangle_compute_uv(triangle);
        }

        triangle_compute_bounds(triangle);
        triangle->bsdf = bsdf;

        scene->triangle_list[i] = triangle;
    }

    return scene;
}
__host__ void scene_group_lights(Scene * scene) {
    size_t triangle_count = scene->triangle_count;

    std::vector<Triangle *> lights;
    lights.reserve(triangle_count / 2);

    for (size_t i = 0; i < triangle_count; i++) {
        Triangle * triangle = scene->triangle_list[i];

        if (triangle->bsdf->type == BSDF_LIGHT)
            lights.push_back(triangle);
    }

    scene->light_count = lights.size();

    cudaMallocManaged(&scene->light_group, scene->light_count * sizeof(Triangle *));

    for (size_t i = 0; i < scene->light_count; i++)
        scene->light_group[i] = lights[i];
}
__host__ void scene_delete(Scene * scene) {
    if (scene) {
        std::unordered_set<BSDF *> bsdf_set;

        for (size_t i = 0; i < scene->triangle_count; i++) {
            Triangle * triangle = scene->triangle_list[i];
            BSDF * bsdf = triangle->bsdf;

            if (bsdf)
                bsdf_set.insert(bsdf);

            cudaFree(triangle);
        }

        cudaFree(scene->triangle_list);
        cudaFree(scene->light_group);
        bvh_tree_delete(scene->bvh_tree);

        std::unordered_set<BSDF *>::iterator it = bsdf_set.begin();

        while (it != bsdf_set.end()) {
            BSDF * bsdf = *(it++);
            cudaFree(bsdf);
        }

        cudaFree(scene);
        scene = nullptr;
    }
}
__host__ void scene_compute_bounds(const Scene * scene, size_t begin, size_t end, Box & bounds) {
    box_make_empty(bounds);

    for (size_t i = begin; i < end; i++) {
        const Triangle * triangle = scene->triangle_list[i];
        box_extend_box(bounds, triangle->bounds);
    }
}
__host__ __device__ bool scene_intersects_ray(
    const Scene * scene, const Ray & ray, Intersection & intersection) {
    intersection_make_empty(intersection);

    BVHTree * bvh_tree = scene->bvh_tree;

    if (bvh_tree->node_count == 0)
        return false;

    BVHNode * root = bvh_tree->node_list[0];
    const Box & bounds = root->bounds;

    Intersection temp;

    if (!box_intersects_ray(bounds, ray, temp))
        return false;

    int top = -1;
    BVHIntersection stack[64];

    BVHIntersection * bvh_intersection = &stack[++top];

    bvh_intersection->distance = temp.distance;
    bvh_intersection->node = root;

    while (top != -1) {
        bvh_intersection = &stack[top--];

        if (bvh_intersection->distance < intersection.distance) {
            BVHNode * node = bvh_intersection->node;

            if (node->leaf) {
                for (size_t i = node->left; i < node->right; i++) {
                    Triangle * triangle = scene->triangle_list[i];

                    triangle_intersects_ray(triangle, ray, temp);

                    if (temp.hit && temp.distance < intersection.distance) {
                        intersection = temp;
                        intersection.index = i;
                    }
                }
            }
            else {
                BVHNode * lhs_node = bvh_tree->node_list[node->left];
                BVHNode * rhs_node = bvh_tree->node_list[node->right];

                const Box & lhs_bounds = lhs_node->bounds;
                const Box & rhs_bounds = rhs_node->bounds;

                Intersection lhs_intersection, rhs_intersection;

                box_intersects_ray(lhs_bounds, ray, lhs_intersection);
                box_intersects_ray(rhs_bounds, ray, rhs_intersection);

                if (lhs_intersection.distance < rhs_intersection.distance) {
                    if (rhs_intersection.distance < SEA_INFINITY) {
                        bvh_intersection = &stack[++top];

                        bvh_intersection->distance = rhs_intersection.distance;
                        bvh_intersection->node = rhs_node;
                    }

                    if (lhs_intersection.distance < SEA_INFINITY) {
                        bvh_intersection = &stack[++top];

                        bvh_intersection->distance = lhs_intersection.distance;
                        bvh_intersection->node = lhs_node;
                    }
                }
                else {
                    if (lhs_intersection.distance < SEA_INFINITY) {
                        bvh_intersection = &stack[++top];

                        bvh_intersection->distance = lhs_intersection.distance;
                        bvh_intersection->node = lhs_node;
                    }

                    if (rhs_intersection.distance < SEA_INFINITY) {
                        bvh_intersection = &stack[++top];

                        bvh_intersection->distance = rhs_intersection.distance;
                        bvh_intersection->node = rhs_node;
                    }
                }
            }
        }
    }

    return intersection.hit;
}

SEA_NAMESPACE_END