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

#include <sea/bvh.h>
#include <sea/scene.h>

#include <vector>
#include <queue>
#include <algorithm>

SEA_NAMESPACE_BEGIN

__host__ void bvh_tree_build(Scene * scene, size_t leaf_size) {
    std::vector<BVHNode *> nodes;
    std::queue<BVHNode *> queue;

    size_t count = scene->triangle_count;

    if (count > 0) {
        nodes.reserve(2 * count / leaf_size - 1);

        BVHNode * root;
        cudaMallocManaged(&root, sizeof(BVHNode));

        scene_compute_bounds(scene, 0, count, root->bounds);

        root->left = 0;
        root->right = count;
        root->leaf = true;

        queue.push(root);

        while (!queue.empty()) {
            BVHNode * node = queue.front();
            queue.pop();

            nodes.push_back(node);

            size_t begin = node->left;
            size_t end = node->right;

            if (end - begin <= leaf_size)
                continue;

            const Box & bounds = node->bounds;
            glm::vec3 size = box_size(bounds);

            glm::vec3::length_type axis = 0;

            if (size[1] > size[axis])
                axis = 1;
            else if (size[2] > size[axis])
                axis = 2;

            std::sort(scene->triangle_list + begin, scene->triangle_list + end,
                [=](const Triangle * lhs, const Triangle * rhs) {
                float a = lhs->bounds.minimum[axis] + lhs->bounds.maximum[axis];
                float b = rhs->bounds.minimum[axis] + rhs->bounds.maximum[axis];

                if (a < b)
                    return true;

                return false;
            });

            size_t index = (begin + end) / 2;

            if (index > begin && index < end) {
                BVHNode * lhs_node, *rhs_node;

                cudaMallocManaged(&lhs_node, sizeof(BVHNode));
                cudaMallocManaged(&rhs_node, sizeof(BVHNode));

                lhs_node->left = begin;
                lhs_node->right = index;
                lhs_node->leaf = true;

                scene_compute_bounds(scene, begin, index, lhs_node->bounds);

                rhs_node->left = index;
                rhs_node->right = end;
                rhs_node->leaf = true;

                scene_compute_bounds(scene, index, end, rhs_node->bounds);

                node->left = nodes.size() + queue.size();
                node->right = node->left + 1;
                node->leaf = false;

                queue.push(lhs_node);
                queue.push(rhs_node);
            }
        }
    }

    size_t node_count = nodes.size();

    BVHTree * bvh_tree;
    cudaMallocManaged(&bvh_tree, sizeof(BVHTree));

    bvh_tree->node_count = node_count;
    bvh_tree->leaf_size = leaf_size;

    cudaMallocManaged(&bvh_tree->node_list, node_count * sizeof(BVHTree *));

    for (size_t i = 0; i < node_count; i++)
        bvh_tree->node_list[i] = nodes[i];

    scene->bvh_tree = bvh_tree;
}
__host__ void bvh_tree_delete(BVHTree * bvh_tree) {
    if (bvh_tree) {
        for (size_t i = 0; i < bvh_tree->node_count; i++)
            cudaFree(bvh_tree->node_list[i]);

        cudaFree(bvh_tree->node_list);
        cudaFree(bvh_tree);

        bvh_tree = nullptr;
    }
}

SEA_NAMESPACE_END