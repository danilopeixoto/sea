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

#include <sea/denoiser.h>
#include <sea/renderer.h>

#include <optix_function_table_definition.h>

SEA_NAMESPACE_BEGIN

__host__ Denoiser * denoiser_create(DenoiserType type) {
	Denoiser * denoiser = new Denoiser;

	denoiser->type = type;

	denoiser->cuda_context = 0;
	denoiser->options = {};

	denoiser->denoiser_options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
	denoiser->denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT3;

	denoiser->state_size = 0;
	denoiser->scratch_size = 0;

	denoiser->state = 0;

	switch (type) {
	case HDR:
		denoiser->model = OPTIX_DENOISER_MODEL_KIND_HDR;
		break;
	default:
		denoiser->model = OPTIX_DENOISER_MODEL_KIND_LDR;
	}

	cudaFree(0);
	optixInit();

	optixDeviceContextCreate(denoiser->cuda_context, &denoiser->options, &denoiser->context);
	optixDenoiserCreate(denoiser->context, &denoiser->denoiser_options, &denoiser->denoiser_handler);
	optixDenoiserSetModel(denoiser->denoiser_handler, denoiser->model, nullptr, 0);

	return denoiser;
}
__host__ void denoiser_update(Denoiser * denoiser, const Renderer * renderer) {
	unsigned int width = renderer->image->width;
	unsigned int height = renderer->image->height;

	OptixDenoiserSizes sizes;
	optixDenoiserComputeMemoryResources(denoiser->denoiser_handler, width, height, &sizes);

	CUstream stream;
	cudaStreamCreate(&stream);

	denoiser->state_size = sizes.stateSizeInBytes;
	denoiser->scratch_size = sizes.recommendedScratchSizeInBytes;

	if (denoiser->state)
		cudaFree((void *)denoiser->state);

	cudaMalloc((void **)&denoiser->state, denoiser->state_size + denoiser->scratch_size);

	optixDenoiserSetup(
		denoiser->denoiser_handler, stream, width, height,
		denoiser->state, denoiser->state_size,
		denoiser->state + denoiser->state_size, denoiser->scratch_size);

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}
__host__ void denoiser_delete(Denoiser * denoiser) {
	if (denoiser) {
		optixDenoiserDestroy(denoiser->denoiser_handler);
		optixDeviceContextDestroy(denoiser->context);

		if (denoiser->state)
			cudaFree((void *)denoiser->state);

		delete denoiser;

		denoiser = nullptr;
	}
}

__host__ void denoiser_denoise(Denoiser * denoiser, Renderer * renderer, float blend_factor) {
	unsigned int width = renderer->image->width;
	unsigned int height = renderer->image->height;

	Image * passes[4] = {
		renderer->frame_radiance_pass,
		renderer->albedo_pass,
		renderer->normal_pass,
		renderer->frame_radiance_pass };

	CUstream stream;
	cudaStreamCreate(&stream);

	OptixDenoiserParams parameters = {};
	parameters.blendFactor = 1.0f - blend_factor;

	OptixImage2D layers[4];

	for (size_t i = 0; i < 4; i++) {
		OptixImage2D & layer = layers[i];

		layer.width = width;
		layer.height = height;
		layer.pixelStrideInBytes = 3 * sizeof(float);
		layer.rowStrideInBytes = layer.pixelStrideInBytes * width;
		layer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

		layer.data = (CUdeviceptr)passes[i]->data;
	}

	OptixImage2D * input_layers = layers;
	OptixImage2D * output_layer = layers + 3;

	optixDenoiserInvoke(
		denoiser->denoiser_handler, stream,
		&parameters, denoiser->state, denoiser->state_size, input_layers, 3, 0, 0,
		output_layer, denoiser->state + denoiser->state_size, denoiser->scratch_size);

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}

SEA_NAMESPACE_END