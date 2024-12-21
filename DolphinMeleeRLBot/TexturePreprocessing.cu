#include "globals.h"
#include <iostream>
#include "device_launch_parameters.h"
#include <texture_indirect_functions.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
void CopyAndSaveTextureAsPNG(uint8_t* gpuTextureData, int buffer_size, const char* filename) {
    std::cout << "entering png saving portion" << std::endl;
	// Allocate CPU memory for the image data
	uint8_t* cpuTextureData = new uint8_t[buffer_size];
    if (!gpuTextureData) {
        std::cout << "GPU texture data null" << std::endl;
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "sync failed" << std::endl;
        return;
    }

    std::cout << "done sync" << std::endl;
	// Copy data from the GPU to the CPU
	 err = cudaMemcpy(cpuTextureData, gpuTextureData, buffer_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
		delete[] cpuTextureData;
		return;
	}
    std::cout << "Starting 10:10:10:2 to 8:8:8:8 conversion..." << std::endl;

    // Convert 10:10:10:2 format to 8:8:8:8 format
   // for (size_t i = 0; i < buffer_size / 4; i++) {
   //     //std::cout << "Pixel " << i << ": ("
   //     //    << static_cast<int>(cpuTextureData[i * 4 + 0]) << ", "
   //     //    << static_cast<int>(cpuTextureData[i * 4 + 1]) << ", "
   //     //    << static_cast<int>(cpuTextureData[i * 4 + 2]) << ", "
   //     //    << static_cast<int>(cpuTextureData[i * 4 + 3]) << ")" << std::endl;
   //     // Extract the 10:10:10:2 format pixel
   //     uint32_t pixel = *reinterpret_cast<uint32_t*>(cpuTextureData + i * 4);  // 4 bytes per pixel

   //     // Extract RGB (10-bit each) and alpha (2-bit)
   //     uint32_t r = (pixel >> 0) & 0x3FF;  // Extract the 10-bit red
   //     uint32_t g = (pixel >> 10) & 0x3FF; // Extract the 10-bit green
   //     uint32_t b = (pixel >> 20) & 0x3FF; // Extract the 10-bit blue
   //     uint32_t a = (pixel >> 30) & 0x03;  // Extract the 2-bit alpha

   //     // Convert to 8-bit by scaling the 10-bit RGB channels down to 8-bit
   //     uint8_t r8 = static_cast<uint8_t>((r * 255) / 1023);  // Scale 10-bit to 8-bit
   //     uint8_t g8 = static_cast<uint8_t>((g * 255) / 1023);  // Scale 10-bit to 8-bit
   //     uint8_t b8 = static_cast<uint8_t>((b * 255) / 1023);  // Scale 10-bit to 8-bit

   //     // Convert 2-bit alpha to 8-bit by scaling
   //     uint8_t a8 = static_cast<uint8_t>((a * 255) / 3);  // Scale 2-bit to 8-bit
  
   //     // Store the converted RGBA values back into the cpuTextureData array
   //     cpuTextureData[i * 4 + 0] = r8;  // Red channel
   //     cpuTextureData[i * 4 + 1] = g8;  // Green channel
   //     cpuTextureData[i * 4 + 2] = b8;  // Blue channel
   //     cpuTextureData[i * 4 + 3] = a8;  // Alpha channel
   ///*     std::cout << "Pixel " << i << ": ("
   //         << static_cast<int>(cpuTextureData[i * 4 + 0]) << ", "
   //         << static_cast<int>(cpuTextureData[i * 4 + 1]) << ", "
   //         << static_cast<int>(cpuTextureData[i * 4 + 2]) << ", "
   //         << static_cast<int>(cpuTextureData[i * 4 + 3]) << ")" << std::endl;*/
   // }

    std::cout << "Finished conversion." << std::endl;
	// Save the image as a PNG using stb_image_write
	// Note: The image will be saved as grayscale (1 channel)
    std::cerr << "saving png before grayscale " << std::endl;
	if (!stbi_write_png(filename, 958, 720, 4, cpuTextureData, 958 * 4)) {
		std::cerr << "Failed to write PNG file: " << filename << std::endl;
	}

	// Clean up
	delete[] cpuTextureData;
}

// Conversion function for 10:10:10:2 format to float4
__device__ float4 convert_1010102_to_float4(uint32_t pixel) {
    // Extract each channel and normalize it
    float r = ((pixel >> 0) & 0x3FF) / 1023.0f;  // 10 bits for Red
    float g = ((pixel >> 10) & 0x3FF) / 1023.0f; // 10 bits for Green
    float b = ((pixel >> 20) & 0x3FF) / 1023.0f; // 10 bits for Blue
    float a = ((pixel >> 30) & 0x3) / 3.0f;       // 2 bits for Alpha
    return make_float4(r, g, b, a);
}

// Kernel to process 10:10:10:2 texture data
__global__ void copy_downscale_grayscale_kernel(uint8_t* texture_data, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 4;  // 4 bytes per pixel (RGBA)

        // Read the 32-bit pixel value from the texture data
        uint32_t pixel = *reinterpret_cast<uint32_t*>(texture_data + idx);

        // Convert the 10:10:10:2 format to float4
        float4 color = convert_1010102_to_float4(pixel);

        // Convert to grayscale using the weighted average of RGB
        float grayscale = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;

        // Store the grayscale value in the texture (set all channels to grayscale value)
        texture_data[idx + 0] = static_cast<uint8_t>(grayscale * 255.0f);  // R
        texture_data[idx + 1] = static_cast<uint8_t>(grayscale * 255.0f);  // G
        texture_data[idx + 2] = static_cast<uint8_t>(grayscale * 255.0f);  // B
        texture_data[idx + 3] = static_cast<uint8_t>(color.w * 255.0f);     // A (Alpha remains unchanged)
    }
}

void CopyAndPreprocess(ComPtr<ID3D11Texture2D> texture, UINT width, UINT height) {
    std::cout << "starting preprocessing..." << std::endl;
    // Initialize CUDA for Direct3D 11 interoperability
   /* cudaError_t cudaStatus = cudaD3D11SetDirect3DDevice(d3dDevice.Get());
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to set Direct3D 11 device for CUDA. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }*/
    std::cout << "registering..." << std::endl;
    // Register the Direct3D 11 buffer with CUDA
    cudaGraphicsResource_t cudaResource = nullptr;
    cudaError_t cudaStatus = cudaGraphicsD3D11RegisterResource(&cudaResource, texture.Get(), cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to register Direct3D 11 tex with CUDA. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    std::cout << "mapping to cuda resource..." << std::endl;
    // Map the resource to access it in CUDA
    cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to map Direct3D 11 buffer for CUDA access. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnregisterResource(cudaResource);
        return;
    }

    // Get the pointer to the buffer data
    void* devicePtr = nullptr;
    size_t mappedSize = 0;
    std::cout << "getting mapped pointer..." << std::endl;
    cudaStatus = cudaGraphicsResourceGetMappedPointer(&devicePtr, &mappedSize, cudaResource);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to get mapped pointer for CUDA buffer. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        cudaGraphicsUnregisterResource(cudaResource);
        return;
    }

    // Ensure the mapped size matches the buffer size
    if (mappedSize < width * height * 4)
    {
        std::cerr << "Warning: Mapped size is smaller than expected buffer size!" << std::endl;
    }

    // Process the data in CUDA (example kernel call)
    //ProcessBufferInCUDA(static_cast<char*>(devicePtr), buffer_size);
    CopyAndSaveTextureAsPNG((uint8_t*)devicePtr, width * height * 4, "before_cuda.png");
    std::cout << "unmapping cuda resource..." << std::endl;
    // Unmap the resource after processing
    cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, 0);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Failed to unmap Direct3D 11 buffer for CUDA. Error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    std::cout << "unregistering cuda resource..." << std::endl;
    // Unregister the resource to clean up
    cudaGraphicsUnregisterResource(cudaResource);
    std::cout << "success" << std::endl;
}

