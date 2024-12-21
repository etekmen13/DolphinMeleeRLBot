#pragma once
#include <Windows.h>
#include <wrl/client.h>
#include <d3d11.h>
#include <dxgi.h>
#include <cuda_runtime.h>  
#include <cuda_d3d11_interop.h> 
#include <texture_indirect_functions.h>
#include "stb_image_write.h"

using Microsoft::WRL::ComPtr;

extern ComPtr<ID3D11Device> d3dDevice;
extern ComPtr<ID3D11DeviceContext> d3dContext;

#define EFFICIENT_NET_B0_DOWNSCALE 224
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif