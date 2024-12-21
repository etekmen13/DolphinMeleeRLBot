#include "globals.h"
#include <iostream>
#include <codecvt>
#include <d3dcompiler.h>
#include <vector>
#include "TexturePreprocessing.h"
using Microsoft::WRL::ComPtr;
struct SharedBufferData
{
    HANDLE sharedHandle;
    UINT width;
    UINT height;
};

HANDLE hMapFile = nullptr;
HANDLE hSharedEvent = nullptr;

ComPtr<ID3D11Device> d3dDevice;
ComPtr<ID3D11DeviceContext> d3dContext;

bool InitializeDirect3D()
{
    int dev;

    IDXGIFactory1* dxgiFactory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&dxgiFactory);
    if (FAILED(hr))
    {
        std::cerr << "Failed to create DXGI Factory!" << std::endl;
        return false;
    }
    ComPtr<IDXGIAdapter1> adap;
    for (unsigned int i = 0; !adap.Get(); ++i) {
        if (FAILED(dxgiFactory->EnumAdapters1(i, adap.GetAddressOf())))
            break;
        if (cudaD3D11GetDevice(&dev, adap.Get()) == cudaSuccess)
            break;
            adap.Reset();
    }
    dxgiFactory->Release();
    DXGI_ADAPTER_DESC1 my_adapter_desc;
    adap->GetDesc1(&my_adapter_desc);
    std::wstring w_description(my_adapter_desc.Description);
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string description = converter.to_bytes(w_description);
    std::cout << "Description: " << description << std::endl;
    D3D_FEATURE_LEVEL featureLevel;
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3
    };

    D3D11_CREATE_DEVICE_FLAG deviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

     hr = D3D11CreateDevice(
        nullptr, // Use default adapter
        D3D_DRIVER_TYPE_HARDWARE, // Use hardware device
        nullptr, // No software device
        deviceFlags, // Device creation flags
        featureLevels, ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION, // SDK version
        &d3dDevice, // The created device
        &featureLevel, // The feature level
        &d3dContext // The device context
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device with hardware driver" << std::endl;
        return false;
    }

    // Register the D3D11 device with CUDA
    cudaError_t result = cudaD3D11SetDirect3DDevice(d3dDevice.Get());
    if (result != cudaSuccess) {
        std::cerr << "CUDA failed to set Direct3D device: " << cudaGetErrorString(result) << std::endl;
        return false;
    }

    return true;
}
void StageandSavePNG(ComPtr<ID3D11Texture2D> shared_texture, UINT width, UINT height) {
    ComPtr<ID3D11Texture2D> staging_texture;
    D3D11_TEXTURE2D_DESC staging_desc;
    shared_texture->GetDesc(&staging_desc);
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; 
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;  // Staging textures cannot have bind flags

    HRESULT hr = d3dDevice->CreateTexture2D(&staging_desc, nullptr, staging_texture.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: " << std::hex << hr << std::endl;
        return;
    }
    d3dContext->Flush(); // Ensure all GPU commands are executed

    d3dContext->CopyResource(staging_texture.Get(), shared_texture.Get());
    // Map the staging texture to CPU memory
    D3D11_MAPPED_SUBRESOURCE mappedResource;
     hr = d3dContext->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: " << std::hex << hr << std::endl;
        return;
    }
    BYTE* data = reinterpret_cast<BYTE*>(mappedResource.pData);

    for (int i = 0; i < width * 4 * sizeof(uint8_t); i += 4) {
        std::cout << (int)data[i] << std::endl;
    }
    if (!stbi_write_png("before_cuda.png", width, height, 4, data, mappedResource.RowPitch)) {
        std::cerr << "failed to write DX11 as PNG" << std::endl;
    }
    //// Unmap and release the staging texture
    d3dContext->Unmap(staging_texture.Get(), 0);
    staging_texture.Reset();
}
void ReadSharedBuffer()
{
    while (true)
    {
        WaitForSingleObject(hSharedEvent, INFINITE);

        // Map the shared memory to retrieve the buffer handle and metadata
        void* pMappedMem = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, sizeof(SharedBufferData));
        if (!pMappedMem)
        {
            std::cerr << "Failed to map shared memory." << std::endl;
            continue;
        }

        SharedBufferData* sharedData = static_cast<SharedBufferData*>(pMappedMem);
        HANDLE sharedHandle = sharedData->sharedHandle;
        UINT width = sharedData->width; // Size of the buffer in bytes
        UINT height = sharedData->height;
        UnmapViewOfFile(pMappedMem);

        // Open the shared buffer
        ComPtr<ID3D11Texture2D> shared_texture;
        HRESULT hr = d3dDevice->OpenSharedResource(sharedHandle, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(shared_texture.GetAddressOf()));
        if (FAILED(hr))
        {
            std::cerr << "Failed to open shared buffer. HRESULT: " << hr << std::endl;

            HRESULT reason = d3dDevice->GetDeviceRemovedReason();
            if (reason == DXGI_ERROR_DEVICE_HUNG)
                std::cerr << "Device hung (DXGI_ERROR_DEVICE_HUNG)" << std::endl;
            else if (reason == DXGI_ERROR_DEVICE_RESET)
                std::cerr << "Device reset (DXGI_ERROR_DEVICE_RESET)" << std::endl;
            else if (reason == DXGI_ERROR_DRIVER_INTERNAL_ERROR)
                std::cerr << "Driver internal error (DXGI_ERROR_DRIVER_INTERNAL_ERROR)" << std::endl;
            else if (reason == DXGI_ERROR_INVALID_CALL)
                std::cerr << "Invalid call (DXGI_ERROR_INVALID_CALL)" << std::endl;
            else
                std::cerr << "Unknown device removal reason: " << std::hex << reason << std::endl;
            continue;
        }
        StageandSavePNG(shared_texture, width, height);
        d3dContext->Flush();

        CopyAndPreprocess(shared_texture, width, height); // Custom function to handle the buffer data

        // Unmap the buffer
        // Signal that the processing is complete
        SetEvent(hSharedEvent);
    }
}


int main()
{

    hMapFile = OpenFileMappingW(FILE_MAP_READ, FALSE, L"D3DSharedHandleMapping");
    if (!hMapFile)
    {
        std::cerr << "Failed to open shared memory mapping." << std::endl;
        return -1;
    }

    hSharedEvent = OpenEventW(EVENT_ALL_ACCESS, FALSE, L"DolphinRLSharedEvent");
    if (!hSharedEvent)
    {
        std::cerr << "Failed to open shared event." << std::endl;
        CloseHandle(hMapFile);
        return -1;
    }

    if (!InitializeDirect3D())
    {
        CloseHandle(hMapFile);
        CloseHandle(hSharedEvent);
        return -1;
    }

    SetEvent(hSharedEvent);

    ReadSharedBuffer();

    CloseHandle(hMapFile);
    CloseHandle(hSharedEvent);

    return 0;
}
