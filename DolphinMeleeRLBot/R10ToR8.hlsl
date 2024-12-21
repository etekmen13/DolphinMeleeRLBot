[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    // This shader assumes the input texture has the DXGI_FORMAT_R10G10B10A2_UNORM format
  // and the output texture is in DXGI_FORMAT_R8G8B8A8_UNORM format.

    RWTexture2D<float4> outputTexture : register(u0);
    Texture2D<float4> inputTexture : register(t0);

    [numthreads(16, 16, 1)]
    void main(uint3 dispatchThreadID : SV_DispatchThreadID)
    {
        // Get the texel color from the input texture in R10G10B10A2_UNORM format
        uint4 color10 = inputTexture.Load(dispatchThreadID.xy);

        // Extract 10-bit channels for Red, Green, Blue and 2-bit Alpha
        uint red10 = color10.x;
        uint green10 = color10.y;
        uint blue10 = color10.z;
        uint alpha2 = color10.w;

        // Perform bit shifting and masking to convert to 8-bit channels
        uint red8 = (red10 >> 2) & 0xFF;        // Shift right by 2 bits and mask to 8 bits
        uint green8 = (green10 >> 2) & 0xFF;    // Same for green
        uint blue8 = (blue10 >> 2) & 0xFF;      // Same for blue
        uint alpha8 = (alpha2 << 6);            // Shift alpha from 2 bits to 8 bits

        // Write the converted color to the output texture
        outputTexture[dispatchThreadID.xy] = float4(red8 / 255.0f, green8 / 255.0f, blue8 / 255.0f, alpha8 / 255.0f);
    }

}