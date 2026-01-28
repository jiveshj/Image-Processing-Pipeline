// Sobel operators
__constant__ int sobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int sobelY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__global__ void sobelNaive(unsigned char* input, unsigned char* output, 
                           int width, int height) {
    // Each thread handles one output pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    if (x == 0 || y == 0 || x == width-1 || y == height-1) {
        output[y * width + x] = 0;  // Border pixels = 0
        return;
    }
    
    int gx = 0, gy = 0;
    
    // Convolve with Sobel kernels
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int pixel = input[(y + dy) * width + (x + dx)];
            gx += pixel * sobelX[dy + 1][dx + 1];
            gy += pixel * sobelY[dy + 1][dx + 1];
        }
    }
    
    // Compute gradient magnitude
    int magnitude = sqrtf((float)(gx*gx + gy*gy));
    output[y * width + x] = min(magnitude, 255);
}
