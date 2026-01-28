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
__global__ void sobelSharedMemory(unsigned char* input, unsigned char* output,
                                  int width, int height) {
    // ===== STEP 1: DECLARE SHARED MEMORY =====
    // 18×18 because 16×16 block + 1-pixel border on each side
    __shared__ unsigned char tile[18][18];
    
    // Thread indices within block (0-15)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // ===== STEP 2: LOAD TILE TO SHARED MEMORY =====
    // Each thread loads ONE pixel from global memory
    
    // Position in shared memory (offset by 1 for border)
    int sharedX = tx + 1;
    int sharedY = ty + 1;
    
    // Load main tile (each thread loads its own pixel)
    if (x < width && y < height) {
        tile[sharedY][sharedX] = input[y * width + x];
        //   ^^^^                      ^^^^
        //   shared memory             global memory
        //   (fast, on-chip)           (slow, DRAM)
    }
    
    // Load border/halo pixels (edge threads load extra)
    // Left border
    if (tx == 0 && x > 0) {
        tile[sharedY][0] = input[y * width + (x-1)];
    }
    // Right border
    if (tx == blockDim.x-1 && x < width-1) {
        tile[sharedY][sharedX+1] = input[y * width + (x+1)];
    }
    // Top border
    if (ty == 0 && y > 0) {
        tile[0][sharedX] = input[(y-1) * width + x];
    }
    // Bottom border
    if (ty == blockDim.y-1 && y < height-1) {
        tile[sharedY+1][sharedX] = input[(y+1) * width + x];
    }
    
    // ===== STEP 3: SYNCHRONIZE =====
    // Wait for ALL threads to finish loading
    __syncthreads();
    // Now the entire tile is in shared memory!
    
    // ===== STEP 4: COMPUTE USING SHARED MEMORY =====
    // Each thread reads from SHARED memory (fast!)
    if (x >= width || y >= height) return;
    if (x == 0 || y == 0 || x == width-1 || y == height-1) {
        output[y * width + x] = 0;
        return;
    }
    
    int gx = 0, gy = 0;
    
    // Read 3×3 neighborhood from SHARED memory
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            // Read from shared memory (fast!)
            unsigned char pixel = tile[sharedY + dy][sharedX + dx];
            //                    ^^^^
            //                    Reading from on-chip shared memory
            //                    NOT from slow global memory!
            
            gx += pixel * sobelX[dy+1][dx+1];
            gy += pixel * sobelY[dy+1][dx+1];
        }
    }
    
    int magnitude = sqrtf((float)(gx*gx + gy*gy));
    output[y * width + x] = min(magnitude, 255);
}
```

---

## Visual Step-by-Step

### Before Loading (initial state):
```
Global Memory:                  Shared Memory:
┌─────────────────┐            ┌──────────────────┐
│ a b c d e f ... │            │ ? ? ? ? ? ? ...  │ ← Empty/garbage
│ g h i j k l ... │            │ ? ? ? ? ? ? ...  │
│ m n o p q r ... │            │ ? ? ? ? ? ? ...  │
│ ...             │            │ ...              │
└─────────────────┘            └──────────────────┘
```

### After Loading (cooperative load):
```
Global Memory:                  Shared Memory:
┌─────────────────┐            ┌──────────────────┐
│ a b c d e f ... │  ──────>   │ a b c d e f ...  │ ← Copied once
│ g h i j k l ... │  ──────>   │ g h i j k l ...  │
│ m n o p q r ... │  ──────>   │ m n o p q r ...  │
│ ...             │            │ ...              │
└─────────────────┘            └──────────────────┘
```

### During Computation:

Each thread reads from **shared memory** (fast!):
```
Thread (5, 5) needs pixels around position (5,5):
┌───┬───┬───┐
│ i │ j │ k │  ← Read from tile[4][4], tile[4][5], tile[4][6]
├───┼───┼───┤     All from SHARED memory (on-chip)
│ o │ p │ q │  ← Read from tile[5][4], tile[5][5], tile[5][6]
├───┼───┼───┤
│ u │ v │ w │  ← Read from tile[6][4], tile[6][5], tile[6][6]
└───┴───┴───┘
```

---

## Why This is Fast

### Memory Hierarchy:
```
Speed (fastest to slowest):
1. Registers            (1 cycle)
2. Shared Memory        (4-40 cycles)    ← We use this!
3. L1/L2 Cache         (20-200 cycles)
4. Global Memory       (400-800 cycles)  ← We avoid this!
