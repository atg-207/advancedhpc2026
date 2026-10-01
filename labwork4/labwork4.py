import math
import time 
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda 

@cuda.jit
def gs_gpu(src, dst, pixel_count):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if tidx < pixel_count:
        g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
        dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

@cuda.jit
def gs_2d_gpu(src, dst, w, h):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if x < w and y < h:
        r = src[y, x, 0]
        g = src[y, x, 1]
        b = src[y, x, 2]
        gray = np.uint8((int(r) + int(g) + int(b)) / 3)
        dst[y, x, 0] = gray
        dst[y, x, 1] = gray
        dst[y, x, 2] = gray

def gs_2d_cpu(img):
    h, w, c = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            gray = np.uint8((int(img[y, x, 0]) + int(img[y, x, 1]) + int(img[y, x, 2])) // 3)
            out[y, x, 0] = gray
            out[y, x, 1] = gray
            out[y, x, 2] = gray
    return out

def main():
    img = plt.imread("image.jpg")

    h, w, _ = img.shape
    pixel_count = h * w
    print(f"Pixels: {pixel_count:,}")

    start_cpu = time.time()
    cpu_result = gs_2d_cpu(img)
    time_cpu = time.time() - start_cpu
    print(f"CPU execution time: {time_cpu:.4f}s")
    plt.imsave("output_gs_cpu.jpg", cpu_result)

    dev_src_2d = cuda.to_device(img)
    dev_dst_2d = cuda.device_array((h, w, 3), dtype=np.uint8)

    flat_src = img.reshape((pixel_count, 3))
    dev_src_1d = cuda.to_device(flat_src)
    dev_dst_1d = cuda.device_array((pixel_count, 3), dtype=np.uint8)

    NUM_ITERS = 5
    print(f"Number of iteration: {NUM_ITERS}")

    block_sizes_1d = [32, 64, 128, 256, 512, 1024]
    speedups_1d = []
    avg_times_1d = []
    labels_1d = [str(b) for b in block_sizes_1d]

    grid_init_1d = math.ceil(pixel_count / 32)
    gs_gpu[grid_init_1d, 32](dev_src_1d, dev_dst_1d, pixel_count)
    cuda.synchronize()

    print("\n1D GPU:")
    for b_size in block_sizes_1d:
        grid_size = math.ceil(pixel_count / b_size)

        iter_times = []
        for _ in range(NUM_ITERS):
            start_gpu = time.time()
            gs_gpu[grid_size, b_size](dev_src_1d, dev_dst_1d, pixel_count)
            cuda.synchronize()
            iter_times.append(time.time() - start_gpu)

        avg_time = sum(iter_times) / NUM_ITERS
        speedup = time_cpu / avg_time
        avg_times_1d.append(avg_time * 1000)
        speedups_1d.append(speedup)
        print(f"BlockSize: {b_size:4d} | GridSize: {grid_size:6d} | GPU Time: {avg_time*1000:6.2f} ms | Speedup: {speedup:6.1f}x")

    block_sizes_2d = [(4, 4), (8, 8), (16, 16), (32, 16), (32, 32)] 
    speedups_2d = []
    avg_times_2d = []
    labels_2d = [f"{bx}x{by}" for bx, by in block_sizes_2d]

    grid_init_2d = (math.ceil(w / 16), math.ceil(h / 16))
    gs_2d_gpu[grid_init_2d, (16, 16)](dev_src_2d, dev_dst_2d, w, h)
    cuda.synchronize()

    print("\n2D GPU:")
    for bx, by in block_sizes_2d:
        gx = math.ceil(w / bx)
        gy = math.ceil(h / by)

        iter_times = []
        for _ in range(NUM_ITERS):
            start_gpu = time.time()
            gs_2d_gpu[(gx, gy), (bx, by)](dev_src_2d, dev_dst_2d, w, h)
            cuda.synchronize()
            iter_times.append(time.time() - start_gpu)

        avg_time = sum(iter_times) / NUM_ITERS
        speedup = time_cpu / avg_time
        avg_times_2d.append(avg_time * 1000)
        speedups_2d.append(speedup)
        print(f"Block: ({bx:2d}, {by:2d}) [{bx*by:4d} th] | Grid: ({gx:4d}, {gy:4d}) | Time: {avg_time*1000:6.2f} ms | Speedup: {speedup:6.1f}x")

    gpu_result = dev_dst_2d.copy_to_host()
    plt.imsave("output_gs_gpu.jpg", gpu_result)

    plt.figure(figsize=(10, 5))
    plt.plot(labels_1d, speedups_1d, marker='s', linestyle='--', color='tab:orange', linewidth=2, label='1D Speedup')
    plt.plot(labels_2d, speedups_2d, marker='o', color='tab:blue', linewidth=2, label='2D Speedup')
    plt.title("1D vs 2D Block Size vs Speedup", fontweight='bold')
    plt.xlabel("Block Size Configuration", fontweight='bold')
    plt.ylabel("Speedup (x CPU)", fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("block_size_vs_speedup.png", dpi=300)
    print("\nSucceed!!!")

if __name__ == "__main__":
    main()