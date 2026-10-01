import math
import time 
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda 

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
            gray = np.uint8((int(img[y, x, 0]) + int(img[y, x, 1]) + int(img[y, x, 2])) / 3)
            out[y, x, 0] = gray
            out[y, x, 1] = gray
            out[y, x, 2] = gray
    return out

def main():
    img = plt.imread("image.jpg")

    h,w,_ = img.shape
    pixel_count = h*w
    print(f"Pixels: {pixel_count:,}")

    start_cpu = time.time()
    cpu_result = gs_2d_cpu(img)
    time_cpu = time.time() - start_cpu
    print(f"CPU execution time: {time_cpu:.4f}s")
    plt.imsave("output_gs_cpu.jpg", cpu_result)

    dev_src = cuda.to_device(img)
    dev_dst = cuda.device_array((h,w,3), dtype=np.uint8)

    block_sizes = [(4, 4), (8, 8), (16, 16), (32, 16), (32, 32)] 
    speedups = []
    avg_gpu_times = []
    labels = [f"{bx}x{by}" for bx, by in block_sizes]
    NUM_ITERS = 5

    grid_init = (math.ceil(w / 16), math.ceil(h / 16))
    gs_2d_gpu[grid_init, (16, 16)](dev_src, dev_dst, w, h)
    cuda.synchronize()

    print(f"Number of iteration: {NUM_ITERS}")
    for bx, by in block_sizes:
        gx = math.ceil(w / bx)
        gy = math.ceil(h / by)

        iter_times = []
        for _ in range(NUM_ITERS):
            start_gpu = time.time()
            gs_2d_gpu[(gx, gy), (bx, by)](dev_src, dev_dst, w, h)
            cuda.synchronize()
            iter_times.append(time.time() - start_gpu)

        avg_time = sum(iter_times) / NUM_ITERS
        speedup = time_cpu / avg_time

        avg_gpu_times.append(avg_time * 1000)
        speedups.append(speedup)
        
        print(f"Block: ({bx:2d}, {by:2d}) [{bx*by:4d} th] | Grid: ({gx:4d}, {gy:4d}) | Time: {avg_time*1000:6.2f} ms | Speedup: {speedup:6.1f}x")

    gpu_result = dev_dst.copy_to_host()
    plt.imsave("output_gs_gpu.jpg", gpu_result)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Block Size (bx x by) [Total Threads]', fontweight='bold')
    ax1.set_ylabel('Speedup (x CPU)', color=color, fontweight='bold')
    line1 = ax1.plot(labels, speedups, marker='o', color=color, linewidth=2, label='Speedup')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (ms)', color=color, fontweight='bold')
    line2 = ax2.plot(labels, avg_gpu_times, marker='s', linestyle='--', color=color, linewidth=2, label='GPU Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("2D Block Size vs Speedup and Execution Time", fontweight='bold')
    fig.tight_layout()
    plt.savefig("block_size_vs_speedup.png", dpi=300)
    print("\nSucceed!!!")

if __name__ == "__main__":
    main()