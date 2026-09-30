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

def gs_cpu(flat_img):
    pixel_count = flat_img.shape[0]
    out = np.empty_like(flat_img)
    for i in range(pixel_count):
        g = np.uint8((int(flat_img[i, 0]) + int(flat_img[i, 1]) + int(flat_img[i, 2])) / 3)
        out[i,0] = out[i,1] = out[i,2] = g
    return out

def main():
    img = plt.imread("image.jpg")

    h,w,c = img.shape
    pixel_count = h*w
    print(f"Pixels: {pixel_count:,}")

    flat_src = img.reshape((pixel_count,3))

    start_cpu = time.time()
    cpu_result_flat = gs_cpu(flat_src)
    time_cpu = time.time() - start_cpu
    print(f"CPU execution time: {time_cpu:.4f}s")

    plt.imsave("output_gs_cpu.jpg", cpu_result_flat.reshape((h,w,3)))

    dev_src = cuda.to_device(flat_src)
    dev_dst = cuda.device_array((pixel_count,3), dtype=np.uint8)

    block_sizes = [32, 64, 128, 256, 512, 1024]
    gpu_times = []

    grid_init = math.ceil(pixel_count/32)
    gs_gpu[grid_init, 32](dev_src, dev_dst, pixel_count)
    cuda.synchronize()

    for b_size in block_sizes:
        grid_size = math.ceil(pixel_count/b_size)

        start_gpu = time.time()
        gs_gpu[grid_size, b_size](dev_src, dev_dst, pixel_count)
        cuda.synchronize()
        duration = time.time() - start_gpu

        gpu_times.append(duration)
        speedup = time_cpu / duration
        print (f"BlockSize: {b_size:4d} | GridSize: {grid_size:6d} | GPU Time: {duration*1000:7.2f} ms | Speedup: {speedup:6.1f}x")

    gpu_result_flat = dev_dst.copy_to_host()
    plt.imsave("output_gs_gpu.jpg", gpu_result_flat.reshape((h, w, 3)))

    plt.figure(figsize=(8, 5))
    plt.plot(block_sizes, [t * 1000 for t in gpu_times], marker='o', color='b', linewidth=2)
    plt.title("Block Size vs Execution Time (GPU)")
    plt.xlabel("Block Size (threads per block)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("block_size_vs_time.png", dpi=300)
    print("\nSucceed!!!")

if __name__ == "__main__":
    main()