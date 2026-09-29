import sys
from numba import cuda

def gpu_info():
    print ("1. numba.cuda.detect():")
    cuda.detect()

    devices = cuda.gpus
    num_devices = len(devices)
    print (f"Number of GPUs: {num_devices}\n")

    for device_id, device in enumerate(devices):
        cuda.select_device(device_id)
        free_mem, total_mem = cuda.current_context().get_memory_info()
        name = device.name.decode ("utf-8") if isinstance(device.name, bytes) else device.name

        print(f"GPU {device_id}: {name}")
        print(f"ID: {device_id}")
        print(f"Compute Capability: {device.compute_capability[0]}.{device.compute_capability[1]}")
        print(f"Multiprocessors (SMs): {device.MULTIPROCESSOR_COUNT}")
        print(f"Warp Size: {device.WARP_SIZE}")
        print(f"Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
        print(f"Max Block Dimensions: ({device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z})")
        print(f"Max Grid Dimensions: ({device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z})")
        print(f"Total Global Memory: {total_mem / (1024**3):.2f} GB ({total_mem:,} bytes)")
        print(f"Free Global Memory: {free_mem / (1024**3):.2f} GB ({free_mem:,} bytes)")

if __name__ == "__main__":
    gpu_info()