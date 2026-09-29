from numba import cuda

CORES_PER_SM = {
    (5, 0): 128, (5, 2): 128,          # Maxwell
    (6, 0): 64,  (6, 1): 128,          # Pascal
    (7, 0): 64,  (7, 5): 64,           # Volta, Turing
    (8, 0): 64,  (8, 6): 128, (8, 9): 128, # Ampere, Ada Lovelace
    (9, 0): 128                        # Hopper
}

def gpu_info():
    print("1. numba.cuda.detect():")
    cuda.detect()

    devices = cuda.gpus
    print(f"Number of GPUs: {len(devices)}\n")

    for device_id, device in enumerate(devices):
        cuda.select_device(device_id)
        free_mem, total_mem = cuda.current_context().get_memory_info()
        name = device.name.decode("utf-8") if isinstance(device.name, bytes) else device.name
        
        # Tính số CUDA cores
        sms = device.MULTIPROCESSOR_COUNT
        cores_per_sm = CORES_PER_SM.get(device.compute_capability, 64)
        total_cores = sms * cores_per_sm

        print(f"GPU {device_id}: {name}")
        print(f"ID: {device_id}")
        print(f"Multiprocessors (SMs): {sms}")
        print(f"CUDA Cores: {total_cores} ({sms} SMs x {cores_per_sm} cores/SM)")
        print(f"Total Global Memory: {total_mem / (1024**3):.2f} GB ({total_mem:,} bytes)")
        print(f"Free Global Memory: {free_mem / (1024**3):.2f} GB ({free_mem:,} bytes)")

if __name__ == "__main__":
    gpu_info()