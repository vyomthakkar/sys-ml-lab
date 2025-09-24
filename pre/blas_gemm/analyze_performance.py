def analyze_performance(results):
    """Analyze whether kernel is memory or compute bound"""
    flops = results['flops']
    memory_bytes = results['memory_bytes']
    naive_time = results['naive_time']
    
    arithmetic_intensity = flops / memory_bytes
    memory_bandwidth_used = memory_bytes / naive_time / 1e9  # GB/s
    
    print(f"\n=== Performance Analysis ===")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
    print(f"Memory BW used (naive): {memory_bandwidth_used:.2f} GB/s")
    print(f"Peak memory BW estimate: ~50-200 GB/s (typical CPU)")
    
    if arithmetic_intensity < 1:
        print("→ Likely MEMORY-BOUND (more data movement than compute)")
    elif arithmetic_intensity > 10:
        print("→ Likely COMPUTE-BOUND (more math than data movement)")
    else:
        print("→ BALANCED (memory and compute similar)")