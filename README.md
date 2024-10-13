# PeakFLOPS.jl

This is just a basic library I made for benchmarking my AMD integrated GPU after I got ROCm and [AMDGPU.jl](https://www.github.com/JuliaGPU/AMDGPU.jl) up and running on my system (AMD Radeon 780M, Fedora 40). This is mostly a translation of [GPUInspector.jl](https://github.com/pc2/GPUInspector.jl) to be compatible with AMDGPU.jl.

The main function is the `_peakflops_gpu_fmas` function, which benchmarks the GPU (in units of TFLOPS) for a fused multiply-add operation (FMA).

```julia
julia> using PeakFlops

julia> PeakFlops._peakflops_gpu_fmas()
```
