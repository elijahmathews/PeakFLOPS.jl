module PeakFLOPS

using AMDGPU
using LinearAlgebra

include("utility_unroll.jl")

_flopcount_per_matmul(n) = Float64(n)^3

function peakflops_gpu_matmul(;
    device = AMDGPU.device(),
    dtype = Float32,
    size = 2^14,
    nmatmuls = 5,
    nbench = 5,
    verbose = true,
    io::IO = stdout,
)
    AMDGPU.device!(device) do
        C = AMDGPU.zeros(dtype, size, size)
        A = AMDGPU.rand(dtype, size, size)
        B = AMDGPU.rand(dtype, size, size)
        
        AMDGPU.@elapsed mul!(C, A, B) # warmup
        
        t = Inf
        for i in 1:nbench
            Δt = AMDGPU.@elapsed for _ in 1:nmatmuls
	        mul!(C, A, B)
	    end
	    t = min(t, Δt)
        end
        
        flops = (_flopcount_per_matmul(size) * nmatmuls * 1e-12) / t
        if verbose
            printstyled(io, "Peakflops (TFLOP/s):\n"; bold=true)
            print(io, " └ max: ")
            printstyled(io, round(flops; digits=2), "\n"; color=:green, bold=true)
        end
        return flops
    end
end

_kernel_fma_nfmas()::Int = 100_000
_kernel_fma_N()::Int = Int((_kernel_fma_nfmas() - 1) ÷ 3)

function _kernel_fma(a, b, c, out)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    @inbounds if i <= length(out)
        a_val = a[i]
	b_val = b[i]
	c_val = c[i]

	@unroll for j in 1:_kernel_fma_N()
	    a_val = fma(a_val, b_val, c_val)
	    b_val = fma(a_val, b_val, c_val)
	    c_val = fma(a_val, b_val, c_val)
	end

	out[i] = fma(a_val, b_val, c_val)
    end

    return nothing
end


function _peakflops_gpu_fmas(;
    size::Integer = 5_000_000,
    dtype = Float32,
    nbench = 5,
    nkernel = 5,
    device::HIPDevice = AMDGPU.device(),
    verbose = true,
    io::IO = stdout,
)
    AMDGPU.device!(device) do
        d_a = AMDGPU.rand(dtype, size)
	d_b = AMDGPU.rand(dtype, size)
	d_c = AMDGPU.rand(dtype, size)
	d_out = AMDGPU.zeros(dtype, size)

	if verbose
	    printstyled(io, "Building kernel...\n", italic=true)
	end

	kernel = @roc launch=false _kernel_fma(d_a, d_b, d_c, d_out)
	config = AMDGPU.launch_configuration(kernel)
	groupsize = min(size, config.groupsize)
	gridsize = cld(size, groupsize)

	AMDGPU.@elapsed kernel(d_a, d_b, d_c, d_out)

	if verbose
	    printstyled(io, "Running test...\n", italic=true)
	end

	t = Inf
	for _ in 1:nbench
	    Δt = AMDGPU.@elapsed begin
	        for _ in 1:nkernel
	            kernel(d_a, d_b, d_c, d_out; groupsize=groupsize, gridsize=gridsize)
	        end
	    end
	    t = min(t, Δt)
	end
	t /= nkernel

	flopcount = 2 * _kernel_fma_nfmas() * size
	flops = (flopcount * 1e-12) / t

	if verbose
	    printstyled(io, "Peakflops (TFLOP/s):\n"; bold=true)
	    print(io, " └ max: ")
	    printstyled(io, round(flops; digits=2), "\n"; color=:green, bold=true)
        end
	return flops
    end
end

end # module PeakFLOPS
