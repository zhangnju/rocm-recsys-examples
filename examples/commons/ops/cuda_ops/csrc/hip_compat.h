/*
 * ROCm/HIP compatibility shims.
 * Maps CUDA-specific types and APIs to their HIP equivalents so the same
 * source compiles with both nvcc and hipcc (via PyTorch ROCm).
 *
 * Supported AMD GPU architectures:
 *   gfx942  – MI300X / MI300A
 *   gfx950  – MI355X
 *   (gfx940, gfx941 also covered)
 *
 * This header is included first in every .cu / .cpp source file.
 * On NVIDIA it is a no-op (the real CUDA headers are used).
 */
#pragma once

#ifdef __HIP_PLATFORM_AMD__

/* ---- Core HIP runtime headers ---- */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

/* ---- CUDA half / bfloat type aliases ---- */
/*  __half   is already defined by hip_fp16.h as the 16-bit float type.
 *  For bfloat16 PyTorch ROCm uses __hip_bfloat16 (from amd_hip_bf16.h).
 *  We expose nv_* aliases used in template specialisations below.       */
using nv_half         = __half;
using nv_bfloat16     = __hip_bfloat16;
using nv_bfloat162    = __hip_bfloat162;
/* __nv_bfloat16 and __nv_bfloat162 may be needed by some template checks */
#ifndef __nv_bfloat16
#define __nv_bfloat16  __hip_bfloat16
#endif
#ifndef __nv_bfloat162
#define __nv_bfloat162 __hip_bfloat162
#endif

/* ---- CUDA → HIP runtime macro aliases ---- */
#define cudaStream_t                              hipStream_t
#define cudaEvent_t                               hipEvent_t
#define cudaEventCreate(e)                        hipEventCreate(e)
#define cudaEventRecord(e, s)                     hipEventRecord(e, s)
#define cudaEventSynchronize(e)                   hipEventSynchronize(e)
#define cudaEventDestroy(e)                       hipEventDestroy(e)
#define cudaStreamCreate(s)                       hipStreamCreate(s)
#define cudaStreamDestroy(s)                      hipStreamDestroy(s)
#define cudaStreamSynchronize(s)                  hipStreamSynchronize(s)
#define cudaStreamWaitEvent(s, e, f)              hipStreamWaitEvent(s, e, f)
#define cudaGetDevice(d)                          hipGetDevice(d)
#define cudaSetDevice(d)                          hipSetDevice(d)
#define cudaDeviceGetAttribute(v, a, d)           hipDeviceGetAttribute(v, a, d)
#define cudaDevAttrMultiProcessorCount            hipDeviceAttributeMultiprocessorCount
#define cudaMalloc(p, s)                          hipMalloc(p, s)
#define cudaFree(p)                               hipFree(p)
#define cudaMallocHost(p, s)                      hipHostMalloc(p, s, 0)
#define cudaFreeHost(p)                           hipHostFree(p)
#define cudaMemcpy(d, s, n, k)                    hipMemcpy(d, s, n, k)
#define cudaMemcpyAsync(d, s, n, k, st)           hipMemcpyAsync(d, s, n, k, st)
#define cudaMemcpyHostToDevice                    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost                    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                  hipMemcpyDeviceToDevice
#define cudaMemset(p, v, n)                       hipMemset(p, v, n)
#define cudaSuccess                               hipSuccess
#define cudaErrorInvalidValue                     hipErrorInvalidValue
#define cudaLaunchKernel                          hipLaunchKernel
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
        hipOccupancyMaxActiveBlocksPerMultiprocessor

/* ---- __ldg: HIP supports __ldg natively for scalar types ---- */
#ifndef __ldg
#define __ldg(ptr) (*(ptr))
#endif

/* ---- __umulhi: HIP already provides this in amd_device_functions.h ---- */
/* Do NOT redefine; it's available after including hip_runtime.h          */

/* ---- Warp-level: __shfl_sync → __shfl ---- */
/* HIP wavefronts are 64-wide by default on MI300/MI355 but warp-level     */
/* ops use 32-wide SIMD groups when launched with 32-thread warps.          */
/* The mask argument is ignored on HIP (all active lanes participate).      */
#define __shfl_sync(mask, val, src_lane)       __shfl(val, src_lane)
/* Note: the 4-arg form with width is rare; add if needed:                 */
/* #define __shfl_sync(mask,val,src,width)  __shfl(val,src,width)          */

/* ---- FP8 support: conditional on architecture ---- */
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx941__) || defined(__gfx940__)
#define HW_FP8_CONVERSION_ENABLED
#include <hip/hip_fp8.h>
#endif

/* ---- cudaCheck helper ---- */
#include <cstdio>
#include <cstdlib>
#ifndef hipCheck
#define hipCheck(call)                                                         \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                        \
                    hipGetErrorString(_e), __FILE__, __LINE__);                \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif
#define cudaCheck(call) hipCheck(call)

#else  /* ---- NVIDIA CUDA path: minimal shims ---- */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using nv_half     = __half;
using nv_bfloat16 = __nv_bfloat16;
using nv_bfloat162= __nv_bfloat162;

#ifndef cudaCheck
#include <cstdio>
#include <cstdlib>
#define cudaCheck(call)                                                        \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif

#endif /* __HIP_PLATFORM_AMD__ */
