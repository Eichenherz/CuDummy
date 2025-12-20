#ifndef __WARPS_H__
#define __WARPS_H__

#include "core_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bit>


constexpr u64 WARP_SIZE = 32u;
constexpr u64 WARP_SZ_SHIFT = std::bit_width( WARP_SIZE ) - 1u;
static_assert( std::has_single_bit( WARP_SIZE ), "WARP_SIZE not POT" );

__device__ inline u64 LaneId() { return threadIdx.x & ( WARP_SIZE - 1 ); }
__device__ inline u64 WarpId() { return threadIdx.x >> WARP_SZ_SHIFT; }

// NOTE: pre-Turing gpus don't support __reduce warp ops
template<typename T>
__device__ T WarpReduceShflDownSync( const T in )
{
    T sum = in;
#pragma unroll
    for( u64 offsetWithinWarp = WARP_SIZE >> 1; offsetWithinWarp > 0; offsetWithinWarp >>= 1 ) 
    {
        sum += __shfl_down_sync( u32( -1 ), sum, offsetWithinWarp );
    }

    return sum;
}

template<typename T>
__device__ T WarpInclusiveScanShflUpSync( const T val )
{
    T inclusvieScan = val;
#pragma unroll
    for( u64 offset = 1; offset < WARP_SIZE; offset <<= 1 )
    {
        const T warpLaneValAtOffset = __shfl_up_sync( 0xffffffff, inclusvieScan, offset );
        if( LaneId() >= offset ) // NOTE: Distributes values
        {
            inclusvieScan += warpLaneValAtOffset;
        }
    }
    return inclusvieScan;
};



#endif // !__WARPS_H__
