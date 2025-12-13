#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <bit>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <span>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <execution>
#include <functional>
#include <ranges>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

#define CUDA_CHECK( call )                                                 \
    do {                                                                   \
        const cudaError_t err = call;                                      \
        if( cudaSuccess != err )                                           \
        {                                                                  \
            fprintf( stderr, "CUDA ERROR %s:%d: %s (%d)\n",                \
                    __FILE__, __LINE__, cudaGetErrorString( err ), err );  \
            abort();                                                       \
        }                                                                  \
    } while( 0 )

template<
    std::ranges::contiguous_range Range1, 
    std::ranges::contiguous_range Range2, 
    typename CmpOp>
inline bool ElementWiseRangeStrictCompare( const Range1 & a, const Range2 & b, CmpOp cmpOp )
{
    if( std::size( a ) != std::size( b ) )
    {
        return false;
    }

    for( u64 i = 0; i < std::size( a ); ++i )
    {
        assert( cmpOp( a[ i ], b[ i ] ) );
    }
    return true;
    //return std::transform_reduce( 
    //    std::execution::par, std::cbegin( a ), std::cend( a ), std::cbegin( b ), true, std::logical_and<>(), cmpOp );
}

template<typename T>
concept Number32BitsMax = ( sizeof( T ) <= 4 );

// NOTE: always POT
constexpr u64 WARP_SIZE = 32u;
static_assert( std::has_single_bit( WARP_SIZE ), "WARP_SIZE not POT" );
constexpr u64 WAPR_SZ_SHIFT = std::bit_width( WARP_SIZE ) - 1u;

__device__ inline u64 LaneId() { return threadIdx.x & ( WARP_SIZE - 1 ); }
__device__ inline u64 WarpId() { return threadIdx.x >> WAPR_SZ_SHIFT; }

template<typename T>
__device__ T WarpReduceShflDownSync( const T in )
{
    T sum = in;
#pragma unroll
    for( u64 offsetWithinWarp = WARP_SIZE >> 1; offsetWithinWarp > 0; offsetWithinWarp >>= 1 ) 
    {
        sum += __shfl_down_sync( 0xffffffff, sum, offsetWithinWarp );
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

constexpr u64 MAX_WARPS_PER_BLOCK = 32;

template<u64 SIZE, typename T>
__device__ void InitSharedMem( T* mem, T value )
{
    if( threadIdx.x < SIZE ) 
    {
        mem[ threadIdx.x ] = value;
    }
}

template<u64 THREADS_PER_BLOCK_X, typename T>
__global__ void KernelReduceBlocksWithWARP( const T* input, u64 workElemCount, T* partialSums )
{
    constexpr u64 WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;
    static_assert( WARPS_PER_BLOCK <= MAX_WARPS_PER_BLOCK, "ERR: Block has more warps !" );

    __shared__ T sharedPartialWarpReductions[ WARP_SIZE ]; 
    // NOTE: only need to zero init if the sharedPartialWarpReductions array is larger than the actual Wraps per Block
    if( threadIdx.x < WARP_SIZE ) 
    {
        sharedPartialWarpReductions[ threadIdx.x ] = T{};
    }
    __syncthreads();

    const u64 globalDataBlockOffset = blockDim.x * blockIdx.x; 
    const u64 globalIdx = globalDataBlockOffset + threadIdx.x;

    T currentElemToSum = T{};
    if( globalIdx < workElemCount )
    {
        currentElemToSum = input[ globalIdx ];
    }
    
    const T currentWarpSum = WarpReduceShflDownSync( currentElemToSum );
    const u64 laneID = LaneId();
    if( laneID == 0 )
    {
        const u64 warpID = WarpId();
        sharedPartialWarpReductions[ warpID ] = currentWarpSum;
    }
    __syncthreads();

    // NOTE: to avoid writing an unnecessay constexpr branch AND only thread0 will write this
    T thisBlockSum = currentWarpSum;
    if constexpr( WARPS_PER_BLOCK > 1 )
    {
        if( threadIdx.x < WARP_SIZE ) 
        {
            const T currentPartialWarpSum = sharedPartialWarpReductions[ threadIdx.x ];
            // TODO: if this incurrs a perf issue ( due to wasted lanes ) we can use a mask to select only the active threads
            constexpr u32 ACTIVE_LANES_MASK = ( 1u << WARPS_PER_BLOCK ) - 1;
            thisBlockSum = WarpReduceShflDownSync( currentPartialWarpSum );
        }
    }
  
    if( threadIdx.x == 0 ) 
    {
        partialSums[ blockIdx.x ] = thisBlockSum;
    }
}

template<typename T>
__device__ T ThreadBlockInclusiveScanWithSync( const T currentThreadValue )
{
    __shared__ T sharedWarpExclusiveScans[ WARP_SIZE ];
    // NOTE: only need to zero init if the sharedPartialWarpReductions array is larger than the actual Wraps per Block
    if( threadIdx.x < WARP_SIZE ) 
    {
        sharedWarpExclusiveScans[ threadIdx.x ] = T{};
    }
    __syncthreads();

    const T warpInclusiveScanCurrentLane = WarpInclusiveScanShflUpSync( currentThreadValue );
    // NOTE: the suffle up will have place the complete warp scan in the last lane !
    if( LaneId() == ( WARP_SIZE - 1 ) )
    {
        sharedWarpExclusiveScans[ WarpId() ] = warpInclusiveScanCurrentLane;
    }
    __syncthreads();

    if( threadIdx.x < WARP_SIZE )
    {
        const T warpExclusiveScan = sharedWarpExclusiveScans[ threadIdx.x ];
        // NOTE: we need exclusive bc we need offset without current sum
        sharedWarpExclusiveScans[ threadIdx.x ] = WarpInclusiveScanShflUpSync( warpExclusiveScan ) - warpExclusiveScan;
    }
    __syncthreads();
    
    const T threadScan = warpInclusiveScanCurrentLane + sharedWarpExclusiveScans[ WarpId() ];
    __syncthreads();

    return threadScan;
}

enum prefix_block_flags : u8
{
    UNAVAILABLE = 0,
    HAS_LOCAL_PREFIX = 1,
    HAS_FULL_PREFIX = 2
};

enum class prefix_scan_t : u8
{
    INCLUSIVE,
    EXCLUSIVE
};

struct alignas( u64 ) prefix_block_state
{
    u32 scan = 0;
    u32 flag = 0;

    __host__ __device__ inline explicit operator u64() const
    {
        return std::bit_cast<u64>( *this );
    }
};

__host__ __device__ inline prefix_block_state PrefixBlockStateFromU64( u64 addr )
{
    return std::bit_cast<prefix_block_state>( addr );
}
static_assert( alignof( prefix_block_state ) == alignof( u64 ), "prefix_block_state doesn't obey alignment requirement !" );
static_assert( sizeof( prefix_block_state ) == sizeof( u64 ), "prefix_block_state doesn't obey size requirement !" );


template<prefix_scan_t SCAN_TYPE, u64 THREADS_PER_BLOCK_X, Number32BitsMax T>
__global__ void KernelChainPrefixScanWithDecoupledLookback( 
    const T*                 input,
    u64                      workElemCount,
    u64*                     globalGroupCounter,
    u64*                     globalBlockStates,
    T*                       scannedOut 
) {
    constexpr u64 WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;
    static_assert( WARPS_PER_BLOCK <= MAX_WARPS_PER_BLOCK, "ERR: Block has more warps !" );

    // NOTE: need to get the dynamic idx of the block
    __shared__ u64 sharedCurrerntBlockIdx;
    if( threadIdx.x == 0 )
    {
        sharedCurrerntBlockIdx = atomicAdd( globalGroupCounter, 1 );
        // NOTE: zero init
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], 0u );
    }   
    __syncthreads();

    // NOTE: need to use the dynamicBlockIdx to get the corresponding work
    const u64 globalDataBlockOffset = blockDim.x * sharedCurrerntBlockIdx;
    const u64 globalIdx = globalDataBlockOffset + threadIdx.x;

    T currentElemToSum = T{};
    if( globalIdx < workElemCount )
    {
        currentElemToSum = input[ globalIdx ];
    }

    const T blockScanThreadElem = ThreadBlockInclusiveScanWithSync( currentElemToSum );

    __shared__ T sharedLocalBlockScan;
    if( threadIdx.x == ( blockDim.x - 1 ) )
    {
        sharedLocalBlockScan = blockScanThreadElem;
    }
    __syncthreads();

    
    __shared__ T sharedPrevScan;
    // NOTE: here we force order the blocks, but we'll keep looking backwards until we find a PREFIX STATE
    if( threadIdx.x == 0 )
    {
        prefix_block_state currentBlockState = {
            .scan = std::bit_cast< u32 >( sharedLocalBlockScan ), .flag = prefix_block_flags::HAS_LOCAL_PREFIX };
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], u64( currentBlockState ) );

        const bool isNotFirstBlock = 0 != sharedCurrerntBlockIdx;

        T lookbackScan = T{};
        for( i64 lookbackBlockIdx = sharedCurrerntBlockIdx - 1; isNotFirstBlock && lookbackBlockIdx >= 0; )
        {
            const prefix_block_state currentLookbackBlockState =
                PrefixBlockStateFromU64( atomicAdd( &globalBlockStates[ lookbackBlockIdx ], 0 ) );

            const T currentLookbackBlockScan = ( const T& ) currentLookbackBlockState.scan;
            if( prefix_block_flags::HAS_LOCAL_PREFIX == currentLookbackBlockState.flag )
            {
                lookbackScan += currentLookbackBlockScan;
                --lookbackBlockIdx;
            }
            else if( prefix_block_flags::HAS_FULL_PREFIX == currentLookbackBlockState.flag )
            {
                lookbackScan += currentLookbackBlockScan;
                break;
            }
        }

        sharedPrevScan = lookbackScan;

        const T blockPrefixScan = lookbackScan + sharedLocalBlockScan;
        currentBlockState = { .scan = std::bit_cast< u32 >( blockPrefixScan ), .flag = prefix_block_flags::HAS_FULL_PREFIX };
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], u64( currentBlockState ) );
    }
    __syncthreads();

    if( globalIdx < workElemCount )
    {
        T currentOut = blockScanThreadElem + sharedPrevScan;
        if constexpr( SCAN_TYPE == prefix_scan_t::EXCLUSIVE )
        {
            currentOut -= currentElemToSum;
        }
        scannedOut[ globalIdx ] = currentOut;
    }
}

template<u64 THREADS_PER_BLOCK_X>
thrust::device_vector<i32> DispatchReductionKernel_CUDA( const thrust::device_vector<i32>& inputCuda )
{
    const u64 size = std::size( inputCuda );
    const u64 blocksDispatchedCount = ( size + THREADS_PER_BLOCK_X - 1 ) / THREADS_PER_BLOCK_X;

    thrust::device_vector<i32> outputCuda;
    outputCuda.resize( blocksDispatchedCount ); // doesn't really matter bc we access the raw mem

    KernelReduceBlocksWithWARP<THREADS_PER_BLOCK_X><<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
        thrust::raw_pointer_cast( std::data( inputCuda ) ), size,
        thrust::raw_pointer_cast( std::data( outputCuda ) )
    );

    // Check for any errors launching the kernel
    CUDA_CHECK( cudaGetLastError() );

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    CUDA_CHECK( cudaDeviceSynchronize() );

    return outputCuda;
};

template<prefix_scan_t SCAN_TYPE, u64 THREADS_PER_BLOCK_X>
thrust::device_vector<i32> DispatchChainPrefixScanKernel_CUDA( const thrust::device_vector<i32>& blockReductions )
{
    const u64 size = std::size( blockReductions );
    const u64 blocksDispatchedCount = ( size + THREADS_PER_BLOCK_X - 1 ) / THREADS_PER_BLOCK_X;

    thrust::device_vector<i32> outputCuda;
    outputCuda.resize( size ); // doesn't really matter bc we access the raw mem

    thrust::device_ptr<u64> globalGroupCounter = thrust::device_new<u64>();
    *globalGroupCounter = 0;

    thrust::device_vector<u64> globalBlockStates;
    globalBlockStates.resize( blocksDispatchedCount );

    KernelChainPrefixScanWithDecoupledLookback<SCAN_TYPE, THREADS_PER_BLOCK_X>
        <<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
        thrust::raw_pointer_cast( std::data( blockReductions ) ), size,
        thrust::raw_pointer_cast( globalGroupCounter ),
        thrust::raw_pointer_cast( std::data( globalBlockStates ) ),
        thrust::raw_pointer_cast( std::data( outputCuda ) )
    );

    // Check for any errors launching the kernel
    CUDA_CHECK( cudaGetLastError() );

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    CUDA_CHECK( cudaDeviceSynchronize() );

    thrust::device_delete( globalGroupCounter );

    return outputCuda;
};


struct cuda_context
{
    inline cuda_context()
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        CUDA_CHECK( cudaSetDevice( 0 ) );
    }

    inline ~cuda_context()
    {
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        CUDA_CHECK( cudaDeviceReset() );
    }
};

struct random_generator
{
    std::mt19937 gen;  

    inline random_generator() : gen{ std::random_device{}() } {}

    template<typename T>
    std::vector<T> GenerateNIntegers( u64 n, T rangeMin, T rangeMax )
    {
        std::uniform_int_distribution<T> dist{ rangeMin, rangeMax };

        std::vector<T> out;
        out.resize( n );

        std::generate( std::begin( out ), std::end( out ), [&] () { return dist( gen ); } );

        return out;
    }
};

template<u64 THREADS_PER_BLOCK_X, typename T>
inline std::vector<T> Reduction_CPU( const std::vector<T>& inputData )
{
    std::vector<T> blockReductionsCPU;
    for( u64 i = 0; i < std::size( inputData );  )
    {
        const u64 step = std::min( THREADS_PER_BLOCK_X, std::size( inputData ) - i );

        const auto cbegin = std::cbegin( inputData ) + i;
        const auto cend = cbegin + step;
        const T reduced = std::reduce( std::execution::par, cbegin, cend );

        blockReductionsCPU.push_back( reduced );

        i += step;
    }

    return blockReductionsCPU;
}

constexpr bool CHECK_CORRECTNESS = true;
constexpr u64 THREADS_PER_BLOCK = 512;

int main()
{
    constexpr u64 elemCount = 1'000'000;

    random_generator randGen;
    std::vector<i32> inputData = randGen.GenerateNIntegers<i32>( 
        elemCount, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() );

    std::vector<i32> resultCpu( std::size( inputData ) );
    std::exclusive_scan( std::execution::par, std::cbegin( inputData ), std::cend( inputData ), std::begin( resultCpu ), 0 );

    cuda_context cudaCtx;
    
    auto resultCudaThrust = DispatchChainPrefixScanKernel_CUDA<prefix_scan_t::EXCLUSIVE, THREADS_PER_BLOCK>( inputData );

    std::vector<i32> resultCuda = { std::cbegin( resultCudaThrust ), std::cend( resultCudaThrust ) };

    if constexpr( CHECK_CORRECTNESS )
    {
        auto CmpOp = [] ( const auto& a, const auto& b ) { return a == b; };
        std::span<const i32> cudaSpan = { thrust::raw_pointer_cast( std::data( resultCuda ) ), std::size( resultCuda ) };
        assert( ElementWiseRangeStrictCompare( resultCpu, cudaSpan, CmpOp ) );
    }
    
    std::cout << "DONE!\n";
    return 0;
}
