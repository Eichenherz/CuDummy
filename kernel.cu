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

template<typename T, typename CmpOp>
inline bool ElementWiseRangeStrictCompare( const std::vector<T>& a, const std::vector<T>& b, CmpOp cmpOp )
{
    if( std::size( a ) != std::size( b ) )
    {
        return false;
    }

    //for( uint64_t i = 0; i < std::size( a ); ++i )
    //{
    //    assert( cmpOp( a[ i ], b[ i ] ) );
    //}
    //
    return std::transform_reduce( 
        std::execution::par, std::cbegin( a ), std::cend( a ), std::cbegin( b ), true, std::logical_and<>(), cmpOp );
}

// NOTE: always POT
constexpr uint64_t WARP_SIZE = 32u;
static_assert( std::has_single_bit( WARP_SIZE ), "WARP_SIZE not POT" );
constexpr uint64_t WAPR_SZ_SHIFT = std::bit_width( WARP_SIZE ) - 1u;

__device__ inline uint64_t LaneId() { return threadIdx.x & ( WARP_SIZE - 1 ); }
__device__ inline uint64_t WarpId() { return threadIdx.x >> WAPR_SZ_SHIFT; }

template<typename T>
__device__ T WarpReduceShflDownSync( const T in )
{
    T sum = in;
#pragma unroll
    for( uint64_t offsetWithinWarp = WARP_SIZE >> 1; offsetWithinWarp > 0; offsetWithinWarp >>= 1 ) 
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
    for( uint64_t offset = 1; offset < WARP_SIZE; offset <<= 1 )
    {
        const T warpLaneValAtOffset = __shfl_up_sync( 0xffffffff, inclusvieScan, offset );
        if( LaneId() >= offset ) // NOTE: Distributes values
        {
            inclusvieScan += warpLaneValAtOffset;
        }
    }
    return inclusvieScan;
};

constexpr uint64_t MAX_WARPS_PER_BLOCK = 32;

template<uint64_t THREADS_PER_BLOCK_X, typename T>
__global__ void KernelReduceBlocksWithWARP( const T* input, uint64_t workElemCount, T* partialSums )
{
    constexpr uint64_t WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;
    static_assert( WARPS_PER_BLOCK <= MAX_WARPS_PER_BLOCK, "ERR: Block has more warps !" );

    __shared__ T sharedPartialWarpReductions[ WARP_SIZE ]; 
    // NOTE: only need to zero init if the sharedPartialWarpReductions array is larger than the actual Wraps per Block
    if( threadIdx.x < WARP_SIZE ) 
    {
        sharedPartialWarpReductions[ threadIdx.x ] = T{};
    }
    __syncthreads();

    const uint64_t globalDataBlockOffset = blockDim.x * blockIdx.x; 
    const uint64_t globalIdx = globalDataBlockOffset + threadIdx.x;

    T currentElemToSum = T{};
    if( globalIdx < workElemCount )
    {
        currentElemToSum = input[ globalIdx ];
    }
    
    const T currentWarpSum = WarpReduceShflDownSync( currentElemToSum );
    const uint64_t laneID = LaneId();
    if( laneID == 0 )
    {
        const uint64_t warpID = WarpId();
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
            constexpr uint32_t ACTIVE_LANES_MASK = ( 1u << WARPS_PER_BLOCK ) - 1;
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

enum class prefix_scan_t : uint8_t
{
    INCLUSIVE,
    EXCLUSIVE
};
template<prefix_scan_t SCAN_TYPE, uint64_t THREADS_PER_BLOCK_X, typename T>
__global__ void KernelChainPrefixScan( 
    const T*      input, 
    uint64_t      workElemCount, 
    uint64_t*     globalGroupCounter,
    uint32_t*     globalSemaphoreFlags,
    T*            globalBlockPartialScans,
    T*            scannedOut 
) {
    constexpr uint64_t WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;
    static_assert( WARPS_PER_BLOCK <= MAX_WARPS_PER_BLOCK, "ERR: Block has more warps !" );

    // NOTE: need to get the dynamic idx of the block
    __shared__ uint64_t sharedCurrerntBlockIdx;
    if( threadIdx.x == 0 )
    {
        sharedCurrerntBlockIdx = atomicAdd( globalGroupCounter, 1 );
    }   
    __syncthreads();

    // NOTE: need to use the dynamicBlockIdx to get the corresponding work
    const uint64_t globalDataBlockOffset = blockDim.x * sharedCurrerntBlockIdx;
    const uint64_t globalIdx = globalDataBlockOffset + threadIdx.x;

    T currentElemToSum = T{};
    if( globalIdx < workElemCount )
    {
        currentElemToSum = input[ globalIdx ];
    }

    const T blockScan = ThreadBlockInclusiveScanWithSync( currentElemToSum );

    __shared__ T localScan;
    if( threadIdx.x == ( blockDim.x - 1 ) )
    {
        localScan = blockScan;
    }
    __syncthreads();

    __shared__ T sharedPrevSum;
    // NOTE: here we force order the blocks
    if( threadIdx.x == 0 )
    {
        // NOTE: we could add a fence here if we care about immediate visiblity
        while( atomicAdd( &globalSemaphoreFlags[ sharedCurrerntBlockIdx ], 0 ) == 0 );

        sharedPrevSum = globalBlockPartialScans[ sharedCurrerntBlockIdx ];
        
        globalBlockPartialScans[ sharedCurrerntBlockIdx + 1 ] = sharedPrevSum + localScan;

        __threadfence();
        atomicAdd( &globalSemaphoreFlags[ sharedCurrerntBlockIdx + 1 ], 1 );
    };
    __syncthreads();

    if( globalIdx < workElemCount )
    {
        T currentOut = blockScan + sharedPrevSum;
        if constexpr( SCAN_TYPE == prefix_scan_t::EXCLUSIVE )
        {
            currentOut -= currentElemToSum;
        }
        scannedOut[ globalIdx ] = currentOut;
    }
}

template<uint64_t THREADS_PER_BLOCK_X>
thrust::device_vector<int32_t> DispatchReductionKernel_CUDA( const thrust::device_vector<int32_t>& inputCuda )
{
    const uint64_t size = std::size( inputCuda );
    const uint64_t blocksDispatchedCount = ( size + THREADS_PER_BLOCK_X - 1 ) / THREADS_PER_BLOCK_X;

    thrust::device_vector<int32_t> outputCuda;
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

template<prefix_scan_t SCAN_TYPE, uint64_t THREADS_PER_BLOCK_X>
thrust::device_vector<int32_t> DispatchChainPrefixScanKernel_CUDA( const thrust::device_vector<int32_t>& blockReductions )
{
    const uint64_t size = std::size( blockReductions );
    const uint64_t blocksDispatchedCount = ( size + THREADS_PER_BLOCK_X - 1 ) / THREADS_PER_BLOCK_X;

    thrust::device_vector<int32_t> outputCuda;
    outputCuda.resize( size ); // doesn't really matter bc we access the raw mem

    thrust::device_ptr<uint64_t> globalGroupCounter = thrust::device_new<uint64_t>();
    *globalGroupCounter = 0;

    thrust::device_vector<uint32_t> globalSemaphoreFlags;
    globalSemaphoreFlags.resize( blocksDispatchedCount, 0 );
    globalSemaphoreFlags[ 0 ] = uint32_t( -1 ); // NOTE: first block/group is ready to go

    thrust::device_vector<int32_t> globalBlockPartialScans;
    globalBlockPartialScans.resize( blocksDispatchedCount, 0 );

    KernelChainPrefixScan<SCAN_TYPE, THREADS_PER_BLOCK_X><<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
        thrust::raw_pointer_cast( std::data( blockReductions ) ), size,
        thrust::raw_pointer_cast( globalGroupCounter ),
        thrust::raw_pointer_cast( std::data( globalSemaphoreFlags ) ),
        thrust::raw_pointer_cast( std::data( globalBlockPartialScans ) ),
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
    std::vector<T> GenerateNIntegers( uint64_t n, T rangeMin, T rangeMax )
    {
        std::uniform_int_distribution<T> dist{ rangeMin, rangeMax };

        std::vector<T> out;
        out.resize( n );

        std::generate( std::begin( out ), std::end( out ), [&] () { return dist( gen ); } );

        return out;
    }
};

template<uint64_t THREADS_PER_BLOCK_X, typename T>
inline std::vector<T> Reduction_CPU( const std::vector<T>& inputData )
{
    std::vector<T> blockReductionsCPU;
    for( uint64_t i = 0; i < std::size( inputData );  )
    {
        const uint64_t step = std::min( THREADS_PER_BLOCK_X, std::size( inputData ) - i );

        const auto cbegin = std::cbegin( inputData ) + i;
        const auto cend = cbegin + step;
        const T reduced = std::reduce( std::execution::par, cbegin, cend );

        blockReductionsCPU.push_back( reduced );

        i += step;
    }

    return blockReductionsCPU;
}

constexpr bool CHECK_CORRECTNESS = true;
constexpr uint64_t THREADS_PER_BLOCK = 512;

int main()
{
    constexpr uint64_t elemCount = 1'000'000;

    random_generator randGen;
    std::vector<int32_t> inputData = randGen.GenerateNIntegers<int32_t>( 
        elemCount, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() );

    std::vector<int32_t> resultCpu( std::size( inputData ) );
    std::exclusive_scan( std::execution::par, std::cbegin( inputData ), std::cend( inputData ), std::begin( resultCpu ), 0 );

    cuda_context cudaCtx;
    
    auto resultCudaThrust = DispatchChainPrefixScanKernel_CUDA<prefix_scan_t::EXCLUSIVE, THREADS_PER_BLOCK>( inputData );

    std::vector<int32_t> resultCuda = { std::cbegin( resultCudaThrust ), std::cend( resultCudaThrust ) };

    if constexpr( CHECK_CORRECTNESS )
    {
        auto CmpOp = [] ( const auto& a, const auto& b ) { return a == b; };
        assert( ElementWiseRangeStrictCompare( resultCpu, resultCuda, CmpOp ) );
    }
    
    std::cout << "DONE!\n";
    return 0;
}
