#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <bit>
#include <stdio.h>

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

#include "core_types.h"
#include "warps.h"

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


struct cuda_timer_section
{
    cudaEvent_t& start;
    cudaEvent_t& stop;

    inline cuda_timer_section( cudaEvent_t& _start, cudaEvent_t& _stop ) : start{ _start }, stop{ _stop }
    {
        CUDA_CHECK( cudaEventRecord( start ) );
    }

    inline ~cuda_timer_section()
    {
        CUDA_CHECK( cudaEventRecord( stop ) );
    }
};

struct cuda_timer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    inline cuda_timer()
    {
        CUDA_CHECK( cudaEventCreate( &start ) );
        CUDA_CHECK( cudaEventCreate( &stop ) );
    }
    inline ~cuda_timer()
    {
        CUDA_CHECK( cudaEventDestroy( start ) );
        CUDA_CHECK( cudaEventDestroy( stop ) );
    }
    inline float GetElapsedTime() const
    {
        float ms = 0.0f;
        CUDA_CHECK( cudaEventElapsedTime( &ms, start, stop ) );
        return ms;
    }

    inline cuda_timer_section GetTimedSection()
    {
        return { start, stop };
    }
};



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

// TODO: shrink shared mem ?
// NOTE: these require FULL WARPS
template<typename T>
__device__ T ThreadBlockInclusiveScanSyncWithMem( const T currentThreadValue )
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
    __shared__ u32 sharedCurrerntBlockIdx;
    if( threadIdx.x == 0 )
    {
        sharedCurrerntBlockIdx = atomicAdd( globalGroupCounter, 1 );
        // NOTE: zero init
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], 0u );
    }
    __syncthreads();

    // NOTE: need to use the dynamicBlockIdx to get the corresponding work
    u64 globalDataBlockOffset = blockDim.x * sharedCurrerntBlockIdx;
    u64 globalIdx = globalDataBlockOffset + threadIdx.x;

    T currentElemToSum = T{};
    if( globalIdx < workElemCount )
    {
        currentElemToSum = input[ globalIdx ];
    }

    T blockScanThreadElem = ThreadBlockInclusiveScanSyncWithMem( currentElemToSum );
    //__syncthreads();
    __shared__ T sharedLocalBlockScan;
    if( threadIdx.x == ( blockDim.x - 1 ) )
    {
        sharedLocalBlockScan = blockScanThreadElem;
    }
    __syncthreads();

    // NOTE: local scan ready
    if( threadIdx.x == 0 )
    {
        prefix_block_state currentBlockState = {
            .scan = std::bit_cast<u32>( sharedLocalBlockScan ), .flag = prefix_block_flags::HAS_LOCAL_PREFIX };
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], u64( currentBlockState ) );
    }

    __shared__ T sharedLookbackScan;
    if( threadIdx.x == 0 )
    {
        sharedLookbackScan = T{};
    }

    __syncthreads();

    // NOTE: we'll keep looking backwards until we find a PREFIX STATE or all threads are outside the lookback bounds
    if( ( WarpId() == 0 ) && ( threadIdx.x < WARP_SIZE ) )
    {
        i64 lookbackOffset;
        if( LaneId() == 0 )
        {
            lookbackOffset = i64( sharedCurrerntBlockIdx ) - WARP_SIZE;
        }

        for( ;; )
        {
            i64 warpLookbackOffset = __shfl_sync( u32( -1 ), lookbackOffset, 0 );
            i64 currentLookbackIdx = warpLookbackOffset + threadIdx.x;
            bool isThreadInRange = currentLookbackIdx >= 0;

            u32 activeLanesMask = __ballot_sync( u32( -1 ), isThreadInRange );
            if( 0 == activeLanesMask )
            {
                break;
            }

            prefix_block_state currentLookbackBlockState = {};
            if( isThreadInRange )
            {
                currentLookbackBlockState = PrefixBlockStateFromU64( atomicAdd( &globalBlockStates[ currentLookbackIdx ], 0 ) );
            }

            u32 flag = currentLookbackBlockState.flag;

            u32 unavailableMask = __ballot_sync( activeLanesMask, prefix_block_flags::UNAVAILABLE == flag );
            if( unavailableMask ) continue;

            u32 hasLocalScanMask = __ballot_sync( activeLanesMask, prefix_block_flags::HAS_LOCAL_PREFIX == flag );
            u32 hasFullScanMask = __ballot_sync( activeLanesMask, prefix_block_flags::HAS_FULL_PREFIX == flag );
            // NOTE: at this stage we can't have any invalid scans, they either all LOCAL or there's at least one FULL

            // NOTE: the bools and mask computation are warp-uniform
            bool warpHasOnlyValidStates = ( hasLocalScanMask | hasFullScanMask ) == activeLanesMask;
            bool warpHasAtLeastOneFullScan = warpHasOnlyValidStates && ( hasFullScanMask != 0 );
            
            u32 warpScanMask = hasLocalScanMask;
            if( warpHasAtLeastOneFullScan )
            {
                u32 fullScanHighestIdx = ( WARP_SIZE - 1 ) - std::countl_zero( hasFullScanMask );
                u32 fullScanMask = ~( ( 1u << fullScanHighestIdx ) - 1 ) & activeLanesMask;
                warpScanMask = fullScanMask;
            }
            
            T currentLookbackScan = ( const T& ) currentLookbackBlockState.scan;
            // NOTE: Pascal doesn't have __reduce instructions and 
            // the easiest way to compute the scan is use a full warp and zero the masked elements
            currentLookbackScan = ( ( warpScanMask >> LaneId() ) & 1 ) ? currentLookbackScan : T{};
            T currentWarpLookbackScan = WarpReduceShflDownSync( currentLookbackScan );

            if( LaneId() == 0 )
            {
                sharedLookbackScan += currentWarpLookbackScan;
                lookbackOffset -= WARP_SIZE;
            }

            if( warpHasAtLeastOneFullScan ) break;
        }
    }

    __syncthreads();

    if( threadIdx.x == 0 )
    {
        T blockPrefixScan = sharedLookbackScan + sharedLocalBlockScan;
        prefix_block_state currentBlockState = { 
            .scan = std::bit_cast<u32>( blockPrefixScan ), .flag = prefix_block_flags::HAS_FULL_PREFIX };
        atomicExch( &globalBlockStates[ sharedCurrerntBlockIdx ], u64( currentBlockState ) );
    }

    if( globalIdx < workElemCount )
    {
        T currentOut = blockScanThreadElem + sharedLookbackScan;
        if constexpr( SCAN_TYPE == prefix_scan_t::EXCLUSIVE )
        {
            currentOut -= currentElemToSum;
        }
        scannedOut[ globalIdx ] = currentOut;
    }
}

// NOTE: taken from https://github.com/GPUOpen-LibrariesAndSDKs/Orochi/blob/main/ParallelPrimitives/RadixSortKernels.h
constexpr u64 RADIX_DIGIT_BIT_COUNT = 8;
constexpr u64 RADIX_BIN_COUNT = 1 << RADIX_DIGIT_BIT_COUNT;
constexpr u64 RADIX_DIGIT_MASK = ( 1u << RADIX_DIGIT_BIT_COUNT ) - 1;

constexpr u64 RADIX_SORT_BLOCK_SIZE = 4096;

constexpr u64 RADIX_HISTOGRAM_ITEM_PER_BLOCK = 2048;
constexpr u64 RADIX_HISTOGRAM_THREADS_PER_BLOCK = 256;
constexpr u64 RADIX_HISTOGRAM_ITEMS_PER_THREAD = RADIX_HISTOGRAM_ITEM_PER_BLOCK / RADIX_HISTOGRAM_THREADS_PER_BLOCK;


using radix_sort_key_t = u32;

constexpr u32 DESCENDING_MASK_32 = u32( -1 );
constexpr u32 ASCENDING_MASK_32 = 0;

template<u32 ORDER_MASK>
__device__ inline u32 GetKeyBits( u32 x ) { return x ^ ORDER_MASK; } // NOTE: clever trick to reverse bits

template <typename T>
__device__ inline T ScanExclusive( T prefix, T* sMemIO, int nElement )
{
    // assert(nElement <= nThreads)
    bool active = threadIdx.x < nElement;
    T value = active ? sMemIO[ threadIdx.x ] : 0;
    T x = value;

    for( u32 offset = 1; offset < nElement; offset <<= 1 )
    {
        if( active && offset <= threadIdx.x )
        {
            x += sMemIO[ threadIdx.x - offset ];
        }

        __syncthreads();

        if( active )
        {
            sMemIO[ threadIdx.x ] = x;
        }

        __syncthreads();
    }

    T sum = sMemIO[ nElement - 1 ];

    __syncthreads();

    if( active )
    {
        sMemIO[ threadIdx.x ] = x + prefix - value;
    }

    __syncthreads();

    return sum;
}

template<u64 THREADS_PER_BLOCK_X, u64 ITEMS_PER_THREAD, u32 ORDER_MASK>
__global__ void KernelOrochiRadixHistogram( 
    const radix_sort_key_t* input, 
    u64                     workElemCount,
    u32*                    gScansBuffer, 
    u32*                    gBlockCounter 
) {
    constexpr u64 ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK_X;

    __shared__ u32 sharedPerDigitHistograms[ sizeof( radix_sort_key_t ) ][ RADIX_BIN_COUNT ];

    for( u64 i = 0; i < sizeof( radix_sort_key_t ); ++i )
    {
        for( u64 j = threadIdx.x; j < RADIX_BIN_COUNT; j += THREADS_PER_BLOCK_X )
        {
            sharedPerDigitHistograms[ i ][ j ] = 0;
        }
    }

    const u32 numBlocks = ( workElemCount + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    __shared__ u64 blockIdx;
    for( ;; )
    {
        if( threadIdx.x == 0 )
        {
            blockIdx = atomicAdd( gBlockCounter, 1 );
        }
        __syncthreads();

        if( numBlocks <= blockIdx ) break;

        const u64 globalOffset = blockIdx * ITEMS_PER_BLOCK + threadIdx.x * ITEMS_PER_THREAD;
        for( u64 itemIdx = 0; itemIdx < ITEMS_PER_THREAD; ++itemIdx )
        {
            const u64 globalIdx = globalOffset + itemIdx;
            if( globalIdx < workElemCount )
            {
                const radix_sort_key_t item = input[ globalIdx ];
                for( u64 radixDigitIdx = 0; radixDigitIdx < sizeof( radix_sort_key_t ); ++radixDigitIdx )
                {
                    u32 bitIdx = radixDigitIdx * RADIX_DIGIT_BIT_COUNT;
                    u32 maskedItemBits = GetKeyBits<ORDER_MASK>( item );
                    u32 bitBin = ( maskedItemBits >> bitIdx ) & RADIX_DIGIT_MASK;
                    atomicAdd( &sharedPerDigitHistograms[ radixDigitIdx ][ bitBin ], 1 );
                }
            }
        }
        __syncthreads();
    }

    for( u64 i = 0; i < sizeof( radix_sort_key_t ); ++i )
    {
        ScanExclusive<u32>( 0, &sharedPerDigitHistograms[ i ][ 0 ], RADIX_BIN_COUNT );
    }

    for( u64 i = 0; i < sizeof( radix_sort_key_t ); ++i )
    {
        for( u64 j = threadIdx.x; j < RADIX_BIN_COUNT; j += THREADS_PER_BLOCK_X )
        {
            atomicAdd( &gScansBuffer[ RADIX_BIN_COUNT * i + j ], sharedPerDigitHistograms[ i ][ j ] );
        }
    }
}

template<u64 RADIX_DIGIT_BIT_WIDTH>
struct radix_config_t
{
    using radix_key_t = u32;

    static constexpr u32 DIGIT_BIT_WIDTH = RADIX_DIGIT_BIT_WIDTH;
    static constexpr u32 DIGIT_BINS = 1u << RADIX_DIGIT_BIT_WIDTH;
    static constexpr u32 DIGITS = sizeof( radix_key_t ) * 8 / RADIX_DIGIT_BIT_WIDTH;
    static constexpr u32 DIGIT_MASK = ( 1u << RADIX_DIGIT_BIT_WIDTH ) - 1;

    __device__ static u32 ExtractDigit( u32 x, u32 digitIdx ) { return ( x >> ( digitIdx * DIGIT_BIT_WIDTH ) ) & DIGIT_MASK; }
};

template<u64 THREADS_PER_BLOCK_X, u64 ITEMS_PER_THREAD, u32 ORDER_MASK>
__global__ void KernelOrochiRadixHistogramWarps( 
    const radix_sort_key_t* input, 
    u64                     workElemCount,
    u32*                    gScansBuffer, 
    u32*                    gBlockCounter 
) {
    constexpr u64 ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK_X;
    constexpr u64 KERNEL_WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;

    using radix_config = radix_config_t<RADIX_DIGIT_BIT_COUNT>;

    __shared__ u32 sharedPerDigitHistograms[ radix_config::DIGITS ][ radix_config::DIGIT_BINS ];

    for( u64 i = 0; i < radix_config::DIGITS; ++i )
    {
        for( u64 j = threadIdx.x; j < radix_config::DIGIT_BINS; j += THREADS_PER_BLOCK_X )
        {
            sharedPerDigitHistograms[ i ][ j ] = 0;
        }
    }

    u32 numBlocks = ( workElemCount + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    __shared__ u64 blockIdx;
    for( ;; )
    {
        if( threadIdx.x == 0 )
        {
            blockIdx = atomicAdd( gBlockCounter, 1 );
        }
        __syncthreads();

        if( numBlocks <= blockIdx ) break;

        u64 globalOffset = blockIdx * ITEMS_PER_BLOCK + threadIdx.x * ITEMS_PER_THREAD;
        for( u64 itemIdx = 0; itemIdx < ITEMS_PER_THREAD; ++itemIdx )
        {
            u64 globalIdx = globalOffset + itemIdx;
            if( globalIdx < workElemCount )
            {
                radix_sort_key_t item = input[ globalIdx ];
                for( u64 radixDigitIdx = 0; radixDigitIdx < radix_config::DIGITS; ++radixDigitIdx )
                {
                    u32 extractedBin = radix_config::ExtractDigit( GetKeyBits<ORDER_MASK>( item ), radixDigitIdx );
                    atomicAdd( &sharedPerDigitHistograms[ radixDigitIdx ][ extractedBin ], 1 );
                }
            }
        }
        __syncthreads();
    }

    for( u64 i = 0; i < radix_config::DIGITS; ++i )
    {
        bool threadInRange = threadIdx.x < radix_config::DIGIT_BINS;
        u32 currentThreadVal = threadInRange ? sharedPerDigitHistograms[ i ][ threadIdx.x ] : 0;

        static_assert( THREADS_PER_BLOCK_X <= radix_config::DIGIT_BINS ); // NOTE: otherwise we miss elems when we sum
        u32 threadScanElem = ThreadBlockInclusiveScanSyncWithMem( currentThreadVal );
        if( threadInRange )
        {
            sharedPerDigitHistograms[ i ][ threadIdx.x ] = threadScanElem - currentThreadVal;
        }
        __syncthreads();
    }

    for( u64 i = 0; i < radix_config::DIGITS; ++i )
    {
        for( u64 j = threadIdx.x; j < radix_config::DIGIT_BINS; j += THREADS_PER_BLOCK_X )
        {
            atomicAdd( &gScansBuffer[ radix_config::DIGIT_BINS * i + j ], sharedPerDigitHistograms[ i ][ j ] );
        }
    }
}

// TODO: we can shrink our data storage as much as ( WORST CASE ): #items per block will fit in one bin
// NOTE: we expect this to have many digits and few bins
template<u64 THREADS_PER_BLOCK_X, u64 ITEMS_PER_THREAD, u32 ORDER_MASK>
__global__ void KernelRadixHistogram( const radix_sort_key_t* input, u64 workElemCount, u32* gHistogram, u32* gBlockCounter )
{
    // NOTE: need this bc we'll not have a 1:1 thread:workItem mapping
    constexpr u64 ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK_X;
    constexpr u64 KERNEL_WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;

    using radix_config = radix_config_t<2>;
    constexpr u64 DATA_SIZE = radix_config::DIGITS * radix_config::DIGIT_BINS;

    // NOTE: transpose so we can coalesce for reduction
    __shared__ u16 ldsWarpPerDigitHistograms[ DATA_SIZE ][ KERNEL_WARPS_PER_BLOCK ];
    for( u64 i = LaneId(); i < DATA_SIZE; i += WARP_SIZE )
    {
        ldsWarpPerDigitHistograms[ i ][ WarpId() ] = 0;
    }

    const u32 numBlocks = ( workElemCount + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    __shared__ u64 currentBlockIdx;
    for( ;; )
    {
        if( threadIdx.x == 0 )
        {
            currentBlockIdx = atomicAdd( gBlockCounter, 1 );
        }
        __syncthreads();

        if( numBlocks <= currentBlockIdx ) break;

        const u64 globalOffset = currentBlockIdx * ITEMS_PER_BLOCK + threadIdx.x * ITEMS_PER_THREAD;
        for( u64 itemIdx = 0; itemIdx < ITEMS_PER_THREAD; ++itemIdx )
        {
            const u64 globalIdx = globalOffset + itemIdx;
            const bool laneActive = globalIdx < workElemCount;
            const u32 activeLanesMask = __ballot_sync( u32( -1 ), laneActive );

            const radix_sort_key_t item = ( laneActive ) ? input[ globalIdx ] : 0;

            for( u32 digitIdx = 0; digitIdx < radix_config::DIGITS; ++digitIdx )
            {
                const u32 digit = radix_config::ExtractDigit( GetKeyBits<ORDER_MASK>( item ), digitIdx );

            #pragma unroll
                for( u32 binIdx = 0; binIdx < radix_config::DIGIT_BINS; ++binIdx )
                {
                    // NOTE: this mask is very important, otherwise we end up summing 0th bin incorrectly
                    u32 currentBinBallot = __ballot_sync( activeLanesMask, binIdx == digit );
                    u16 warpCountForBin = std::popcount( currentBinBallot );
                    if( ( LaneId() == 0 ) && warpCountForBin )
                    {
                        u32 binHistoIdx = digitIdx * radix_config::DIGIT_BINS + binIdx;
                        u64 warpId = WarpId();
                        ldsWarpPerDigitHistograms[ binHistoIdx ][ warpId ] += warpCountForBin;
                    }
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
    // NOTE: exclusive scan the warp-local histos to warp 0th slot
    for( u32 digitIdx = WarpId(); digitIdx < radix_config::DIGITS; digitIdx += KERNEL_WARPS_PER_BLOCK )
    {
        u16 prevScan = 0;
        
    #pragma unroll
        for( u64 binIdx = 0; binIdx < radix_config::DIGIT_BINS; ++binIdx )
        {
            u32 binHistoIdx = digitIdx * radix_config::DIGIT_BINS + binIdx;
    
            // NOTE: need a full WARP for the reduction so we 0 the "invalid" threads
            u16 binItem = ( LaneId() < KERNEL_WARPS_PER_BLOCK ) ? ldsWarpPerDigitHistograms[ binHistoIdx ][ LaneId() ] : 0;
    
            u16 blockBinItem = WarpReduceShflDownSync( binItem );
            if( LaneId() == 0 )
            {
                ldsWarpPerDigitHistograms[ binHistoIdx ][ 0 ] = prevScan;
                prevScan += blockBinItem;
            }
        }
    }
    __syncthreads();

#pragma unroll 4
    for( u64 i = threadIdx.x; i < DATA_SIZE; i += THREADS_PER_BLOCK_X )
    {
        atomicAdd( &gHistogram[ i ], ldsWarpPerDigitHistograms[ i ][ 0 ] );
    }
}

constexpr auto N_RADIX{ 8 };
constexpr auto BIN_SIZE{ 1 << N_RADIX };
constexpr int REORDER_NUMBER_OF_WARPS = 8;

template<u32 THREADS_PER_BLOCK_X, u32 ELEM_COUNT, typename T>
__device__ __forceinline__ void ClearShared( T* lds, T value )
{
    for( u32 i = threadIdx.x; i < ELEM_COUNT; i += THREADS_PER_BLOCK_X )
    {
        lds[ i ] = value;
    }
}

struct onesweep_lds
{
    struct Phase1
    {
        u16 blockHistogram[ BIN_SIZE ];
        u16 lpSum[ BIN_SIZE * REORDER_NUMBER_OF_WARPS ];
    };
    struct Phase2
    {
        radix_sort_key_t elements[ RADIX_SORT_BLOCK_SIZE ];
    };

    union
    {
        Phase1 phase1;
        Phase2 phase2;
    };
};

struct partition_id
{
    u64 value : 32;
    u64 block : 30;
    u64 flag : 2;

    __host__ __device__ inline explicit operator u64() const
    {
        return std::bit_cast<u64>( *this );
    }
};

__host__ __device__ inline partition_id PartitionIDFromU64( u64 addr )
{
    return std::bit_cast<partition_id>( addr );
}

template<u32 SIZE = 1024>
struct circular_buffer_lookback
{
    static constexpr u32 LOOKBACK_TABLE_SIZE = SIZE;
    static constexpr u32 MAX_LOOK_BACK = 64;
    // NOTE: use tail bits to minimize contention
    static constexpr u32 TAIL_BITS = 5;
    static constexpr u32 TAIL_MASK = 0xFFFFFFFFu << TAIL_BITS;
    static_assert( MAX_LOOK_BACK < LOOKBACK_TABLE_SIZE, "SIZE must be greater than the MAX LOOKBACK" );

    alignas( 64 ) u32 tailIterator;
    u64 data[ LOOKBACK_TABLE_SIZE ];

    // NOTE: waits until the blockIdx has enough space
    __device__ __forceinline__ void SpinWaitForSpace( u32 blockIndex )
    {
        if( LOOKBACK_TABLE_SIZE <= blockIndex )
        {
            // NOTE: wait until blockIdx.x < tail - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE
            while( ( atomicAdd( &tailIterator, 0 ) & TAIL_MASK ) - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE <= blockIndex );
        }
    }

    __device__ __forceinline__ void IncrementTailIter( u32 blockIndex, u32 blockCount )
    {
        // NOTE: The lower bits of the tail iterator are incremented out of order to reduce spin waiting.
        while( ( atomicAdd( &tailIterator, 0 ) & TAIL_MASK ) != ( blockIndex & TAIL_MASK ) );
        atomicAdd( &tailIterator, blockCount - 1 /* after the very last item, it will be zero */ );
    }

    __device__ __forceinline__ auto& operator[]( u32 idx ) 
    {
        return data[ idx ];
    }
};



// NOTE: taken from orochi as well ( link above )
template<u64 THREADS_PER_BLOCK_X, u64 ITEMS_PER_THREAD, u32 ORDER_MASK>
__global__ void KernelOrochiOnesweepReorder( 
    const radix_sort_key_t* inputKeys, 
    radix_sort_key_t* outputKeys, 
    u32 workElemCount, 
    u32* gpSumBuffer,
    volatile circular_buffer_lookback<1024>& lookBackBuffer,  
    u32 iteration 
) {
    constexpr u64 KERNEL_WARPS_PER_BLOCK = THREADS_PER_BLOCK_X / WARP_SIZE;

    using radix_config = radix_config_t<RADIX_DIGIT_BIT_COUNT>;

    __shared__ u32 pSum[ radix_config::DIGIT_BINS ];
    __shared__ onesweep_lds lds;

    u32 digitIdx = iteration;
    u32 blockGlobalOffset = blockIdx.x * RADIX_SORT_BLOCK_SIZE;
    u32 totalBlockCount = ( workElemCount + RADIX_SORT_BLOCK_SIZE - 1 ) / RADIX_SORT_BLOCK_SIZE;

    ClearShared<THREADS_PER_BLOCK_X, radix_config::DIGIT_BINS * KERNEL_WARPS_PER_BLOCK, u16>( lds.phase1.lpSum, 0 );
    __syncthreads();

    radix_sort_key_t keys[ ITEMS_PER_THREAD ];
    u32 warpLaneOffsets[ ITEMS_PER_THREAD ];

    u32 warp = WarpId();
    u32 lane = LaneId();

    // NOTE: warp level radix binning
    for( u32 i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, ++k )
    {
        u32 itemIndex = blockGlobalOffset + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + lane + i;
        bool threadWithinBounds = itemIndex < workElemCount;
        if( threadWithinBounds )
        {
            keys[ k ] = inputKeys[ itemIndex ];
        }

        u32 activeThreadsMask = __ballot_sync( u32( -1 ), threadWithinBounds );

        u32 binIdx = radix_config::ExtractDigit( GetKeyBits<ORDER_MASK>( keys[ k ] ), digitIdx );
        // NOTE: equiv to an atomicAdd to a shared counter
        for( u32 j = 0; j < radix_config::DIGIT_BIT_WIDTH; ++j )
        {
            u32 bit = ( binIdx >> j ) & 0x1;
            u32 setLanesMask = __ballot_sync( u32( -1 ), bit );
            u32 difference = ( u32( -1 ) * bit ) ^ setLanesMask; // NOTE: conditionally get the diff mask
            activeThreadsMask &= ~difference; // NOTE: remove difference
        }

        u32 lowerLanesMask = ( 1u << lane ) - 1;
        auto digitCount = lds.phase1.lpSum[ binIdx * KERNEL_WARPS_PER_BLOCK + warp ];
        // NOTE: only want digitCount + ( lower lanes for the current offset )
        warpLaneOffsets[ k ] = digitCount + std::popcount( activeThreadsMask & lowerLanesMask ); 

        __syncwarp( u32( -1 ) );

        u32 leaderIdx = __ffs( activeThreadsMask ) - 1;
        if( lane == leaderIdx )
        {
            // NOTE: store the full warp offset
            lds.phase1.lpSum[ binIdx * KERNEL_WARPS_PER_BLOCK + warp ] = digitCount + std::popcount( activeThreadsMask );
        }
        __syncwarp( u32( -1 ) );
    }

    __syncthreads();

    for( u32 binIdx = threadIdx.x; binIdx < radix_config::DIGIT_BINS; binIdx += THREADS_PER_BLOCK_X )
    {
        u32 binSum = 0;
        for( u32 warpIdx = 0; warpIdx < KERNEL_WARPS_PER_BLOCK; ++warpIdx )
        {
            binSum += lds.phase1.lpSum[ binIdx * KERNEL_WARPS_PER_BLOCK + warpIdx ];
        }
        lds.phase1.blockHistogram[ binIdx ] = binSum;
    }
    
    if( threadIdx.x == 0 )
    {
        lookBackBuffer.SpinWaitForSpace( blockIdx.x );
    }
    __syncthreads();

    for( u32 binIdx = threadIdx.x; binIdx < radix_config::DIGIT_BINS; binIdx += THREADS_PER_BLOCK_X )
    {
        u32 binSum = lds.phase1.blockHistogram[ binIdx ];
        u32 pIndex = radix_config::DIGIT_BINS * ( blockIdx.x % lookBackBuffer.LOOKBACK_TABLE_SIZE ) + binIdx;

        {
            partition_id pa = { .value = binSum, .block = blockIdx.x, .flag = 1 };
            lookBackBuffer[ pIndex ] = u64( pa );
        }

        u32 lookBackSum = 0;
        for( u32 currBlockIdx = blockIdx.x - 1; 0 <= bi; --currBlockIdx )
        {
            u32 lookBackIndex = radix_config::DIGIT_BINS * ( currBlockIdx % lookBackBuffer.LOOKBACK_TABLE_SIZE ) + binIdx;
            // when you reach to the maximum, flag must be 2. flagRequire = 0b10
            // Otherwise, flag can be 1 or 2 flagRequire = 0b11
            u32 flagRequire = ( lookBackBuffer.MAX_LOOK_BACK == ( blockIdx.x - currBlockIdx ) ) ? 2 : 3;

            partition_id pa;
            for( ;; )
            {
                pa = PartitionIDFromU64( lookBackBuffer[ lookBackIndex ] );
                bool keepLooping = ( ( pa.flag & flagRequire ) == 0 ) || ( pa.block != currBlockIdx );
                if( !keepLooping ) break;
            }

            lookBackSum += pa.value;
            if( pa.flag == 2 ) break;
        }

        partition_id pa = { .value = lookBackSum + binSum, .block = blockIdx.x, .flag = 2 };
        lookBackBuffer[ pIndex ] = u64( pa );

        u32 gp = gpSumBuffer[ iteration * BIN_SIZE + binIdx ];
        u32 globalOutput = gp + lookBackSum;
        pSum[ binIdx ] = globalOutput;
    }

    __syncthreads();

    
    if( threadIdx.x == 0 )
    {
        lookBackBuffer.IncrementTailIter( blockIdx.x, totalBlockCount );
    }

    __syncthreads();

    u32 prefix = 0;
    for( u32 i = 0; i < radix_config::DIGIT_BINS; i += THREADS_PER_BLOCK_X )
    {
        prefix += ScanExclusive<u16>( prefix, lds.phase1.blockHistogram + i, std::min( THREADS_PER_BLOCK_X, radix_config::DIGIT_BINS ) );
    }

    for( u32 binIdx = threadIdx.x; binIdx < radix_config::DIGIT_BINS; binIdx += THREADS_PER_BLOCK_X )
    {
        u32 s = lds.phase1.blockHistogram[ binIdx ];

        // NOTE: pre-substruct to avoid pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex] to calculate destinations
        pSum[ binIdx ] -= s; 
        for( u32 w = 0; w < KERNEL_WARPS_PER_BLOCK; w++ )
        {
            u32 index = binIdx * KERNEL_WARPS_PER_BLOCK + w;
            u32 n = lds.phase1.lpSum[ index ];
            lds.phase1.lpSum[ index ] = s;
            s += n;
        }
    }

    __syncthreads();

    for( u32 k = 0; k < ITEMS_PER_THREAD; k++ )
    {
        u32 binIdx = radix_config::ExtractDigit( GetKeyBits<ORDER_MASK>( keys[ k ] ), digitIdx );
        warpLaneOffsets[ k ] += lds.phase1.lpSum[ binIdx * REORDER_NUMBER_OF_WARPS + warp ];
    }

    __syncthreads();

    for( u32 i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
    {
        u32 itemIndex = blockGlobalOffset + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
        if( itemIndex < workElemCount )
        {
            lds.phase2.elements[ warpLaneOffsets[ k ] ] = keys[ k ];
        }
    }

    __syncthreads();

    for( u32 i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += THREADS_PER_BLOCK_X )
    {
        u32 itemIndex = blockGlobalOffset + i;
        if( itemIndex < workElemCount )
        {
            auto item = lds.phase2.elements[ i ];
            u32 binIdx = radix_config::ExtractDigit( GetKeyBits<ORDER_MASK>( item ), digitIdx );
            u32 dstIndex = pSum[ binIdx ] + i;
            outputKeys[ dstIndex ] = item;
        }
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

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    thrust::device_delete( globalGroupCounter );

    return outputCuda;
};

template<u64 THREADS_PER_BLOCK_X = RADIX_HISTOGRAM_THREADS_PER_BLOCK>
auto DispatchOrochiHistogramKernel( const thrust::device_vector<radix_sort_key_t>& keys )
{
    constexpr u64 ITEMS_PER_BLOCK = THREADS_PER_BLOCK_X * RADIX_HISTOGRAM_ITEMS_PER_THREAD;

    const u64 size = std::size( keys );
    const u64 blocksDispatchedCount = ( size + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    thrust::device_vector<u32> prefixBinsOut;
    prefixBinsOut.resize( RADIX_BIN_COUNT * sizeof( radix_sort_key_t ), 0 );

    auto globalGroupCounter = thrust::device_new<u32>();
    *globalGroupCounter = 0;

    KernelOrochiRadixHistogram<THREADS_PER_BLOCK_X, RADIX_HISTOGRAM_ITEMS_PER_THREAD, ASCENDING_MASK_32>
        <<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
            thrust::raw_pointer_cast( std::data( keys ) ), size,
            thrust::raw_pointer_cast( std::data( prefixBinsOut ) ),
            thrust::raw_pointer_cast( globalGroupCounter )
    );

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    thrust::device_delete( globalGroupCounter );

    return prefixBinsOut;
}

template<
    u32 ORDER_MASK, 
    u32 THREADS_PER_BLOCK_X = RADIX_HISTOGRAM_THREADS_PER_BLOCK,
    u32 ITEMS_PER_THREAD = RADIX_HISTOGRAM_ITEMS_PER_THREAD>
auto DispatchOrochiHistogramKernelWarps( const thrust::device_vector<radix_sort_key_t>& keys )
{
    constexpr u64 ITEMS_PER_BLOCK = THREADS_PER_BLOCK_X * ITEMS_PER_THREAD;

    const u64 size = std::size( keys );
    const u64 blocksDispatchedCount = ( size + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    thrust::device_vector<u32> prefixBinsOut;
    prefixBinsOut.resize( RADIX_BIN_COUNT * sizeof( radix_sort_key_t ), 0 );

    auto globalGroupCounter = thrust::device_new<u32>();
    *globalGroupCounter = 0;

    KernelOrochiRadixHistogramWarps<THREADS_PER_BLOCK_X, ITEMS_PER_THREAD, ORDER_MASK>
        <<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
            thrust::raw_pointer_cast( std::data( keys ) ), size,
            thrust::raw_pointer_cast( std::data( prefixBinsOut ) ),
            thrust::raw_pointer_cast( globalGroupCounter )
            );

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    thrust::device_delete( globalGroupCounter );

    return prefixBinsOut;
}


template<u64 THREADS_PER_BLOCK_X, u64 ITEMS_PER_THREAD, u32 ORDER_MASK>
auto DispatchRadixHistogramKernel( const thrust::device_vector<radix_sort_key_t>& keys )
{
    constexpr u64 ITEMS_PER_BLOCK = THREADS_PER_BLOCK_X * ITEMS_PER_THREAD;

    const u64 size = std::size( keys );
    const u64 blocksDispatchedCount = ( size + ITEMS_PER_BLOCK - 1 ) / ITEMS_PER_BLOCK;

    thrust::device_vector<u32> prefixBinsOut;
    prefixBinsOut.resize( radix_config_t<2>::DATA_SIZE, 0 );

    auto globalGroupCounter = thrust::device_new<u32>();
    *globalGroupCounter = 0;

    KernelRadixHistogram<THREADS_PER_BLOCK_X, ITEMS_PER_THREAD, ORDER_MASK>
        <<<blocksDispatchedCount, THREADS_PER_BLOCK_X>>>( 
            thrust::raw_pointer_cast( std::data( keys ) ), size,
            thrust::raw_pointer_cast( std::data( prefixBinsOut ) ),
            thrust::raw_pointer_cast( globalGroupCounter )
    );

    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    thrust::device_delete( globalGroupCounter );

    return prefixBinsOut;
}


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

    inline random_generator( u32 seed ) : gen{ seed } {}

    template<typename T>
    auto GenerateNIntegers( u64 n, T rangeMin, T rangeMax )
    {
        std::uniform_int_distribution<T> dist{ rangeMin, rangeMax };

        std::vector<T> out;
        out.resize( n );

        std::generate( std::begin( out ), std::end( out ), [&] () { return dist( gen ); } );

        return out;
    }
};

template<u64 THREADS_PER_BLOCK_X, typename T>
inline auto Reduction_CPU( const std::vector<T>& inputData )
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

template<u64 RADIX>
auto RadixHisto_CPU( const std::vector<radix_sort_key_t>& input )
{
    constexpr u64 RADIX_BIN_COUNT = 1 << RADIX;
    constexpr u64 DIGIT_COUNT = sizeof( radix_sort_key_t ) * 8 / RADIX;
    constexpr u64 DIGIT_MASK = ( 1u << RADIX ) - 1;

    std::vector<u32> prefixHisto;
    prefixHisto.resize( RADIX_BIN_COUNT * DIGIT_COUNT, 0 );

    for( auto elem : input )
    {
        for( u64 digit = 0; digit < DIGIT_COUNT; ++digit )
        {
            const u64 shift = digit * RADIX;
            u64 bin = ( elem >> shift ) & DIGIT_MASK;
            prefixHisto[ digit * RADIX_BIN_COUNT + bin ]++;
        }
    }

    for( u64 digit = 0; digit < DIGIT_COUNT; ++digit )
    {
        const u64 shift = digit * RADIX_BIN_COUNT;
        auto begIt = std::begin( prefixHisto ) + shift;
        auto endIt = begIt + RADIX_BIN_COUNT;
        std::exclusive_scan( std::execution::par, begIt, endIt, begIt, 0 );
    }
    
    return prefixHisto;
}

constexpr bool CHECK_CORRECTNESS = true;
constexpr u64 THREADS_PER_BLOCK = 512;

int main()
{
    constexpr u64 elemCount = 10'000'000;

    random_generator randGen = { std::random_device{}( ) };
    auto inputData = randGen.GenerateNIntegers<radix_sort_key_t>( 
        elemCount, std::numeric_limits<u16>::max(), std::numeric_limits<u32>::max() );

    auto radixHistoGroundTruth = RadixHisto_CPU<RADIX_DIGIT_BIT_COUNT>( inputData );

    cuda_context cudaCtx;
    
    thrust::device_vector<radix_sort_key_t> gpuInputData = { std::cbegin( inputData ), std::cend( inputData ) };

    auto orochiHistoCuda = DispatchOrochiHistogramKernel<RADIX_HISTOGRAM_THREADS_PER_BLOCK>( gpuInputData );

    auto orochiHistoCuda2 = DispatchOrochiHistogramKernelWarps<ASCENDING_MASK_32>( gpuInputData );

    std::vector<u32> orochiHistoCpu = { std::cbegin( orochiHistoCuda ), std::cend( orochiHistoCuda ) };
    std::vector<u32> orochiHistoCpu2 = { std::cbegin( orochiHistoCuda2 ), std::cend( orochiHistoCuda2 ) };

    if constexpr( CHECK_CORRECTNESS )
    {
        auto CmpOp = [] ( const auto& a, const auto& b ) { return a == b; };
        assert( ElementWiseRangeStrictCompare( radixHistoGroundTruth, orochiHistoCpu, CmpOp ) );
        assert( ElementWiseRangeStrictCompare( radixHistoGroundTruth, orochiHistoCpu2, CmpOp ) );
    }
    
    std::cout << "DONE!\n";
    return 0;
}
