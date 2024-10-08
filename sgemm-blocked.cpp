#include "matrix_config.h"
#include "matrix_helper.h"
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

const char* sgemm_desc = "Simple blocked sgemm.";

using vf = __m512;

template<size_t N, class T>
std::array<vf, N> loadN(float *ptr, ptrdiff_t step, T loader) {
    std::array<vf, N> result{};
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        result[i] = loader(ptr + i * step);
    }
    return result;
}

template<size_t N, class V, class T>
void storeN(float *ptr, ptrdiff_t step, std::array<V, N> value, T storer) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        storer(ptr + i * step, value[i]);
    }
}

// A              B^T
//  1  2  3  4    a e i m
//  5  6  7  8    b f j n
//  9 10 11 12    c g k o
// 13 14 15 16    d h l p

// Elementwise product
// Left rotate 1 element for each N
//  1a+ 5b+ 9c+13d    2e+ 6f+10g+14h    3i+ 7j+11k+15l    4m+ 8n+12o+16p
//  1e+ 5f+ 9g+13h    2i+ 6j+10k+14l    3m+ 7n+11o+15p    4a+ 8b+12c+16d
//  1i+ 5j+ 9k+13l    2m+ 6n+10o+14p    ...
//  1m+ 5n+ 9o+13p    2a+ 6b+10c+14d    ...

// Swap to correct location with 2 (*4) blend
// a b c d   a c c a   a a a a
// b c d a   b b d d   b b b b
// c d a b   c a a c   c c c c
// d a b c   d d b b   d d d d


//  Compute a M=16 N=4 tile
template<size_t K>
void do_tile(int strideA, int strideB, int strideC,
             float *A, float *B, float *C) {
    // A: 16 x K
    // B: K x 4
    // C: 16 x 4

    static_assert(K % 4 == 0);

    using vfx4 = std::array<vf, 4>;

    vfx4 c{};
    // TODO: unroll this one reasonably
    for (size_t k = 0; k < K; k += 4) {
        // Load A
        vfx4 a = loadN<4>(A + k * strideA, strideA, _mm512_loadu_ps);

        // Load B
        vfx4 b = loadN<4>(B + k, strideB, [](float* ptr) {
            return _mm512_broadcast_f32x4(_mm_load_ps(ptr));
        });

        // B^T
        vfx4 tmp = {
                _mm512_castps_pd(_mm512_unpacklo_ps(b[0], b[1])),
                _mm512_castps_pd(_mm512_unpacklo_ps(b[2], b[3])),
                _mm512_castps_pd(_mm512_unpackhi_ps(b[0], b[1])),
                _mm512_castps_pd(_mm512_unpackhi_ps(b[2], b[3]))
        };
        vfx4 bt = {
                _mm512_castpd_ps(_mm512_unpacklo_pd(tmp[0], tmp[1])),
                _mm512_castpd_ps(_mm512_unpackhi_pd(tmp[0], tmp[1])),
                _mm512_castpd_ps(_mm512_unpacklo_pd(tmp[2], tmp[3])),
                _mm512_castpd_ps(_mm512_unpackhi_pd(tmp[2], tmp[3]))
        };

        // FMA
#pragma unroll
        for (size_t ki = 0; ki < 4; ++ki) {
#pragma unroll
            for (size_t n = 0; n < 4; ++n) {
                c[ki] = _mm512_fmadd_ps(a[n], bt[n], c[ki]);
                bt[n] = _mm512_shuffle_ps(bt[n], bt[n], _MM_SHUFFLE(0,3,2,1));
            }
        }
    }

    // swap
    __mmask16 mask1 = 0b1010101010101010;
    __mmask16 mask2 = 0b0110011001100110;
    __mmask16 mask3 = 0b1100110011001100;

    vfx4 tmp1 = {
            _mm512_mask_blend_ps(mask1, c[0], c[1]),
            _mm512_mask_blend_ps(mask1, c[1], c[0]),
            _mm512_mask_blend_ps(mask1, c[2], c[3]),
            _mm512_mask_blend_ps(mask1, c[3], c[2]),
    };

    vfx4 tmp2 = {
            _mm512_mask_blend_ps(mask2, tmp1[0], tmp1[2]),
            _mm512_mask_blend_ps(mask3, tmp1[1], tmp1[3]),
            _mm512_mask_blend_ps(mask2, tmp1[2], tmp1[0]),
            _mm512_mask_blend_ps(mask3, tmp1[3], tmp1[1]),
    };

    auto ci = loadN<4>(C, strideC, _mm512_loadu_ps);

    vfx4 co = {
            _mm512_add_ps(ci[0], tmp2[0]),
            _mm512_add_ps(ci[1], tmp2[1]),
            _mm512_add_ps(ci[2], tmp2[2]),
            _mm512_add_ps(ci[3], tmp2[3]),
    };

    storeN<4>(C, strideC, co, _mm512_storeu_ps);
}

void do_block(
        // clang-format off
        int M, int K, int N,
        int strideA, int strideB, int strideC,
        float *A, float *B, float *C,

        float* packed_A
        // clang-format on
) {
    for (int j = 0; j < N; ++j) {
        int i = 0;

        for (; i + 16 <= M; i += 16) {
            __m512 mmCij = _mm512_load_ps(&C(i, j));
            for (int k = 0; k < K; ++k) {
                auto p = ((i / 16) * K + k) * 16;
                __m512 a_packed = _mm512_load_ps(&packed_A[p]);
                __m512 b = _mm512_set1_ps(B[k + strideB * j]);
                mmCij = _mm512_fmadd_ps(a_packed, b, mmCij);
            }
            _mm512_store_ps(&C(i, j), mmCij);
        }

        for (; i + 8 <= M; i += 8) {
            __m256 mmCij = _mm256_load_ps(&C(i, j));
            for (int k = 0; k < K; ++k) {
                auto p = ((i / 16) * K + k) * 16;
                if (i % 16 >= 8) {
                    p += 8;
                }
                __m256 a_packed = _mm256_load_ps(&packed_A[p]);
                __m256 b = _mm256_set1_ps(B[k + strideB * j]);
                mmCij = _mm256_fmadd_ps(a_packed, b, mmCij);
            }
            _mm256_store_ps(&C(i, j), mmCij);
        }

        for (; i + 4 <= M; i += 4) {
            __m128 mmCij = _mm_load_ps(&C(i, j));
            for (int k = 0; k < K; ++k) {
                auto p = ((i / 16) * K + k) * 16;
                if (i % 16 >= 12) {
                    p += 12;
                } else if (i % 16 >= 8) {
                    p += 8;
                } else if (i % 16 >= 4) {
                    p += 4;
                } else {
                };
                __m128 a_packed = _mm_load_ps(&packed_A[p]);
                __m128 b = _mm_set_ps1(B[k + strideB * j]);
                mmCij = _mm_fmadd_ps(a_packed, b, mmCij);
            }
            _mm_store_ps(&C(i, j), mmCij);
        }

        for (; i < M; ++i) {
            float cij = C(i, j);
            for (int k = 0; k < K; ++k) {
                cij += packed_A[((i / 16) * K + k) * 16 + i % 16] * B[k + strideB * j];
            }
            C(i, j) = cij;
        }
    }
}

// dst 大小为 BLOCK_M * BLOCK_K 这么大
// src 是一个数组，他存储的内容为 A E I M B F J N C G K O D H L P Q U Y 3 R V Z 4 S W 1 5 T X 2 6
// src 的内容解释为
// A B C D Q R S T
// E F G H U V W X
// I J K L Y Z 1 2
// M N O P 3 4 5 6
//
// dst 的目标内容为
// A E I M B F J N C G K O D H L P Q U Y 3 R V Z 4 S W 1 5 T X 2 6
//
// 可以看到，dst 的意思是把 src 按照 4 行 4 行展开
// 行,列 -> dst 下标
//
// 当 src 的行列不能被4整除时我们这样操作
// src 是一个数组，他存储的内容为 A E I B F J C G K D H L M O Q N P R
// src 的内容解释为
// A B C D M N 0 0
// E F G H O P 0 0
// I J K L Q R 0 0
// 0 0 0 0 0 0 0 0
//
// dst 的目标内容为
// A E I 0 B F J 0 C G K 0 D H L 0 M O Q 0 N P R 0 0 0 0 0 0 0 0 0
//
// 一个混合的例子
// src 是一个数组，他存储的内容为 A E I B F J C G K D H L M O Q N P R
// src 的内容解释为
// A B C D 3 4 0 0
// E F G H 5 6 0 0
// I J K L 7 8 0 0
// M N O P 9 a 0 0
// Q R S T b c 0 0
// U V W X d e 0 0
// Y Z 1 2 f g 0 0
// 0 0 0 0 0 0 0 0
//
// dst 的目标内容为
// A E I M B F J N C G K O D H L P 3 5 7 9 4 6 8 a 0 0 0 0 0 0 0 0 Q U Y Z R V Z 0 S W 1 0 T X 2 0 b d f 0 c e g 0 0 0 0 0 0 0 0 0
void packA(float* dst, int srcStride, int srcHeight, int srcWidth, float* src) {
    for (int j = 0; j < srcWidth; j++) {
        int i = 0;
        for (; i + 16 <= srcHeight; i += 16) {
            __m512 lane = _mm512_load_ps(&src[srcStride * j + i]);
            _mm512_store_ps(&dst[((i / 16) * srcWidth + j) * 16], lane);
        }
        for (; i < srcHeight; i++) {
            dst[((i / 16) * srcWidth + j) * 16 + i % 16] = src[srcStride * j + i];
        }
    }
}

float packed_A[BLOCK_K * BLOCK_M];

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    // A
    //  M行 K列
    auto strideA = M;

    // B
    //  K行 N列
    auto strideB = K;

    // C
    //  M行 N列
    auto strideC = M;

// #if WIN32
//     float* packed_A = (float*) _aligned_malloc(BLOCK_M * BLOCK_K * sizeof(float), 4096);
//     if (!packed_A) {
//         throw std::runtime_error("Can not align malloc packed pool!");
//     }
// #else
//     float* packed_A = nullptr;
//     if (posix_memalign((void**) &packed_A, 4096, BLOCK_M * BLOCK_K * sizeof(float)) != 0) {
//         throw std::runtime_error("Can not align malloc packed pool!");
//     }
// #endif

    do_tile<1024>(strideA, strideB, strideC, A, B, C);
    return;

    for (int k = 0; k < K; k += BLOCK_K) {
        int K0 = min(BLOCK_K, K - k);
        for (int m = 0; m < M; m += BLOCK_M) {
            int M0 = min(BLOCK_M, M - m);
            packA(packed_A, strideA, M0, K0, &A(m, k));
            // print_matrix(&A(k, m), strideA, M0, K0);
            // print_matrix(packed_A, BLOCK_K, BLOCK_K, BLOCK_M);

            for (int n = 0; n < N; n += BLOCK_N) {
                int N0 = min(BLOCK_N, N - n);
                do_block(M0, K0, N0, strideA, strideB, strideC, &(A(m, k)), &(B(k, n)), &(C(m, n)), packed_A);
            }
        }
    }

// #if WIN32
//     _aligned_free(packed_A);
// #else
//     free(packed_A);
// #endif

    // throw std::runtime_error("");
}
