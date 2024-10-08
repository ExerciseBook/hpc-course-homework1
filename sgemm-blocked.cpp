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
