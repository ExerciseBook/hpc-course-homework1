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

        for (; i + 3 < M; i += 4) {
            int ip1 = i + 1;
            int ip2 = i + 2;
            int ip3 = i + 3;

            float cij = C(i, j);
            float cip1j = C(ip1, j);
            float cip2j = C(ip2, j);
            float cip3j = C(ip3, j);

            for (int k = 0; k < K; ++k) {
                cij += packed_A[k + i * BLOCK_K] * B[k + strideB * j];
                cip1j += packed_A[k + ip1 * BLOCK_K] * B[k + strideB * j];
                cip2j += packed_A[k + ip2 * BLOCK_K] * B[k + strideB * j];
                cip3j += packed_A[k + ip3 * BLOCK_K] * B[k + strideB * j];
            }

            C(i, j) = cij;
            C(ip1, j) = cip1j;
            C(ip2, j) = cip2j;
            C(ip3, j) = cip3j;
        }

        for (; i < M; ++i) {
            float cij = C(i, j);
            for (int k = 0; k < K; ++k) {
                // 所以 A的 i行k列 被映射到了 packA的 k行i列
                // cij += A[i + strideA* k]* B[k + strideB * j] ;
                cij += packed_A[k + i * BLOCK_K] * B[k + strideB * j];
            }
            C(i, j) = cij;
        }
    }
}

#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7)                                              \
    do {                                                                                                               \
        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;                                                         \
        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;                                                 \
        __t0 = _mm256_unpacklo_ps(row0, row1);                                                                         \
        __t1 = _mm256_unpackhi_ps(row0, row1);                                                                         \
        __t2 = _mm256_unpacklo_ps(row2, row3);                                                                         \
        __t3 = _mm256_unpackhi_ps(row2, row3);                                                                         \
        __t4 = _mm256_unpacklo_ps(row4, row5);                                                                         \
        __t5 = _mm256_unpackhi_ps(row4, row5);                                                                         \
        __t6 = _mm256_unpacklo_ps(row6, row7);                                                                         \
        __t7 = _mm256_unpackhi_ps(row6, row7);                                                                         \
        __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));                                                \
        __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));                                                \
        __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));                                                \
        __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));                                                \
        __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));                                                \
        __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));                                                \
        __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));                                                \
        __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));                                                \
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);                                                             \
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);                                                             \
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);                                                             \
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);                                                             \
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);                                                             \
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);                                                             \
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);                                                             \
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);                                                             \
    } while (0)

// 把 src 的 height 行 width 列，其 stride 为 stride
// 转置为 dst 的 width 行 height 列，其 stride 为 width
// 原来A的 i行j列 被打包到了 j行i列
void packA(float* dst, int height, int width, int stride, float* src) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst[width * i + j] = src[stride * j + i];
        }
    }
}

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

#if WIN32
    float* packed_A = (float*) _aligned_malloc(BLOCK_M * BLOCK_K * sizeof(float), 4096);
    if (!packed_A) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#else
    float* packed_A = nullptr;
    if (posix_memalign((void**) &packed_A, 4096, BLOCK_M * BLOCK_K * sizeof(float)) != 0) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#endif
    for (int k = 0; k < K; k += BLOCK_K) {
        int K0 = min(BLOCK_K, K - k);
        for (int m = 0; m < M; m += BLOCK_M) {
            int M0 = min(BLOCK_M, M - m);
            packA(packed_A, BLOCK_M, BLOCK_K, strideA, &A(m, k));
            // print_matrix(&A(k, m), strideA, BLOCK_M, BLOCK_K);
            // print_matrix(packed_A, BLOCK_K, BLOCK_K, BLOCK_M);

            for (int n = 0; n < N; n += BLOCK_N) {
                int N0 = min(BLOCK_N, N - n);
                do_block(M0, K0, N0, strideA, strideB, strideC, &(A(m, k)), &(B(k, n)), &(C(m, n)), packed_A);
            }
        }
    }

#if WIN32
    _aligned_free(packed_A);
#else
    free(packed_A);
#endif
}
