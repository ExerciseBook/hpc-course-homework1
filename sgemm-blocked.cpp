#include "matrix_config.h"
#include "matrix_helper.h"
#include <array>
#include <immintrin.h>
#include <xmmintrin.h>

const char* sgemm_desc = "Simple blocked sgemm.";

// 分块大小
constexpr int block_size_M = 16; // M维度的分块大小
constexpr int block_size_N = 4;  // N维度的分块大小
constexpr int block_size_K = 4;  // K维度的分块大小

using vf = __m512;
using vfd = __m512d;

template <size_t N, class T>
std::array<vf, N> loadN(float* ptr, ptrdiff_t step, T loader) {
    std::array<vf, N> result{};
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
        result[i] = loader(ptr + i * step);
    }
    return result;
}

template <size_t N, class V, class T>
void storeN(float* ptr, ptrdiff_t step, std::array<V, N> value, T storer) {
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
void do_tile(int strideA, int strideB, int strideC, float* A, float* B, float* C, size_t K) {
    // A: 16 x K
    // B: K x 4
    // C: 16 x 4

    if (K % 4 != 0) {
        throw std::invalid_argument("");
    }

    using vfx4 = std::array<vf, 4>;
    using vfdx4 = std::array<vfd, 4>;

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
        vfdx4 tmp = {_mm512_castps_pd(_mm512_unpacklo_ps(b[0], b[1])),
                     _mm512_castps_pd(_mm512_unpacklo_ps(b[2], b[3])),
                     _mm512_castps_pd(_mm512_unpackhi_ps(b[0], b[1])),
                     _mm512_castps_pd(_mm512_unpackhi_ps(b[2], b[3]))};
        vfx4 bt = {_mm512_castpd_ps(_mm512_unpacklo_pd(tmp[0], tmp[1])),
                   _mm512_castpd_ps(_mm512_unpackhi_pd(tmp[0], tmp[1])),
                   _mm512_castpd_ps(_mm512_unpacklo_pd(tmp[2], tmp[3])),
                   _mm512_castpd_ps(_mm512_unpackhi_pd(tmp[2], tmp[3]))};

        // FMA
#pragma unroll
        for (size_t ki = 0; ki < 4; ++ki) {
#pragma unroll
            for (size_t n = 0; n < 4; ++n) {
                c[ki] = _mm512_fmadd_ps(a[n], bt[n], c[ki]);
                bt[n] = _mm512_shuffle_ps(bt[n], bt[n], _MM_SHUFFLE(0, 3, 2, 1));
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

void do_tile_remain(int strideA, int strideB, int strideC, float* A, float* B, float* C, size_t mainK, size_t remainK) {
    int M = block_size_M;
    int N = block_size_N;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float cij = C[i + j * strideC];
            for (int k = mainK; k < mainK + remainK; k++) {
                cij += A[i + k * strideA] * B[k + j * strideB];
            };
            C[i + j * strideC] = cij;
        }
    }
}

void do_tail_plain(int M, int K, int N, float* A, float* B, float* C) {
    int lda = M;
    int ldb = K;
    int ldc = M;
    /* For each row i of A */
    for (int i = 0; i < M; ++i) /* For each column j of B */
        for (int j = 0; j < N; ++j) {
            /* Compute C(i,j) */
            float cij = C[i + j * ldc];
            for (int k = 0; k < K; k++) cij += A[i + k * lda] * B[k + j * ldb];
            C[i + j * ldc] = cij;
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

    for (int m = 0; m < M; m += block_size_M) {
        int cur_M = min(block_size_M, M - m);

        for (int n = 0; n < N; n += block_size_N) {
            int cur_N = min(block_size_N, N - n);

            if (K % block_size_K == 0) {
                float* sub_A = A + m;
                float* sub_B = B + n * strideB;
                float* sub_C = C + m + n * strideC;
                do_tile(strideA, strideB, strideC, sub_A, sub_B, sub_C, K);
            } else {
                int main_K = K - (K % block_size_K);
                float* sub_A = A + m;
                float* sub_B = B + n * strideB;
                float* sub_C = C + m + n * strideC;

                if (main_K > 0) {
                    do_tile(strideA, strideB, strideC, sub_A, sub_B, sub_C, main_K);
                }
                int remaining_K = K % block_size_K;
                do_tile_remain(strideA, strideB, strideC, sub_A, sub_B, sub_C, main_K, remaining_K);
            }
        }
    }

    // TODO M N remains
}
