#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

const char *sgemm_desc = "Simple blocked sgemm.";

constexpr size_t BLOCK_SIZE = 64;

auto inline min(auto a, auto b)
{
  return a < b ? a : b;
}

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int ldb, int ldc, int M, int K, int N, float *A, float *B, float *C)
{
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float cij = C[i + j * ldc];
      for (int k = 0; k < K; ++k)
      {
        cij += A[i + k * lda] * B[k + j * ldb];
      }
      C[i + j * ldc] = cij;
    }
  }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are MxK, KxN, MxN matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void custom_sgemm(int M, int K, int N, float *A, float *B, float *C)
{
  int lda = M;
  int ldb = K;
  int ldc = M;

  for (int i = 0; i < M; i += BLOCK_SIZE)
  {
    for (int j = 0; j < N; j += BLOCK_SIZE)
    {
      for (int k = 0; k < K; k += BLOCK_SIZE)
      {
        int M0 = min(BLOCK_SIZE, M - i);
        int K0 = min(BLOCK_SIZE, K - k);
        int N0 = min(BLOCK_SIZE, N - j);

        do_block(lda, ldb, ldc, M0, K0, N0, A + i + k * lda, B + k + j * ldb, C + i + j * ldc);
      }
    }
  }
}
