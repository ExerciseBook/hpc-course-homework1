#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

const char *sgemm_desc = "Simple blocked sgemm.";

constexpr size_t BLOCK_SIZE = 64;

constexpr size_t BLOCK_ROW = 64;
constexpr size_t BLOCK_COL = 128;

constexpr size_t UNROLL_NUM = 4;

constexpr size_t SIMD_UNROLL = 32;

auto inline min(auto a, auto b)
{
  return a < b ? a : b;
}

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
// 我现在在用 icc，有什么桥段可以简单用一下的呢？
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

int should_padding(int m, int n, int *new_m_ptr, int *new_n_ptr)
{
  *new_m_ptr = (m + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  *new_n_ptr = (n + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  return m != *new_m_ptr || n != *new_n_ptr;
}

static int64_t tempDur0 = 0, tempDur1 = 0;

float *get_padding_matrix(int lda, int m, int n, int new_m, int new_n, const float *A)
{
  float *new_A = NULL;
  int ret = posix_memalign((void **)&new_A, 4096, sizeof(float) * new_m * new_n);
  if (ret != 0)
  {
    fprintf(stderr, "Can not align malloc padding matrix!\n");
    exit(-1);
  }
  int j;
  for (j = 0; j < n; ++j)
  {
    memcpy(new_A + j * new_m, A + j * lda, sizeof(float) * m);
    memset(new_A + m + j * new_m, 0, sizeof(float) * (new_m - m));
  }
  for (; j < new_n; ++j)
  {
    memset(new_A + j * new_m, 0, sizeof(float) * new_m);
  }
  return new_A;
}

void back_padding(int lda, int m, int n, int new_m, int new_n, float *A, float *padding_A)
{
  for (int j = 0; j < n; ++j)
  {
    memcpy(A + j * lda, padding_A + j * new_m, sizeof(float) * m);
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
