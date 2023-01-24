#include <stdint.h>

#include "xor.hu"

__global__ void gpu_xor(uint8_t *result, const uint8_t *op1, const uint8_t *op2, size_t opsize)
{
	int idx = threadIdx.x;

	if ((size_t)idx >= opsize)
		return;

	result[idx] = op1[idx] ^ op2[idx];
}
