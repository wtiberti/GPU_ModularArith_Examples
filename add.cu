#include <stdio.h>

#include "add.hu"

__global__ void gpu_add(uint8_t *result, const uint8_t *op1, const uint8_t *op2, uint8_t *carryin, size_t opsize)
{
	int tid = threadIdx.x;

	if (tid >= opsize)
		return;

	__shared__ int selected;

	if (tid == 0)
		selected = 0;
	__syncthreads();

	for (int t = 0; t < blockDim.x; t++) {
		if (t == selected) {
			uint16_t sum = (uint16_t)op1[t] + (uint16_t)op2[t] + (uint16_t)carryin[t];

			result[t] = (uint8_t)(sum & 0xFF);

			if (t == opsize-1) {
				result[t+1] = (uint8_t)((sum >> 8) & 0xFF); // set the carry as last byte
			} else {
				carryin[t+1] = (uint8_t)((sum >> 8) & 0xFF); // set the carry
			}

			if (t < opsize-1)
				selected = selected + 1;
				//ready[t+1] = 1;
			__syncthreads();
		}
	}
}
