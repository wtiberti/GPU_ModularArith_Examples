#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "xor.hu"
#include "add.hu"

#define OPSIZE 32

uint8_t op1[OPSIZE] = { 0x92, 0xa8, 0xef, 0x61, 0x6b, 0xc9, 0x13, 0xb1, 0x6d, 0xce, 0x4f, 0xdf, 0x57, 0xe9, 0x46, 0x20, 0x5d, 0x98, 0x7a, 0x25, 0xc7, 0x08, 0x06, 0x17, 0x90, 0xf5, 0xc5, 0x37, 0x9d, 0xf9, 0x7c, 0x17 };
uint8_t op2[OPSIZE] = { 0xed, 0x04, 0x4d, 0xb1, 0xb7, 0xf4, 0xa3, 0xed, 0xc1, 0x6c, 0x45, 0x46, 0x20, 0x2c, 0x88, 0x0a, 0xb7, 0xea, 0x24, 0x65, 0x94, 0x5f, 0x38, 0xa2, 0x2d, 0x13, 0x46, 0x7c, 0x70, 0x9d, 0xe3, 0x51 };
uint8_t test_xor[OPSIZE];
uint8_t test_sum[OPSIZE+1];
uint8_t test_expected[OPSIZE+1];

void test_XOR()
{
	uint8_t *y;
	uint8_t *a;
	uint8_t *b;

	// Compute the expected result
	for (int i = 0; i < OPSIZE; i++) {
		test_expected[i] = op1[i] ^ op2[i];
		printf("%02x ", test_expected[i]);
	}
	printf("\n");

	// alloc
	cudaMalloc(&a, OPSIZE);
	cudaMalloc(&b, OPSIZE);
	cudaMalloc(&y, OPSIZE);

	printf("cudaMalloc (a) = %p\n", a);
	printf("cudaMalloc (b) = %p\n", b);
	printf("cudaMalloc (y) = %p\n", y);

	cudaMemcpy(a, op1, OPSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(b, op2, OPSIZE, cudaMemcpyHostToDevice);

	printf("Inputs have been filled. Launching kernel...\n");

	// launch kernel
	gpu_xor<<<1, OPSIZE>>>(y, a, b, OPSIZE);

	// wait kernel
	cudaDeviceSynchronize();
	printf("Kernel done. Retrieving results...\n");
	cudaMemcpy(test_sum, y, OPSIZE, cudaMemcpyDeviceToHost);

	printf("Checking...\n");
	// comparison
	for (int i = 0; i < OPSIZE; i++) {
		if (test_sum[i] != test_expected[i]) {
			printf("Mismatch at index %d: expected %02x, found %02x\n", i, test_expected[i], test_sum[i]);
			break;
		} else {
			printf("Success at index %d\n", i);
		}
	}

	cudaFree(y);
	cudaFree(a);
	cudaFree(b);
}

void test_ADD()
{
	uint8_t *y;
	uint8_t *a;
	uint8_t *b;
	uint8_t *carries;

	// Compute the expected result
	uint8_t cpu_carries[OPSIZE+1];
	memset(cpu_carries, 0, OPSIZE);
	for (int i = 0; i < OPSIZE; i++) {
		uint16_t sum = op1[i] + op2[i] + cpu_carries[i];
		test_expected[i] = (uint8_t)(sum & 0xFF);
		if (i != OPSIZE-1)
			cpu_carries[i+1] = (uint8_t)(sum >> 8);
		else
			test_expected[i+1] = (uint8_t)(sum >> 8);
	}
	for (int i = OPSIZE-1; i >= 0; i--) {
		printf("%02x", test_expected[i]);
	}
	printf("\n");

	// alloc
	cudaMalloc(&a, OPSIZE);
	cudaMalloc(&b, OPSIZE);
	cudaMalloc(&y, OPSIZE+1);
	cudaMalloc(&carries, OPSIZE+1);

	printf("cudaMalloc (a) = %p\n", a);
	printf("cudaMalloc (b) = %p\n", b);
	printf("cudaMalloc (y) = %p\n", y);
	printf("cudaMalloc (carries) = %p\n", carries);

	cudaMemcpy(a, op1, OPSIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(b, op2, OPSIZE, cudaMemcpyHostToDevice);
	cudaMemset(carries, 0, OPSIZE+1);
	
	printf("Inputs have been filled. Launching kernel...\n");

	// launch kernel
	gpu_add<<<1, OPSIZE>>>(y, a, b, carries, OPSIZE);

	// wait kernel
	cudaDeviceSynchronize();
	printf("Kernel done. Retrieving results...\n");
	cudaMemcpy(test_sum, y, OPSIZE+1, cudaMemcpyDeviceToHost);

	printf("Checking...\n");
	// comparison
	for (int i = 0; i < OPSIZE; i++) {
		if (test_sum[i] != test_expected[i]) {
			printf("Mismatch at index %d: expected %02x, found %02x\n", i, test_expected[i], test_sum[i]);
			break;
		} else {
			printf("Success at index %d\n", i);
		}
	}

	cudaFree(carries);
	cudaFree(y);
	cudaFree(a);
	cudaFree(b);
}

int main(int argc, char *argv[])
{
	//test_XOR();
	test_ADD();
	cudaDeviceReset();
	return 0;
}
