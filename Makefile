all:
	nvcc -o add_xor_test main.cu add.cu xor.cu
