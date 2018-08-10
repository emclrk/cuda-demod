# cuda-demod

files included:
functions.h   -- header file for C functions
functions.c   -- C functions 
gpu.h         -- header file for CUDA functions
gpu.cu        -- CUDA functions
demod_cuda.cu-- basic program to simulate a single block of QPSK-modulated data. The binary data is pulled from a file or generated randomly (revise code to select)
  nvcc demod_cuda.cu gpu.cu functioncs.c -o dmod_cuda -lcufft
ber_cuda.cu  -- program to simulate bit error rate testing
    nvcc ber_cuda.cu gpu.cu functions.c -o biterr_cuda -lcufft
