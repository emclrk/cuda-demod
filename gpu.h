#ifndef GPU_H
#define GPU_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "cufft.h"

typedef struct{
    int T;          //sample period
    int N;          //upsampling factor
    int nsym;       //# of symbols (needed?)
    int bps;        //bits per symbol
    float fc;       //carrier frequency
    float alpha;    //excess bandwidth
    int  Lp;        //pulse truncation length
    int sigLen;     //length of signal
    float pow;      //signal power
}signalHdr;

__global__ void dev_convMult(float *convArr, float *filter, float *dataArr, int filtLen, int dataLen, int sigLen);
//Performs multi-threaded convolution
//filter:  pointer to the filter array
//dataArr: pointer to the data array
//filtLen: length of the filter
//dataLen: length of the data array
//sigLen:  length of the return array (= to filtLen + dataLen - 1)

__global__ void dev_conv(float *convArr, float *filter, float *dataArr, int filtLen, int dataLen, int sigLen);
//Performs "inner product" convolution
//convArr: pointer to the return array. holds the results of the convolution
//filter:  pointer to the filter array
//dataArr: pointer to the data array
//filtLen: length of the filter
//dataLen: length of the data array
//sigLen:  length of the return array (= to filtLen + dataLen - 1)

__global__ void dev_srrcDelay(float *pulse, float alpha, float N, int Lp, float Ts, float tau, int rev);
//Generates a square root raised cosine pulse with a timing delay
//pulse : pointer to the return array
//alpha : excess bandwidth parameter
//N     : upsampling factor
//Lp    : truncation length
//Ts    : sample period
//tau   : time delay parameter
//rev   : "reversal" flag: if 1, the pulse array is reversed

__global__ void dev_cmplxPow4(float *data, float *yr, float *yi, int len);
//Calculates the complex 4th power of the re/im input ararys
//data  : pointer to the return array. The real/imaginary outputs are interleaved. Should be allocated in main with 2*len*sizeof(float) bytes
//yr    : pointer to input array containing the 'real' portion of the data
//yi    : pointer to input array containing the 'imaginary' portion of the data
//len   : number of complex pairs

__global__ void dev_magComplx(float *mag, cufftComplex *data, int len);
//Computes the complex magnitude of the input data array
//mag   : pointer to the output magnitude array
//data  : pointer to the input array of complex data
//len   : number of complex points in the input array

__global__ void dev_initArr(float *data, int len);
//Initializes the elements of data to 0
//data  : pointer to array which is to be initialized to 0
//len   : number of points in the array

__global__ void dev_initArr(cufftComplex *data, int len);
//Initializes the complex elements of data to 0
//data  : pointer to array which is to be initialized to 0
//len   : number of complex points in the array

__global__ void dev_demix(float *sbb, float *spb, int sigLen, int inPhase, float arg, float ph_off);
//Performs the de-mixing step on the received signal
//sbb   : pointer to the return array (signal at baseband)
//spb   : pointer to the input array (signal at passband)
//sigLen: number of points in the signal
//inPhase : flag for in-phase/quad-phase. 1: in-phase portion; 0: quad-phase portion
//arg   : argument of the sin/cos (calculate in main to avoid computation in kernel)
//ph_off: phase offset

__global__ void dev_downsample(float *syms, float *upsamp, int len, int offs, int N);
//Performs the downsampling step on the filtered signal
//syms  : pointer to the return array of symbols
//upsamp: pointer to input array of upsampled data
//len   : number of output symbols
//offs  : number of transient points at the beginning (to be tossed)

__global__ void dev_cfo(float *ups_it, float *ups_qt, float *I, float *Q, float arg, int len);
//De-mixes the carrier frequency offset (rotation by a frequency term)
//ups_it: pointer to returned upsampled in-phase data array
//ups_qt: pointer to returned, upsampled quad-phase data array
//I     : pointer to the in-phase portion of the filtered signal
//Q     : pointer to the quad-phase portion of the filtered signal
//arg   : argument of the sin/cos (calculate in main to avoid computation in kernel)
//len   : length of the input signal

__global__ void dev_rotate(float *xr, float *yr, float *I, float *Q, float phi, int len);
//Performs the rotation of the symbols by an angle
//xr    : pointer to the returned, rotated in-phase data array
//yr    : pointer to the returned, rotated quad-phase data array
//I     : pointer to the in-phase portion of the input symbols
//Q     : pointer to the quad-phase portion of the input symbols
//phi   : angle by which the array is rotated
//len   : length of the input signal

__global__ void dev_getMin(float *minArr, int *indArr, float *array, int len);
//Finds the local minimum of an array. Reduction is performed using shared memory, so it should be run twice (the second time, blocks per grid should be = 1)
//minArr: pointer to the returned array of local minimums
//indArr: pointer to the returned array of the indexes of the local minimums
//array : pointer to the input array to be searched
//len   : length of the input array
//---------------------------------------Usage Example---------------------------------------//
//    dev_getMin<<<blocksPerGrid, threadsPerBlk>>>(dev_minArr, dev_idxArr, dev_dataArr, nfilt);
//    dev_getMin<<<1,             threadsPerBlk>>>(dev_min, dev_idx, dev_minArr, blocksPerGrid);
//      then, dev_idx holds the [location in indArr] of the [location of the minimum value in the input]
//      So: [minimum,ind] = min(dev_dataArr) -->    ind = dev_idxArr[dev_idx]
//                                              minimum = dev_dataArr[ind]

__global__ void dev_getMax(float *maxArr, int *indArr, float *array, int len);
//Finds the local maximum of an input array. Works in the same way as the getMin function.
//maxArr: pointer to the returned array of local maximums
//indArr: pointer to the returned array of the indexes of the local maximums
//array : pointer to the input array to be searched
//len   : length of the input array

__global__ void dev_getSum(float *result, float *data,  int nsym);
//Finds the local (partial) sums of an input array
//result: pointer to the returned array of local sums
//data  : pointer to the input data array to be summed
//nsym  : number of points in the input array
//Usual usage: if blocks per grid is 1, then *result is a single number. 

__global__ void dev_cmplxKSums(float *ym4Arr, float *ym2Arr, float *y2rArr, float *y2iArr, float *real, float *imag, int nsym);
//Computes the 'intermediate' sums for the values used in the complex kurtosis. Operates on a complex input array; so y = a + jb 
//ym4Arr: pointer to the output array for |y|^4
//ym2Arr: pointer to the output array for |y|^2
//y2rArr: pointer to the real portion of the output array for y^2
//y2iArr: pointer to the imag portion of the output array for y^2
//real  : pointer to the real portion of the input
//imag  : pointer to the imag portion of the input
//nsym  : number of complex symbols (aka length of the input arrays)
//Usage: calculate, then send each 'intermediate sum' array through the getSum function

__global__ void dev_cmplxKurt(float *kurt, float *ym4, float *ym2, float *y2r, float *y2i, int nsym);
//Computes the complex kurtosis given the sum of the values used in the kurtosis 
//kurt  : pointer to the output (the calculated kurtosis) which is equal to a single float value
//ym4   : pointer to the sum of |y|^4
//ym2   : pointer to the sum of |y|^2
//y2r   : pointer to the real portion of the sum of y^2
//y2i   : pointer to the imag portion of the sum of y^2
//nsym  : number of symbols in the original array for which the kurtosis is calculated (used for finding expected values)

__global__ void dev_mult(float *res, float *m1, float *m2, int len);
//Multiplies the elements of two arrays and returns their bit-wise product
//res   : pointer to the return array (results)
//m1    : pointer to the first array to be multiplied
//m2    : pointer to the second array to be multiplied
//len   : length of the input arrays

__global__ void dev_multCmplx(cuComplex *result, cuComplex *m1, cuComplex *m2, int len);
//Multiplies two arrays of complex numbers (a+jb)*(c+jd) = ac-bd + j*(
//result: pointer to the complex result of the multiplication
//m1    : pointer to the first array to be multiplied
//m2    : pointer to the second array to be multiplied; assumed to be a half-length array of complex numbers (e.g. the fft of a real signal)
//len   : length of the first input array (second input is of length len/2)

__global__ void dev_multConst(float *res, float *in, float scale, int len);
//Performs a scaling operation on the input array (multiplication by a constant value)
//res   : pointer to the returned result of the scaling
//in    : pointer to the input array to be scaled
//scale : value by which the array is multiplied
//len   : length of the input array

__global__ void dev_multConst(cuComplex *res, cuComplex *in, float scale, int len);
//Performs a scaling operation on the complex input array (multiplication by a constant value)
//res   : pointer to the returned complex result of the scaling
//in    : pointer to the complex input array to be scaled
//scale : value by which the array is multiplied
//len   : length of the input array

__global__ void dev_abs(float *data, int len);
//Returns the absolute value of the input data

#endif
