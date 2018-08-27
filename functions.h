#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define PI 3.14159265358979

typedef struct{
    float re;
    float im;
}complx;

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
}sigHead;

void diffenc(int *inbits, int *delta, int nbits);
//Performs differential encoding
//inbits: pointer to an array of input bits
//delta : pointer to output array of differentially encoded bits
//nbits : number of bits in the arrays
void diffdec(int *outbits, int *delta, int nbits);
//Performs differential decoding
//outbits: pointer to output array of bits
//delta  : pointer to input array of demodulated bits
//nbits  : number of bits in the arrays
void bits2sym(complx *syms, int *bits, int nsym, int bps, complx *lut);
//Converts an array of bits into complex QPSK symbols. realistically, this is only correct for qpsk
//syms  : complex array of output symbols (uses complx struct)
//bits  : input array of (differentially encoded) bits
//nsym  : number of symbols (should be number of bits / bits per symbol)
//bps   : bits per symbol (equals 2 for qpsk)
//lut   : complex array containing the lookup table for the constellation
void srrcDelay(float *p, float alpha, float N, int Lp, float Ts, float tau, int rev);
//Creates a square root raised cosine pulse shaping filter with a time delay
//p     : pointer to the output array (the pulse)
//alpha : excess bandwidth parameter
//N     : upsampling factor
//Lp    : pulse truncation length
//Ts    : sample period 
//tau   : time delay
//rev   : reversal flag -- if rev == 1, the output array is the flipped version
void conv(float *retArr, float *filter, float *dataArr, int filtLen, int dataLen);
//Performs linear convolution
//retArr: pointer to the return array
//filter: pointer to the input filter array
//dataArr: pointer to the input data array
//filtLen: length of the filter
//dataLen: length of the data array (# of data points)
void decisionBlk(int *bits, float *isyms, float *qsyms, complx *lut, int nsym, int bps);
//Outputs hard decisions on demodulated symbols (inverse of bits2sym)
//bits  : pointer to the output array of decided bits 
//isyms : pointer to the array of the in-phase portion of the symbols (input)
//qsyms : pointer to the array of the quad-phase portion of the symbols (input)
//lut   : complex array containing the lookup table for the constellation
//nsym  : number of input symbols
//bps   : number of bits per symbol (but really, it only works for qpsk)
float clt(float *noise, int len, float sigPow, float targetSNR);
//Generates gaussian white noise using the central limit theorem
//noise : output array of gaussian distributed noise (memory should be allocated in main)
//len   : number of points of noise to be generated
//sigPow: RMS power of the signal to which the noise will be added
//targetSNR : used to calculate the scaling for the noise
//returns the RMS noise power (a float)
float dstCmplx(float srcReal, float srcImag, float dstReal, float dstImag);
//Calculates the distance between two complex numbers 
//srcReal: real portion of the starting point
//srcImag: imaginary portion of the starting point
//dstReal: real portion of the ending point
//dstImag: imaginary portion of the ending point
//returns the distance (a float)
float absCmplx2(float numReal, float numImag);
//Calculates the squared magnitude of a complex number
//numReal: real portion of the input complex number
//numImag: imag portion of the input complex number
//returns the squared magnitude (a float)
float absCmplx(float numReal, float numImag);
//Calculates the magnitude of a complex number
//numReal: real portion of the input complex number
//numImag: imag portion of the input complex number
//returns the magnitude (a float)
float kurtCmplx(float *datReal, float *datImag, int len);
//Computes the complex kurtosis of an input sequence
//datReal: pointer to the real portion of the input data
//datImag: pointer to the imaginary portion of the input data
//len    : length of the input data
//returns the complex kurtosis (a float)
float kurtReal(float *data, int len);
//Computes the kurtosis for a zero-mean, real valued sequence
//data  : pointer to the input array of data
//len   : length of the input data
float deg2rad(float angle);
//Converts an angle in degrees to an angle in radians
//angle : input angle in degrees
//returns the angle in radians (a float)
float rad2deg(float phase);
//Converts an angle in radians to an angle in degrees
//phase : input angle in radians
//returns the angle in degrees (a float)
float getPower(float *data, int len);
//Computes the RMS power of an input sequence
//data  : pointer to the input data
//len   : length of the input data
//returns the RMS power (a float)
complx cmpSq(float numReal, float numImag);
//Computes the square of a complex number
//numReal : real portion of the input complex number
//numImag : imag portion of the input complex number
//returns a complex number equal to (numReal + j*numImag)^2 (a complx)
complx cmpAdd(complx num1, complx num2);
//Computes the sum of two complex numbers
//num1  : complex number to be added
//num2  : complex number to be added
//returns the complex sum of the two inputs (a complx)
#endif
