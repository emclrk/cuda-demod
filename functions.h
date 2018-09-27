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
void diffdec(int *outbits, int *delta, int nbits);
void bits2sym(complx *syms, int *bits, int nsym, int bps, complx *lut);
void srrcDelay(float *p, float alpha, float N, int Lp, float Ts, float tau, int rev);
void conv(float *retArr, float *filter, float *dataArr, int filtLen, int dataLen);
void decisionBlk(int *bits, float *isyms, float *qsyms, complx *lut, int nsym, int bps);
void boxmuller(float *noise, int len, float mean, float var);
float dstCmplx(float srcReal, float srcImag, float dstReal, float dstImag);
float absCmplx2(float numReal, float numImag);
float absCmplx(float numReal, float numImag);
float kurtCmplx(float *datReal, float *datImag, int len);
float kurtReal(float *data, int len);
float deg2rad(float angle);
float rad2deg(float phase);
float getPower(float *data, int len);
complx cmpSq(float numReal, float numImag);
complx cmpAdd(complx num1, complx num2);
float rand_float();
#endif
