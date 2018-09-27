#include "functions.h"

void diffenc(int *inbits, int *delta, int nbits){
//Performs differential encoding
//inbits: pointer to an array of input bits
//delta : pointer to output array of differentially encoded bits
//nbits : number of bits in the arrays
    int dprev[2]={0,0};
    int i;
    for(i=0; i<nbits-1; i+=2){
        if(inbits[i] == 0 && inbits[i+1] == 0){
            delta[i] = dprev[0];
            delta[i+1] = dprev[1];
        }else if(inbits[i] == 0 && inbits[i+1] == 1){
            delta[i] = dprev[1];
            delta[i+1] = 1-dprev[0];
        }else if(inbits[i] == 1 && inbits[i+1] == 0){
            delta[i] = 1-dprev[1];
            delta[i+1]=dprev[0];
        }else if(inbits[i] == 1 && inbits[i+1] == 1){
            delta[i] = 1-dprev[0];
            delta[i+1]=1-dprev[1];
        }
        dprev[0] = delta[i];
        dprev[1] = delta[i+1];
    }
}

void diffdec(int *bits, int *delta, int nbits){
//Performs differential decoding
//outbits: pointer to output array of bits
//delta  : pointer to input array of demodulated bits
//nbits  : number of bits in the arrays
    int dprev[2]={0,0};
    for(int i=0; i<nbits-1; i+=2){
        if(dprev[0] == 0 && dprev[1] == 0){
            bits[i]   = delta[i];
            bits[i+1] = delta[i+1];
        }else if(dprev[0] == 0 && dprev[1] == 1){
            bits[i]   = 1-delta[i+1];
            bits[i+1] = delta[i];
        }else if(dprev[0] == 1 && dprev[1] == 0){
            bits[i]   = delta[i+1];
            bits[i+1] = 1-delta[i];
        }else if(dprev[0] == 1 && dprev[1] == 1){
            bits[i]   = 1-delta[i];
            bits[i+1] = 1-delta[i+1];
        }
        dprev[0] = delta[i];
        dprev[1] = delta[i+1];
    }
}

void bits2sym(complx *syms, int *bits, int nsym, int bps, complx *lut){
//Converts an array of bits into complex QPSK symbols. realistically, this is only correct for qpsk
//syms  : complex array of output symbols (uses complx struct)
//bits  : input array of (differentially encoded) bits
//nsym  : number of symbols (should be number of bits / bits per symbol)
//bps   : bits per symbol (equals 2 for qpsk)
//lut   : complex array containing the lookup table for the constellation
    int ind;
    for(int i=0; i<nsym; i++){
        ind = bits[i*2]*2 + bits[i*2+1];
        syms[i] = lut[ind];
    }
}

void srrcDelay(float *p, float alpha, float N, int Lp, float Ts, float tau, int rev){
//Creates a square root raised cosine pulse shaping filter with a time delay
//p     : pointer to the output array (the pulse)
//alpha : excess bandwidth parameter
//N     : upsampling factor
//Lp    : pulse truncation length
//Ts    : sample period
//tau   : time delay
//rev   : reversal flag -- if rev == 1, the output array is the flipped version
    int i, len = 2*Lp*N+1;
    float *n;
    n  =   (float*)calloc(len, sizeof(float));
    for(i=0; i<len; i++){
        n[i] = i - Lp*N - tau;
    }
    for(i=0; i<len; i++){
        if(n[i]*Ts/N == 0){
            p[i] = (1+alpha*(4/PI - 1));
        }else if(n[i]*Ts/N == Ts/(4*alpha) || n[i]*Ts/N == -Ts/(4*alpha)){
            p[i] = alpha*((1+2/PI)*sin(PI/(4*alpha))+(1-2/PI)*(cos(PI/(4*alpha))))/sqrt(2);
        }
        else{
            p[i] = (sin(PI*(1-alpha)*n[i]/N) + (4*alpha*n[i]/N)*cos(PI*(1+alpha)*n[i]/N))/((n[i]*PI/N)*(1-pow((4*alpha*n[i]/N),2)));
        }
        p[i] = p[i]/sqrt(N);
    }
    if(rev == 1){
        memcpy(n, p, sizeof(float)*len);
        for(int i=0; i<len; i++){
            p[i] = n[len-1-i];
        }
    }
    free(n);
}

void conv(float *retArr, float *filter, float *dataArr, int filtLen, int dataLen){
//Performs linear convolution
//retArr: pointer to the return array
//filter: pointer to the input filter array
//dataArr: pointer to the input data array
//filtLen: length of the filter
//dataLen: length of the data array (# of data points)
    int convLen = dataLen+filtLen-1;
    float *buff;
    buff = (float*)calloc(convLen, sizeof(float));
    memcpy(buff, dataArr, sizeof(float)*dataLen);
    double mac;
    for(int n=0; n<convLen; n++){
        mac = 0.0;
        for(int m=0; m<filtLen; m++){
            mac+=(double)filter[m]*(double)buff[(n-m+convLen)%convLen];
        }
        retArr[n] = (double)mac;
    }
    free(buff);
}

void decisionBlk(int *bits, float *isyms, float *qsyms, complx *lut, int nsym,  int bps){
//Outputs hard decisions on demodulated symbols (inverse of bits2sym)
//bits  : pointer to the output array of decided bits
//isyms : pointer to the array of the in-phase portion of the symbols (input)
//qsyms : pointer to the array of the quad-phase portion of the symbols (input)
//lut   : complex array containing the lookup table for the constellation
//nsym  : number of input symbols
//bps   : number of bits per symbol (but really, it only works for qpsk)
    float dst, min;
    int bin[2];
    int ind;
    int k = 0;
    for(int i=0; i<nsym; i++){
        min = 1e3; ind = -1;
        for(int j=0; j<4; j++){
            dst = dstCmplx(isyms[i], qsyms[i], lut[j].re, lut[j].im);
            assert(dst>=0);
            if(dst < min){
                min = dst;
                ind = j;
            }
        }
        switch(ind){
            case 0: bits[k] = 0; bits[k+1] = 0; break;
            case 1: bits[k] = 0; bits[k+1] = 1; break;
            case 2: bits[k] = 1; bits[k+1] = 0; break;
            case 3: bits[k] = 1; bits[k+1] = 1; break;
            default: bits[k] = -1; bits[k+1] = -1; break;
        }
        k+=2;
    }
}

void boxmuller(float *noise, int len, float mean, float var){
    float u1, u2, z1, z2;
    float nstd = sqrt(var);
    assert(len%2 == 0);
    for(int i=0; i<len/2; i++){
      u1 = rand_float();
      u2 = rand_float();
      z1 = sqrt(-2*log(u1))*cos(2*PI*u2);
      z2 = sqrt(-2*log(u1))*sin(2*PI*u2);
      noise[i*2] = (z1 + mean)*nstd;
      noise[i*2+1]=(z2 + mean)*nstd;
    }
}

float dstCmplx(float srcReal, float srcImag, float dstReal, float dstImag){
//Calculates the distance between two complex numbers
//srcReal: real portion of the starting point
//srcImag: imaginary portion of the starting point
//dstReal: real portion of the ending point
//dstImag: imaginary portion of the ending point
//returns the distance (a float)
    float distance;
    distance = sqrt((srcReal-dstReal)*(srcReal-dstReal) + (srcImag-dstImag)*(srcImag-dstImag));
    return distance;
}

float absCmplx2(float numReal, float numImag){  //squared magnitude of complex #
//Calculates the squared magnitude of a complex number
//numReal: real portion of the input complex number
//numImag: imag portion of the input complex number
//returns the squared magnitude (a float)
    return numReal*numReal + numImag*numImag;
}

float absCmplx(float numReal, float numImag){   //magnitude of complex #
//Calculates the magnitude of a complex number
//numReal: real portion of the input complex number
//numImag: imag portion of the input complex number
//returns the magnitude (a float)
    return sqrt(numReal*numReal + numImag*numImag);
}

float kurtCmplx(float *datReal, float *datImag, int len){
//Computes the complex kurtosis of an input sequence
//datReal: pointer to the real portion of the input data
//datImag: pointer to the imaginary portion of the input data
//len    : length of the input data
//returns the complex kurtosis (a float)
    float ym2_sum, ym4_sum, ym2;
    float eym4, eym2, ey2m, k;
    float scale = 1/(float)len;
    complx y2, y2_sum, ey2m2;
    ym2 = 0;                                          //|y|^2
    ym2_sum = 0;                                      //sum(|y|^2)
    ym4_sum = 0;                                      //sum(|y|^4)
    y2.re = 0; y2.im = 0;                             //y^2
    y2_sum.re = 0; y2_sum.im = 0;                     //sum(y^2)

    for(int i=0; i<len; i++){
        ym2 = absCmplx2(datReal[i], datImag[i]);      //|y|^2
        ym2_sum += ym2;                               //accumulate |y|^2
        ym4_sum += (ym2*ym2);                         //accumulate |y|^4
        y2 = cmpSq(datReal[i], datImag[i]);           //y^2
        y2_sum = cmpAdd(y2_sum, y2);                  //accumulate y^2
    }
    eym4 = scale*ym4_sum;                             //E{|y|^4}
    eym2 = scale*ym2_sum;                             //E{|y|^2}
    ey2m2.re = scale*y2_sum.re;                       //E{y^2}
    ey2m2.im = scale*y2_sum.im;
    ey2m = absCmplx2(ey2m2.re, ey2m2.im);             //|E{y^2}|^2
    k = eym4 - 2*(eym2*eym2) - ey2m;                  //E{|y|^4} - 2(E{|y|^2})^2 - |E{y^2}|^2
    return k;
}

float kurtReal(float *data, int len){
//Computes the kurtosis for a zero-mean, real valued sequence
//data  : pointer to the input array of data
//len   : length of the input data
    float scale = 1/(float)len;
    float y2, y2_sum, y4_sum;
    float k;
    y2_sum = 0; y4_sum = 0;
    for(int i=0; i<len; i++){
        y2 = data[i]*data[i];
        y2_sum += y2;
        y4_sum += y2*y2;
    }
    y2_sum = y2_sum * scale;
    k = scale*y4_sum - 3*(y2_sum*y2_sum);
    return k;
}

float deg2rad(float angle){
//Converts an angle in degrees to an angle in radians
//angle : input angle in degrees
//returns the angle in radians (a float)
    return angle*PI/180.0;
}

float rad2deg(float phase){
//Converts an angle in radians to an angle in degrees
//phase : input angle in radians
//returns the angle in degrees (a float)
    return phase*180.0/PI;
}

float getPower(float *data, int len){
//Computes the RMS power of an input sequence
//data  : pointer to the input data
//len   : length of the input data
//returns the RMS power (a float)
   float pow = 0;
   for(int i=0; i<len; i++){
       pow += data[i]*data[i];
   }
   return sqrt(pow/len);
}

complx cmpSq(float numReal, float numImag){     //square of complex #
//Computes the square of a complex number
//numReal : real portion of the input complex number
//numImag : imag portion of the input complex number
//returns a complex number equal to (numReal + j*numImag)^2 (a complx)
    complx retVal;
    retVal.re = numReal*numReal - numImag*numImag;
    retVal.im = 2*numReal*numImag;
    return retVal;
}


complx cmpAdd(complx num1, complx num2){        //addition of complex #s
//Computes the sum of two complex numbers
//num1  : complex number to be added
//num2  : complex number to be added
//returns the complex sum of the two inputs (a complx)
    complx retVal;
    retVal.re = num1.re+num2.re;
    retVal.im = num1.im+num2.im;
    return retVal;
}

float rand_float(){
//Returns a random, uniformly distributed number between 0 and 1
  return (float)rand()/RAND_MAX;
}
