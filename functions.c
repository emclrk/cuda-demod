#include "functions.h"

void diffenc(int *inbits, int *delta, int nbits){
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
    int ind;
    for(int i=0; i<nsym; i++){
        ind = bits[i*2]*2 + bits[i*2+1];
        syms[i] = lut[ind];
    }
}

void srrcDelay(float *p, float alpha, float N, int Lp, float Ts, float tau, int rev){
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
    float dst, min;
    int bin[2];
    int ind;
    int k = 0;
    for(int i=0; i<nsym; i++){
        min = 1e3; ind = -1;
        for(int j=0; j<4; j++){
            dst = dstCmplx(isyms[i], qsyms[i], lut[j].re, lut[j].im);
            assert(dst>0);
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

float clt(float *noise, int len, float sigPow, float targetSNR){
    int write_FLAG=0;
    int randMax = 100;
    int nSamp   = 100;
    float total = 0;
    float power = 0;
    float randN = 1;
    float max = -1;
    float target, corr;
    float nBase=0;
    for(int j=0; j<len; j++){
        randN=0;
        for(int i=0; i<nSamp; i++){         
            randN += rand()%randMax;
        }
        randN /= (float)nSamp;
        total += randN;
        if(randN>max)max = randN;
        noise[j] = randN;
    }
    float mean = total/(float)len;
    for(int i=0; i<len; i++){
        noise[i] = (noise[i]-mean)/max; //shift and scale
        nBase += noise[i]*noise[i];
    }
    nBase = sqrt(nBase/len);    //base noise power (RMS)
    target = sigPow/(pow(10, targetSNR/20)); 
    corr = target/nBase;
    power = 0; total = 0;         // Multiply noise to hit target power
    for(int i=0; i<len; i++){
        noise[i] *= corr;
        power += noise[i]*noise[i];
        total += noise[i];
    }
    power = sqrt(power/len);        //noise power (RMS)
    mean = total/len;
    return power;
}

float dstCmplx(float srcReal, float srcImag, float dstReal, float dstImag){
    float distance;
    distance = sqrt((srcReal-dstReal)*(srcReal-dstReal) + (srcImag-dstImag)*(srcImag-dstImag));
    return distance;
}

float absCmplx2(float numReal, float numImag){  //squared magnitude of complex #
    return numReal*numReal + numImag*numImag;
}

float absCmplx(float numReal, float numImag){   //magnitude of complex #
    return sqrt(numReal*numReal + numImag*numImag);
}

float kurtCmplx(float *datReal, float *datImag, int len){
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
    return angle*PI/180.0;
}

float rad2deg(float phase){
    return phase*180.0/PI;
}

float getPower(float *data, int len){
   float pow = 0;
   for(int i=0; i<len; i++){
       pow += data[i]*data[i];
   }
   return sqrt(pow/len);
}

complx cmpSq(float numReal, float numImag){     //square of complex #
    complx retVal;
    retVal.re = numReal*numReal - numImag*numImag;
    retVal.im = 2*numReal*numImag;
    return retVal;
}


complx cmpAdd(complx num1, complx num2){        //addition of complex #s
    complx retVal;
    retVal.re = num1.re+num2.re;
    retVal.im = num1.im+num2.im;
    return retVal;
}

