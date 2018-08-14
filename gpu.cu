#include "gpu.h"

#define PI 3.14159265358979323846

__global__ void dev_fastConv(cuComplex *result, cuComplex *m1, cuComplex *m2, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int mid = tid;
    int step = blockDim.x*gridDim.x;
    float scale = 1/(float)len;
    while(tid < len){
        if(tid > len/2)mid = tid - (len/2);
        result[tid].x = scale*(m1[tid].x*m2[mid].x - m1[tid].y*m2[mid].y);
        result[tid].y = scale*(m1[tid].y*m2[mid].x + m1[tid].x*m2[mid].y);
        tid += step;
    }
}

__global__ void dev_convMult(float *convArr, float *filter, float *dataArr, int filtLen, int dataLen, int sigLen){
    int tid = blockIdx.x*blockDim.x; //thread index
    int l  = threadIdx.x;            //accumulator index
    int i  = 0;                      //current time index
    int m  = 0;                      //coefficient index
    int N  = filtLen;                //size of block and # of accumulators
    int str = blockDim.x*blockIdx.x;
    int end = blockDim.x*(blockIdx.x+1)+N;
    float x;
    register float acc = 0.f;
    while(tid < end){
        m = (l-i+N)%N;
        if(tid < dataLen && tid >=str)
            x = dataArr[tid];
        else
            x = 0.0;
        acc += filter[m]*x;
        if(m == 0){
            if(tid>= str && tid < sigLen){
                convArr[tid] = acc;
                acc = 0.0;
            }
        }
        i = (i+1+N)%N;
        tid++;
    }
}

__global__ void dev_conv(float *convArr, float *filter, float *dataArr, int filtLen, int dataLen, int sigLen){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int convLen = filtLen + dataLen - 1;
    int step = blockDim.x*gridDim.x;
    int l = threadIdx.x;
    __shared__ float filt[1024];              //put the filter in shared memory; max threads/block is 1024
    float tmp, x;
    int ind;
    while(l < filtLen){
        filt[l] = filter[l];
        l += blockDim.x;
    }
    __syncthreads();
    while(tid < convLen){
        tmp = 0.0;
        for(int i=0; i<filtLen; i++){
            ind = (tid - i + sigLen)%sigLen; //% is expensive
            if(ind > dataLen - 1) x = 0;
            else x = dataArr[ind];
            tmp += filt[i]*x;
        }
        convArr[tid] = tmp;
        tid += step;
    }
}

__global__ void dev_srrcDelay(float *pulse, float alpha, float N, int Lp, float Ts, float tau, int rev) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int len = 2*Lp*N+1;
    int step = blockDim.x*gridDim.x;
    float n, p;
    while(tid < len){
        n = tid - Lp*N - tau;
        if(n*Ts/N == 0) {
            p = (1+alpha*(4/PI - 1));
        }else if(n*Ts/N == Ts/(4*alpha) || n*Ts/N == -Ts/(4*alpha)){
            p = alpha*((1+2/PI)*sinf(PI/(4*alpha))+(1-2/PI)*(cosf(PI/(4*alpha))))/sqrtf(2);
        }else{
            p = (sinf(PI*(1-alpha)*n/N) + (4*alpha*n/N)*cosf(PI*(1+alpha)*n/N))/((n*PI/N)*(1-powf((4*alpha*n/N),2)));
        }
        p = p/sqrtf(N);
        if(rev == 1){
            pulse[len-1-tid] = p;
        }else{
            pulse[tid] = p;
        }

        tid += step;
    }
}

__global__ void dev_cmplxPow4(float *data, float *yr, float *yi, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float y2r, y2i, y4r, y4i;
    int step = blockDim.x*gridDim.x;
    while(tid < len){                               //len is the # of complex pairs
        y2r = yr[tid]*yr[tid] - yi[tid]*yi[tid];    //Re(y^2): a^2 - b^2
        y2i = 2*yr[tid]*yi[tid];                    //Im(y^2): 2*a*b
        y4r = y2r*y2r - y2i*y2i;                    //Re(y^4): re(y^2)^2 - im(y^2)^2
        y4i = 2*y2r*y2i;                            //Im(y^4): 2*re(y^2)*im(y^2)
        data[tid*2]   = y4r;                      //real/imaginary portions interleaved
        data[tid*2+1] = y4i;

        tid += step;
    }
}

__global__ void dev_magComplx(float *mag, cufftComplex *data, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        mag[tid] = data[tid].x*data[tid].x + data[tid].y*data[tid].y;
        tid += step;
    }
}

__global__ void dev_initArr(float *data, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        data[tid] = 0.0;
        tid += step;
    }
}
__global__ void dev_initArr(cufftComplex *data, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        data[tid].x = 0.0;
        data[tid].y = 0.0;
        tid += step;
    }
}

__global__ void dev_demix(float *sigBaseBand, float *sigPassBand, int sigLen, int inPhase, float arg, float ph_off){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    float sqrt2 = sqrtf(2);
    if(inPhase == 1){       //in-phase portion
        while(tid < sigLen){
            sigBaseBand[tid] = sqrt2*cosf(arg*tid + ph_off)*sigPassBand[tid];
           tid += step;
        }
    }else{                  //quadrature phase portion
        while(tid < sigLen){
            sigBaseBand[tid] =-sqrt2*sinf(arg*tid + ph_off)*sigPassBand[tid];
           tid += step;
       }
    }
}

__global__ void dev_downsample(float *syms, float *upsamp, int len, int offs, int N){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        syms[tid] = upsamp[(offs+tid)*N];
        tid += step;
    }
}

__global__ void dev_cfo(float *ups_it, float *ups_qt, float *I, float *Q, float arg, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float cos_theta, sin_theta;
    int step = blockDim.x*gridDim.x;
    while(tid<len){
        cos_theta = cosf(arg*tid);
        sin_theta = sinf(arg*tid);
        ups_it[tid] = I[tid]*cos_theta - Q[tid]*sin_theta;
        ups_qt[tid] = I[tid]*sin_theta + Q[tid]*cos_theta;
        tid += step;
    }
}

__global__ void dev_rotate(float *xr, float *yr, float *I, float *Q, float phi, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float C, S;
    int step = blockDim.x*gridDim.x;
    C = cosf(phi); S = sinf(phi);
    while(tid<len){
        xr[tid] = C*I[tid] - S*Q[tid];
        yr[tid] = S*I[tid] + C*Q[tid];
        tid += step;
    }
}

__global__ void dev_getMin(float *minArr, int *indArr, float *array, int len){  //***problem??
    __shared__ float minCache[1024];
    __shared__ float indCache[1024];
    int cacheIdx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    float min = 1e10;
    int ind = -1;
    while(tid<len){
        if(array[tid] < min){
            min = array[tid];
            ind = tid;
        }
        tid += step;
    }

    minCache[cacheIdx] = min;
    indCache[cacheIdx] = ind;
    __syncthreads();

    int i = blockDim.x/2;
    //Do the reduction
    while(i!=0){
        if(cacheIdx < i){
            if(minCache[cacheIdx+i] < minCache[cacheIdx]){
                //printf("%d\n", cacheIdx);
                minCache[cacheIdx] = minCache[cacheIdx+i]; 
                indCache[cacheIdx] = indCache[cacheIdx+i];
            }
        }
        __syncthreads();
        i/=2;
    }
    __syncthreads();
    if(cacheIdx == 0){
        minArr[blockIdx.x] = minCache[0];
        indArr[blockIdx.x] = indCache[0];
    }
}

__global__ void dev_getMax(float *maxArr, int *indArr, float *array, int len){
    __shared__ float maxCache[1024];
    __shared__ float indCache[1024];
    int cacheIdx = threadIdx.x;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float max = -1e10;
    int ind = -1;
    int step = blockDim.x*gridDim.x;
    while(tid<len){
        if(array[tid] > max){
            max = array[tid];
            ind = tid;
        }
        tid += step;
    }

    maxCache[cacheIdx] = max;
    indCache[cacheIdx] = ind;
    __syncthreads();

    int i = blockDim.x/2;
    //Do the reduction
    while(i!=0){
        if(cacheIdx < i){
            if(maxCache[cacheIdx+i] > maxCache[cacheIdx]){
                maxCache[cacheIdx] = maxCache[cacheIdx+i]; 
                indCache[cacheIdx] = indCache[cacheIdx+i];
            }
        }
        __syncthreads();
        i/=2;
    }
    __syncthreads();
    if(cacheIdx == 0){
        maxArr[blockIdx.x] = maxCache[0];
        indArr[blockIdx.x] = indCache[0];
    }
}

__global__ void dev_getSum(float *result, float *data,  int nsym){
    // Sum the elements of data; put partial results in result
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int cacheIdx = threadIdx.x;
    float localSum = 0;
    int step = blockDim.x*gridDim.x;
    __shared__ float cache[1024];
    while(tid < nsym){
        localSum += data[tid];
        tid += step;
    }

    cache[cacheIdx] = localSum;     //set cache values
    __syncthreads();                //sync threads before accessing cache data

    int i = blockDim.x/2;           //assume threads per block is a power of 2
    while(i!=0){
        if(cacheIdx < i){
            cache[cacheIdx] += cache[cacheIdx+i];
        }
        __syncthreads();
        i/=2;
    }
    __syncthreads();
    if(cacheIdx == 0){
        result[blockIdx.x] = cache[0];
    }
}

__global__ void dev_cmplxKSums(float *ym4Arr, float *ym2Arr, float *y2rArr, float *y2iArr, float *real, float *imag, int nsym){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int cacheIdx = threadIdx.x;
    int step = blockDim.x*gridDim.x;
    __shared__ float ym4Cache[1024];
    __shared__ float ym2Cache[1024];
    __shared__ float y2rCache[1024];
    __shared__ float y2iCache[1024];
    float ym4, ym2, y2r, y2i, mag;
    ym4 = 0.0; ym2 = 0.0; y2r = 0.0; y2i = 0.0;
    while(tid < nsym){
        mag  = real[tid]*real[tid] + imag[tid]*imag[tid];
        ym4 += mag*mag;
        ym2 += mag;
        y2r += real[tid]*real[tid] - imag[tid]*imag[tid];
        y2i += 2.0*real[tid]*imag[tid];
        tid += step;
    }
    ym4Cache[cacheIdx] = ym4;
    ym2Cache[cacheIdx] = ym2;
    y2rCache[cacheIdx] = y2r;
    y2iCache[cacheIdx] = y2i;
    __syncthreads();

    int i = blockDim.x/2;
    while(i!=0){
        if(cacheIdx < i){   
            ym4Cache[cacheIdx] += ym4Cache[cacheIdx+i];
            ym2Cache[cacheIdx] += ym2Cache[cacheIdx+i];
            y2rCache[cacheIdx] += y2rCache[cacheIdx+i];
            y2iCache[cacheIdx] += y2iCache[cacheIdx+i];
        }
        __syncthreads();
        i/=2;
    }
    __syncthreads();
    if(cacheIdx == 0){
        ym4Arr[blockIdx.x] = ym4Cache[0];
        ym2Arr[blockIdx.x] = ym2Cache[0];
        y2rArr[blockIdx.x] = y2rCache[0];
        y2iArr[blockIdx.x] = y2iCache[0];
    }
}

__global__ void dev_cmplxKurt(float *kurt, float *ym4, float *ym2, float *y2r, float *y2i, int nsym){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float k;
    if(tid == 0){
        float eym4, eym2, ey2m;
        eym4 = ym4[0]/(float)nsym;
        eym2 = (ym2[0]/(float)nsym)*(ym2[0]/(float)nsym);
        ey2m = (y2r[0]/(float)nsym)*(y2r[0]/(float)nsym) + (y2i[0]/(float)nsym)*(y2i[0]/(float)nsym);
        k = eym4 - 2*eym2 - ey2m;
        kurt[0] = k;
    }
}

__global__ void dev_mult(float *res, float *m1, float *m2, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        res[tid] = m1[tid]*m2[tid];
        tid += step;
    }
}

__global__ void dev_multCmplx(cuComplex *result, cuComplex *m1, cuComplex *m2, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int mid = tid;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        if(tid > len/2)mid = tid - (len/2);
        result[tid].x = m1[tid].x*m2[mid].x - m1[tid].y*m2[mid].y;
        result[tid].y = m1[tid].y*m2[mid].x + m1[tid].x*m2[mid].y;
        tid += step;
    }
}

__global__ void dev_multConst(float *res, float *in, float scale, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        res[tid] = in[tid]*scale;
        tid += step;
    }
}

__global__ void dev_multConst(cuComplex *res, cuComplex *in, float scale, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        res[tid].x = in[tid].x*scale;
        res[tid].y = in[tid].y*scale;
        tid += step;
    }
}

__global__ void dev_abs(float *data, int len){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    while(tid < len){
        if(data[tid] < 0)
            data[tid] = -data[tid];
        tid += step;
    }
}
