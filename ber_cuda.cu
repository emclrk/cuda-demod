#include "gpu.h"

extern "C"{
    #include "functions.h"
}

int main(int argc, char* argv[]){
    srand(time(0));
    int T = 1;                          //symbol period
    int N = 5;                          //upsampling factor
    float Ts = T/(float)N;              //sample period
    float fc = 1.5;                     //carrier frequency
    int nsym = 500;                     //number of symbols
    int bps = 2;                        //bits per symbol
    complx lut[4] = {                   //lookup table
        (complx){1/sqrt(2),1/sqrt(2)},
        (complx){-1/sqrt(2),1/sqrt(2)},
        (complx){1/sqrt(2),-1/sqrt(2)},
        (complx){-1/sqrt(2),-1/sqrt(2)}};
    float alpha = 0.55;                 //excess bandwidth
    int Lp = 12;                        //pulse truncation length
    int offs = Lp*2;                    //# of transient samples from filtering
    int nbits = nsym*bps;               //number of bits
    int filtLen = 2*Lp*N+1;             //length of filters
    int sigLen = filtLen + N*nsym - 1;  //length of transmitted signal
    int convLen = sigLen + filtLen - 1; //length of received, filtered signal
    int nfft = pow(2, (int)log2((float)convLen)+1);
    int rot_method = 0;                 //0: kurtosis 1: min-max
    int nfilt = 7;                      //number of matched filters in filter bank
    float snr;                          //signal-to-noise ratio
    float t_offset;                     //timing offset
    float v;                            //carrier frequency offset
    float ph_off;                       //phase offset
    float nvar;                         //noise variance
    float phi, tau;
    float k, kmin, max, max1;
    float kx, ky;
    float cfo, arg_cfo;
    float cos_mix, sin_mix;
    float arg = 2*PI*fc*T/(float)N;
    float minmax;
    int ind_cfo, ind_rot;
    int ind, ind1;
    int minIdx;
    int print_FLAG=0;
    int total_bits;
    int bit_errs;
    int diff = 0;
    int id = 4*nfft*.05/N;
    int i2 = nfft-id;
    int M = 64;
    int blocksPerGrid = (nsym+M-1)/M;
    int threadsPerBlk = M;
    int np;
    int devNum;
    int nPts;

    if(argc>1)nPts=atoi(argv[1]);
    else nPts = 15;
    assert(nPts < 21);
    float snr_arr[20];
    for(int i=0; i<nPts; i++)snr_arr[i] = i + 10*log10(bps) - 10*log10(T/Ts);

    if(argc>2)np = atoi(argv[2]);
    else np = 5;
    if(argc>3)nfilt = atoi(argv[3]);
    else nfilt = 5;
    if(argc>4)devNum = atoi(argv[4]);
    else devNum = 1;

    printf("======================\n");
    printf("# snr points: %d\n", nPts);
    printf("# phase points: %d\n", np);
    printf("# matched filters: %d\n", nfilt);
    printf("# of symbols: %d\n", nsym);
    printf("On device %d\n", devNum);
    printf("======================\n");
    cudaSetDevice(devNum);


    int   *inbits, *deltIn, *deltOut,  *outbits;    //bitstream
    complx *syms;                                   //array of symbols
    float *upsamp;
    float *upsamp_it, *upsamp_qt;                   //upsampled symbols
    float *sig, *sig_it, *sig_qt;                   //filtered signals
    float *noise;                                   //gaussian white noise
    float *s_r;                                     //received signal
    float *xr, *yr, *rot;                           //rotated symbols

    //Allocate memory
    inbits  = (int*)calloc(nsym*bps, sizeof(int));
    deltIn  = (int*)calloc(nsym*bps, sizeof(int));
    syms    = (complx*)calloc(nsym,  sizeof(complx));
    upsamp  = (float*)calloc(2*convLen, sizeof(float));
    sig     = (float*)calloc(2*sigLen, sizeof(float));
    s_r     = (float*)calloc(sigLen, sizeof(float));    
    noise   = (float*)calloc(sigLen, sizeof(float));
    rot     = (float*)calloc(nsym*2, sizeof(float));
    outbits = (int*)calloc(nbits,    sizeof(int));
    deltOut = (int*)calloc(nbits,    sizeof(int));
    
    upsamp_it = upsamp;
    upsamp_qt = upsamp + convLen;
    sig_it    = sig; 
    sig_qt    = sig + sigLen;
    xr        = rot; 
    yr        = rot + nsym;

    //Declare and allocate device memory
    cudaError_t err; 
    cufftHandle plan;
    cufftResult res;
    res = cufftPlan1d(&plan, nfft, CUFFT_C2C, 1);       //do this before allocations
    if(res!=0) fprintf(stderr, "error creating plan, %d\n", res);

    cufftComplex *dev_data;
    float *dev_cmplxData;
    cudaMalloc(&dev_data, nfft*sizeof(cufftComplex));
    cudaMalloc(&dev_cmplxData, 2*convLen*sizeof(float));  //alloc. for cmplx pairs

    float *dev_filt, *dev_sig_it, *dev_sig_qt, *dev_sig, *dev_signal;
    cudaMalloc(&dev_filt, filtLen*sizeof(float));
    cudaMalloc(&dev_signal, sigLen*2*sizeof(float));
    cudaMalloc(&dev_sig, sigLen*sizeof(float));
    dev_sig_it = dev_signal + 0*sigLen; 
    dev_sig_qt = dev_signal + 1*sigLen;

    float *dev_ups_it, *dev_ups_qt, *dev_it_offs, *dev_qt_offs, *dev_ups, *dev_offs;
    cudaMalloc(&dev_ups, convLen*2*sizeof(float));
    cudaMalloc(&dev_offs,convLen*2*sizeof(float));
    dev_ups_it = dev_ups; 
    dev_ups_qt = dev_ups + convLen;
    dev_it_offs = dev_offs; 
    dev_qt_offs = dev_offs + convLen;

    float *dev_convArr;
    cudaMalloc(&dev_convArr, 2*convLen*nfilt*sizeof(float));
    dev_initArr<<<(2*convLen*nfilt+M-1)/M, M>>>(dev_convArr, 2*convLen*nfilt);

    float *dev_isyms, *dev_qsyms, *dev_syms;
    cudaMalloc(&dev_syms, nsym*2*sizeof(float));
    dev_isyms = dev_syms; 
    dev_qsyms = dev_syms + nsym;

    float *dev_ym4, *dev_ym2, *dev_y2r, *dev_y2i, *dev_y;
    cudaMalloc(&dev_y, nsym*4*sizeof(float));
    dev_ym4 = dev_y + nsym*0;
    dev_ym2 = dev_y + nsym*1;
    dev_y2r = dev_y + nsym*2;
    dev_y2i = dev_y + nsym*3;

    float *dev_ym4Sum, *dev_ym2Sum, *dev_y2rSum, *dev_y2iSum, *dev_kurt, *dev_ySum;
    cudaMalloc(&dev_ySum, 5*sizeof(float));
    dev_ym4Sum = dev_ySum; 
    dev_ym2Sum = dev_ySum + 1;
    dev_y2rSum = dev_ySum + 2;
    dev_y2iSum = dev_ySum + 3;
    dev_kurt   = dev_ySum + 4;

    float *dev_kurtArr, *dev_minArr, *dev_min, *dev_array;
    int *dev_idxArr, *dev_idx;
    cudaMalloc(&dev_idxArr, nfilt*sizeof(int));
    cudaMalloc(&dev_min, sizeof(float));
    cudaMalloc(&dev_idx, sizeof(int));
    cudaMalloc(&dev_array, 2*nfilt*sizeof(float));
    dev_kurtArr = dev_array; dev_minArr = dev_array + nfilt;


    float *dev_maxArr, *dev_max;
    int *dev_indArr, *dev_indArr1, *dev_ind, *dev_p_indArr;
    cudaMalloc(&dev_maxArr, M*sizeof(float));
    cudaMalloc(&dev_max, sizeof(float));
    cudaMalloc(&dev_ind, sizeof(int));
    cudaMalloc(&dev_p_indArr, 2*M*sizeof(int));
    dev_indArr = dev_p_indArr; dev_indArr1 = dev_p_indArr + M;

    float *dev_y4mag;
    cudaMalloc(&dev_y4mag, nfft*sizeof(float));

    float *dev_xr, *dev_yr, *dev_zr, *dev_rot;
    cudaMalloc(&dev_rot, nsym*3*sizeof(float));
    dev_xr = dev_rot + 0*nsym; 
    dev_yr = dev_rot + 1*nsym;
    dev_zr = dev_rot + 2*nsym;
    dev_initArr<<<(nsym+M-1)/M,M>>>(dev_zr, nsym);

    float *dev_pulsebank;
    cudaMalloc(&dev_pulsebank, nfilt*filtLen*sizeof(float));

    //Build matched filter bank with various time delays
    for(int i=0; i<nfilt; i++) {
        tau = -N*T/2.0 + N*T*i/(float)(nfilt-1);
        dev_srrcDelay<<<(filtLen+M-1)/M,M>>>((dev_pulsebank+i*filtLen), alpha, (float)N, Lp, Ts, tau, 1);
    }
 
 cudaEvent_t start, stop;
 clock_t begin, end;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 err = cudaGetLastError();
 if(err != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(err));

//------------------------------------------------------------------------------------//
printf("Eb/No\tBER\n");


//Iterate over SNR values
cudaEventRecord(start);
begin = clock();
for(int x=0; x<nPts; x++){
    v = -.05 + (float)(rand()%1001)*1e-4;
    snr = snr_arr[x];
    nvar = .5 * pow(10, -(float)snr/10.0);
    bit_errs = 0;
    total_bits = 0;
while(bit_errs < 1000){
    t_offset = (-.5 + (float)(rand()%1001)*1e-3)*N*T;
    ph_off = deg2rad(-45+(float)(rand()%101)*.9);
    //Generate random stream of digital data
    for(int i=0;i<nsym*bps;i++)inbits[i] = rand()%2;    //random bits

    //Bits --> symbols
    diffenc(inbits, deltIn, nsym*bps);                   //differential encoding
    bits2sym(syms, deltIn, nsym, bps, lut);              //bits to symbols

    //Upsample by N
    int j=0;
    for(int i=0; i<nsym; i++){
        upsamp_it[j] = syms[i].re;                 //in-phase
        upsamp_qt[j] = syms[i].im;                 //quadrature phase
        j+=N;
    }

    //Pass through pulse shaping filter on I and Q branches
    dev_srrcDelay<<<(filtLen+M-1)/M, M>>>(dev_filt, alpha, (float)N, Lp, Ts, t_offset, 0);
    cudaMemcpy(dev_ups, upsamp, 2*convLen*sizeof(float), cudaMemcpyHostToDevice);
    dev_conv<<<(sigLen+M-1)/M, M>>>(dev_sig_it, dev_filt, dev_ups, filtLen, N*nsym, sigLen);
    dev_conv<<<(sigLen+M-1)/M, M>>>(dev_sig_qt, dev_filt, dev_ups+convLen, filtLen, N*nsym, sigLen);
    cudaMemcpy(sig, dev_signal, 2*sigLen*sizeof(float), cudaMemcpyDeviceToHost);

    //Modulate I by cos and Q by sin; sum together to produce transmitted waveform
    for(int i=0; i<sigLen; i++){
        cos_mix = sqrt(2)*cos(2*PI*(fc+v)*T*i/N + ph_off)*sig_it[i];
        sin_mix =-sqrt(2)*sin(2*PI*(fc+v)*T*i/N + ph_off)*sig_qt[i];
        s_r[i] = cos_mix + sin_mix;
    }

//------------------------------------------------------------------------------------//
//Channel noise

    boxmuller(noise, sigLen, 0, nvar);
    for(int i=0; i<sigLen; i++){
        s_r[i] += noise[i];
    }

//------------------------------------------------------------------------------------//
//Push data to GPU

    cudaMemcpy(dev_sig, s_r, sigLen*sizeof(float), cudaMemcpyHostToDevice);

    //Receiver processing (demixing)
    dev_demix<<<(sigLen+M-1)/M, M>>>(dev_sig_it, dev_sig, sigLen, 1, arg, 0);
    dev_demix<<<(sigLen+M-1)/M, M>>>(dev_sig_qt, dev_sig, sigLen, 0, arg, 0);

    //Timing offset detection/correction via Kurtosis
    minIdx = -1;
    for(int i=0; i<nfilt; i++){
        //Matched filtering
        dev_conv<<<(convLen+M-1)/M, M>>>((dev_convArr+2*i*convLen), (dev_pulsebank+i*filtLen), dev_sig_it, filtLen, sigLen, convLen);
        dev_conv<<<(convLen+M-1)/M, M>>>((dev_convArr+(2*i+1)*convLen), (dev_pulsebank+i*filtLen), dev_sig_qt, filtLen, sigLen, convLen);

        //Downsampling
        dev_downsample<<<(nsym+M-1)/M, M>>>(dev_isyms, (dev_convArr + 2*i*convLen), nsym, offs, N);
        dev_downsample<<<(nsym+M-1)/M, M>>>(dev_qsyms, (dev_convArr + (2*i+1)*convLen), nsym, offs, N);

        dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_isyms, dev_qsyms, nsym);
        dev_getSum<<<1, threadsPerBlk>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
        dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurtArr+i, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
    }

    //Perform matched filtering with appropriate timing offset
    dev_getMin<<<1, threadsPerBlk>>>(dev_minArr, dev_idxArr, dev_kurtArr, nfilt);
    cudaMemcpy(&minIdx, dev_idxArr, sizeof(int), cudaMemcpyDeviceToHost);

    dev_ups_it = dev_convArr + 2*minIdx*convLen;
    dev_ups_qt = dev_ups_it + convLen;

    cudaMemcpy(dev_offs, dev_ups_it, 2*convLen*sizeof(float), cudaMemcpyDeviceToDevice);

    //Carrier frequency offset correction (using FFT)
    assert(nfft>=convLen);
    dev_cmplxPow4<<<(convLen+M-1)/M, M>>>(dev_cmplxData, dev_ups_it, dev_ups_qt, convLen);  //data is interleaved real/imaginary
    dev_initArr<<<(nfft+M-1)/M, M>>>(dev_data, nfft);   //initialize data to zeros
    dev_initArr<<<(id+M-1)/M, M>>>(dev_maxArr, M);
    cudaMemcpy(dev_data, dev_cmplxData, convLen*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    res = cufftExecC2C(plan, dev_data, dev_data, CUFFT_FORWARD);
    if(res != 0) fprintf(stderr, "error in cufft execution: %d\n", res);
    dev_magComplx<<<(nfft+M-1)/M,M>>>(dev_y4mag, dev_data, nfft);

    dev_getMax<<<(id+M-1)/M,M>>>(dev_maxArr, dev_indArr, dev_y4mag, id);            //Get maximum value between 0 and .05
    dev_getMax<<<    1     ,M>>>(dev_max, dev_ind, dev_maxArr, M);
    cudaMemcpy(&max, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ind, dev_ind, sizeof(int),  cudaMemcpyDeviceToHost);

    dev_getMax<<<(id+M-1)/M,M>>>(dev_maxArr, dev_indArr1, (dev_y4mag+i2), id);      //Get maximum value between -.05 and 0
    dev_getMax<<<    1     ,M>>>(dev_max, dev_ind, dev_maxArr, M);

    cudaMemcpy(&max1, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ind1, dev_ind, sizeof(int),   cudaMemcpyDeviceToHost);
    if(max > max1){                                                                 //Take the largest of the maximums
        cudaMemcpy(&ind_cfo, (dev_indArr+ind), sizeof(int), cudaMemcpyDeviceToHost);
    }else{
        cudaMemcpy(&ind_cfo, (dev_indArr1+ind1), sizeof(int), cudaMemcpyDeviceToHost);
        ind_cfo = ind_cfo + i2 - nfft;
    }
    //Find the frequency peak of y^4
    cfo = -(float)ind_cfo*(float)N/(4.0*nfft); 
    arg_cfo = 2*PI*T*cfo/(float)N;
    dev_cfo<<<(convLen+M-1)/M, M>>>(dev_ups_it, dev_ups_qt, dev_it_offs, dev_qt_offs, arg_cfo, convLen);

    //Downsample
    dev_downsample<<<(nsym+M-1)/M, M>>>(dev_isyms, dev_ups_it, nsym, offs, N);
    dev_downsample<<<(nsym+M-1)/M, M>>>(dev_qsyms, dev_ups_qt, nsym, offs, N);

    //Phase offset correction (multiple methods)
    if(rot_method == 0){                                //0: kurtosis
        k=0;
        kmin = 1e4; ind_rot = -1;
        for(int i=0; i<np; i++){
            phi = -PI/4 + i*PI/(2*(np-1));
            dev_rotate<<<(nsym+M-1)/M,M>>>(dev_xr, dev_yr, dev_isyms, dev_qsyms, phi, nsym);
            dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_xr, dev_zr, nsym);
            dev_getSum<<<1, threadsPerBlk>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
            dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurt, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
            cudaMemcpy(&kx, dev_kurt, sizeof(float), cudaMemcpyDeviceToHost);

            dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_yr, dev_zr, nsym);
            dev_getSum<<<1, threadsPerBlk>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
            dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurt, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
            cudaMemcpy(&ky, dev_kurt, sizeof(float), cudaMemcpyDeviceToHost);
            k = kx + ky;
            //assert(k<0);
            if(k >= 0) fprintf(stderr, "%s: %f\n", "kurtosis is not less than 0", k);
            if(k<kmin){
                kmin = k;
                ind_rot = i;
            }
        }
        phi = -PI/4+ind_rot*PI/(2*(np-1));
        dev_rotate<<<(nsym+M-1)/M,M>>>(dev_xr, dev_yr, dev_isyms, dev_qsyms, phi, nsym);
        cudaMemcpy(rot, dev_rot, 2*nsym*sizeof(float), cudaMemcpyDeviceToHost);
    } else if(rot_method == 1) {                        //"min-max" method
        ind_rot = -1; max = -1;
        minmax = 1e10;
        for(int i=0; i<np; i++){
            phi = -PI/4 + i*PI/(2*(np-1));
            dev_rotate<<<(nsym+M-1)/M,M>>>(dev_xr, dev_yr, dev_isyms, dev_qsyms, phi, nsym);
            dev_abs<<<(nsym+M-1)/M,M>>>(dev_yr, nsym);
            dev_getMax<<<(nsym+M-1)/M,M>>>(dev_maxArr, dev_indArr, dev_yr, nsym);
            dev_getMax<<<    1     ,M>>>(dev_max, dev_ind, dev_maxArr, M);
            cudaMemcpy(&max, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
            if(max < minmax){
                minmax = max;
                ind_rot = i;
            }
        }
        phi = -PI/4+ind_rot*PI/(2*(np-1));
        dev_rotate<<<(nsym+M-1)/M,M>>>(dev_xr, dev_yr, dev_isyms, dev_qsyms, phi, nsym);
        cudaMemcpy(rot, dev_rot, 2*nsym*sizeof(float), cudaMemcpyDeviceToHost);
    } else {                                            //no correction
        phi = 0;
        cudaMemcpy(rot, dev_syms, 2*nsym*sizeof(float), cudaMemcpyDeviceToHost);
    }

//------------------------------------------------------------------------------------//
//Push data back to host

    //Symbol decisions
    decisionBlk(deltOut, xr, yr, lut, nsym, bps);
    diffdec(outbits, deltOut, nsym*bps);

    diff=0;
    for(int i=2; i<nbits; i++){                 //1st symbol likely wrong due to diff. enc
        diff += (outbits[i] != inbits[i]);
        //if(print_FLAG<1)printf("%d ", outbits[i]!=inbits[i]);
    }
    bit_errs += diff;
    total_bits += nbits-2;
}//end while
    printf("%d\t%e\n", x, (float)bit_errs/total_bits);
    if(print_FLAG>0){
        printf("-----------------\n");
        tau = -N*T/2.0 + N*T*minIdx/(float)(nfilt-1);
        printf("Est. delay: %f \n", tau);
        printf("Est. freq offset: %f\n", cfo);
        printf("Est. phase offset: %f\n", rad2deg(phi));
        printf("SNR: \t%f\n", snr);
        printf("Bit Errors: %d/%d\n", diff, nbits-2);
        printf("BER: %e\n", (float)diff/(nbits-2));
        printf("-----------------\n");
    }
}//end for

cudaEventRecord(stop);
end = clock();
cudaEventSynchronize(stop);
float gpu_time;
cudaEventElapsedTime(&gpu_time, start, stop);
double cpu_time = ((double)(end-begin))/CLOCKS_PER_SEC;

printf("GPU execution time: %f ms\n", gpu_time);
printf("CPU execution time: %f ms\n", cpu_time*1000);

//------------------------------------------------------------------------------------//
    //De-allocate memory
    cufftDestroy(plan);
    cudaFree(dev_data);
    cudaFree(dev_cmplxData);
    cudaFree(dev_filt);
    cudaFree(dev_signal);
    cudaFree(dev_sig);
    cudaFree(dev_ups);
    cudaFree(dev_offs);
    cudaFree(dev_syms);
    cudaFree(dev_convArr);
    cudaFree(dev_y);
    cudaFree(dev_ySum);
    cudaFree(dev_idxArr);
    cudaFree(dev_min);
    cudaFree(dev_idx);
    cudaFree(dev_array);
    cudaFree(dev_maxArr);
    cudaFree(dev_max);
    cudaFree(dev_ind);
    cudaFree(dev_p_indArr);
    cudaFree(dev_y4mag);
    cudaFree(dev_rot);
    cudaFree(dev_pulsebank);
    free(inbits);
    free(deltIn);
    free(syms);
    free(upsamp);
    free(sig);
    free(s_r);
    free(noise);
    free(rot);
    free(outbits);
    free(deltOut);
}
