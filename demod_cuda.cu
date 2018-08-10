#include "gpu.h"

extern "C"{
    #include "functions.h"
}

//Demodulator
int main(int argc, char *argv[]){
//---------------------------------------------------------------------------------//
//QPSK parameters
    srand(time(0));
    int T = 1;                          //symbol period
    int N = 5;                          //upsampling factor
    float Ts = T/(float)N;              //sample period
    float fc = 1.5;                     //carrier frequency
    int nsym = 500;                     //number of symbols
    int bps = 2;                        //bits per symbol
    complx lut[4] = {                           //lookup table
        (complx){ 1/sqrt(2), 1/sqrt(2)}, 
        (complx){-1/sqrt(2), 1/sqrt(2)}, 
        (complx){ 1/sqrt(2),-1/sqrt(2)}, 
        (complx){-1/sqrt(2),-1/sqrt(2)}};
    float alpha = 0.55;                 //excess bandwidth
    int Lp = 12;                        //pulse truncation length
    int offs = Lp*2;                    //# of transient samples from filtering
    int nbits = nsym*bps;               //number of bits in signal 
    int nfilt =  5;                     //number of matched filters in filter bank
    int filtLen = 2*Lp*N+1;             //length of filters
    int sigLen = filtLen + N*nsym - 1;  //length of transmitted signal
    int rot_method = 0;                 //0: kurtosis
    float snr;
    if(argc > 1)snr = atof(argv[1]);
    else snr = 8.0;
    if(argc > 2)rot_method=atoi(argv[2]);
    else rot_method = 0;
    float phi;
    float max, max1;
    float cfo;
    float tau;
    float t_off = (-.5 + (float)(rand()%1001)*1e-3)*N*T;
    int ind_rot, ind_cfo;
    int ind, ind1;
    int print_FLAG = 0;
    int write_FLAG = 1;
    int convLen = sigLen+filtLen-1;
    int nfft = pow(2, (int)log2((float)convLen)+1);
    int np = 5;
    int id = 4*nfft*.05/N;
    int i2 = nfft-id;

    int M = 64;
    int blocksPerGrid = (nsym+M-1)/M;
    int threadsPerBlk = M;
    int cudaDeviceNum = 1;

    cudaError_t err;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDeviceNum);
    cudaSetDevice(cudaDeviceNum);
    assert(M <= prop.maxThreadsPerBlock);

//---------------------------------------------------------------------------------//
//  Allocate host memory
    int   *inbits, *outbits;            //bits
    int   *deltaIn, *deltaOut;          //for differential enc/dec
    complx *syms;                       //array of symbols
    float *ups_it, *ups_qt;             //upsampled symbols
    float *sig_it, *sig_qt;             //filtered signals
    float *cos_mix, *sin_mix;           //modulated signals
    float *pulse;                       //pulse shaping filter
    float *s;                           //transmitted signal
    float *noise;                       //additive white gaussian noise
    float *s_r;                         //received signal
    float *rot, *xr, *yr;               //rotated symbols
    inbits  = (int*)calloc(nsym*bps, sizeof(int));
    deltaIn = (int*)calloc(nsym*bps, sizeof(int));
    syms    = (complx*)calloc(nsym, sizeof(complx));
    ups_it  = (float*)calloc(N*nsym, sizeof(float));
    ups_qt  = (float*)calloc(N*nsym, sizeof(float));
    pulse   = (float*)calloc(filtLen, sizeof(float));
    sig_it  = (float*)calloc(sigLen, sizeof(float));
    sig_qt  = (float*)calloc(sigLen, sizeof(float));
    cos_mix = (float*)calloc(sigLen, sizeof(float));
    sin_mix = (float*)calloc(sigLen, sizeof(float));
    s       = (float*)calloc(sigLen, sizeof(float));
    noise   = (float*)calloc(sigLen, sizeof(float));
    s_r     = (float*)calloc(sigLen, sizeof(float));
    rot     = (float*)calloc(nsym*2, sizeof(float));
    outbits = (int*)calloc(nbits, sizeof(int));
    deltaOut= (int*)calloc(nbits, sizeof(int));

    xr = rot;
    yr = rot+nsym;

//---------------------------------------------------------------------------------//
//  Allocate device memory and setup cuFFT
    
    cufftHandle plan;
    cufftResult res;
    res = cufftPlan1d(&plan, nfft, CUFFT_C2C, 1);
    if(res!=0) fprintf(stderr, "error creating plan, %d\n", res);
    cufftComplex *dev_fftData;
    cudaMalloc(&dev_fftData, nfft*sizeof(cuComplex)); //allocate room on device for cufft data

    cudaStream_t s1, s2;
    err = cudaStreamCreate(&s1);
    if(err != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(err));
    err = cudaStreamCreate(&s2);
    if(err != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(err));

    float *dev_sig, *dev_rsig, *dev_rsig_it, *dev_rsig_qt; 
    cudaMalloc(&dev_sig, sigLen*sizeof(float));
    cudaMalloc(&dev_rsig, sigLen*2*sizeof(float));
    dev_rsig_it = dev_rsig;
    dev_rsig_qt = dev_rsig + sigLen;

    float *dev_filt, *dev_ups_it, *dev_ups_qt;
    cudaMalloc(&dev_filt, filtLen*sizeof(float));     //pulse_bank[i]

    float *dev_it_offs, *dev_qt_offs, *dev_offs;
    cudaMalloc(&dev_offs, convLen*2*sizeof(float));
    dev_it_offs = dev_offs; dev_qt_offs = dev_offs + convLen;

    float *dev_syms, *dev_isyms, *dev_qsyms;
    cudaMalloc(&dev_syms, nsym*2*sizeof(float));
    dev_isyms = dev_syms; dev_qsyms = dev_syms + nsym;

    float *dev_ym4, *dev_ym2, *dev_y2r, *dev_y2i, *dev_y2, *dev_kurt;
    cudaMalloc(&dev_ym4, nsym*sizeof(float));
    cudaMalloc(&dev_ym2, nsym*sizeof(float));
    cudaMalloc(&dev_y2, nsym*2*sizeof(float));
    dev_y2r = dev_y2; dev_y2i = dev_y2 + nsym;

    float *dev_ym4Sum, *dev_ym2Sum, *dev_y2rSum, *dev_y2iSum, *dev_ySum;
    cudaMalloc(&dev_ySum, 4*sizeof(float));
    dev_ym4Sum = dev_ySum;
    dev_ym2Sum = dev_ySum + 1;
    dev_y2rSum = dev_ySum + 2;
    dev_y2iSum = dev_ySum + 3;
    cudaMalloc(&dev_kurt, sizeof(float));

    float *dev_kurtArr, *dev_minArr, *dev_min;
    int *dev_idxArr, *dev_idx;
    cudaMalloc(&dev_kurtArr, nfilt*sizeof(float));
    cudaMalloc(&dev_minArr, nfilt*sizeof(float));
    cudaMalloc(&dev_idxArr, nfilt*sizeof(int));
    cudaMalloc(&dev_min, sizeof(float));
    cudaMalloc(&dev_idx, sizeof(int));

    float *dev_cmplxData;
    cudaMalloc(&dev_cmplxData, 2*convLen*sizeof(float));          //alloc. on device for cmplx pairs

    float *dev_maxArr, *dev_max;
    int *dev_indArr, *dev_indArr1, *dev_ind;
    cudaMalloc(&dev_maxArr, M*sizeof(float));
    cudaMalloc(&dev_max, sizeof(float));
    cudaMalloc(&dev_indArr, M*sizeof(int));
    cudaMalloc(&dev_indArr1, M*sizeof(int));
    cudaMalloc(&dev_ind, sizeof(int));

    float *dev_y4mag;
    cudaMalloc(&dev_y4mag, nfft*sizeof(float));

    float *dev_zeros;
    cudaMalloc(&dev_zeros, nsym*sizeof(float));
    dev_initArr<<<blocksPerGrid, threadsPerBlk>>>(dev_zeros, nsym);

    float *pulse_bank;              //bank of matched filters w/different time delays
    cudaMalloc(&pulse_bank, nfilt*filtLen*sizeof(float)); 

    float *dev_convArr;
    cudaMalloc(&dev_convArr, 2*nfilt*convLen*sizeof(float));
    dev_initArr<<<(2*convLen+M-1)/M, M>>>(dev_convArr, 2*nfilt*convLen);

    float kx, ky;
    float *dev_rotArr, *dev_xrot, *dev_yrot;
    cudaMalloc(&dev_rotArr, np*nsym*2*sizeof(float));
    
    //Build matched filter bank with various time delays
    for(int i=0; i<nfilt; i++) {
        tau = -N*T/2.0 + N*T*i/(float)(nfilt-1);
        dev_srrcDelay<<<(filtLen+M-1)/M,M>>>((pulse_bank+filtLen*i), alpha, (float)N, Lp, Ts, tau, 1);
    }

//---------------------------------------------------------------------------------//
//  Generate transmitted signal

    //Generate/read in random stream of digital data
    //for(int i=0;i<nsym*bps;i++)inbits[i] = rand()%2;    //random bits
    FILE *fbits;
    if(NULL==(fbits = fopen("inbits.bin", "rb"))){
        printf("ERROR: could not open %s for input.\n", "inbits.bin");
        exit(1);
    }
    fread(&nbits, sizeof(int), 1, fbits);
    assert(nbits == nsym*bps);
    fread(inbits, sizeof(int), nbits, fbits);
    fclose(fbits);

    //Bits --> symbols
    diffenc(inbits, deltaIn, nsym*bps);                   //differential encoding
    bits2sym(syms, deltaIn, nsym, bps, lut);              //bits to symbols

    //Upsample by N
    int j=0;
    for(int i=0; i<nsym; i++){
        ups_it[j] = syms[i].re;                 //in-phase
        ups_qt[j] = syms[i].im;                 //quadrature phase
        j+=N;
    }

    //Pass through pulse shaping filter
    srrcDelay(pulse, alpha, (float)N, Lp, Ts, t_off, 0);
    conv(sig_it, pulse, ups_it, filtLen, N*nsym);
    conv(sig_qt, pulse, ups_qt, filtLen, N*nsym);


    //Modulate I by cos and Q by sin; sum together to produce transmitted waveform
    float sigPow = 0;
    for(int i=0; i<sigLen; i++){
        cos_mix[i] = sqrt(2)*cos(2*PI*fc*T*i/N)*sig_it[i];
        sin_mix[i] =-sqrt(2)*sin(2*PI*fc*T*i/N)*sig_qt[i];
        s[i] = cos_mix[i] + sin_mix[i];
        sigPow += s[i]*s[i];
    }
    sigPow = sqrt(sigPow/sigLen);     //signal power(RMS)
    memcpy(s_r, s, sigLen*sizeof(float));

    //Add white noise (AWGN channel)
    float nPow = clt(noise, sigLen, sigPow, snr);
    for(int i=0; i<sigLen; i++){
        s_r[i] += noise[i];
    }
    if(print_FLAG>0){
        printf("%f\n", snr);
        printf("target snr: %f\n", snr);
        printf("act. snr: %f\n", 20*log10(sigPow/nPow));
    }

//---------------------------------------------------------------------------------//
//  Channel effects
    clock_t begin, end;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float v = -.05 + (float)(rand()%1001)*1e-4;
    float ph_off = deg2rad(-45+(float)(rand()%101)*.9);
    float arg = 2*PI*(fc+v)*T/(float)N;
    float arg_cfo;

//---------------------------------------------------------------------------------//
// Push data to GPU
    //De-mixing 
    cudaMemcpy(dev_sig, s_r, sigLen*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    begin = clock();

    dev_demix<<<(sigLen+M-1)/M,M>>>(dev_rsig_it, dev_sig, sigLen, 1, arg, ph_off);
    dev_demix<<<(sigLen+M-1)/M,M>>>(dev_rsig_qt, dev_sig, sigLen, 0, arg, ph_off);

    //Find min kurtosis; determine timing offset
    int minIdx = -1;
    int C = 256;
    for(int i=0; i<nfilt; i++){
        //Matched filtering
        dev_conv<<<(convLen+C-1)/C, C, 0, s1>>>((dev_convArr+2*i*convLen),     (pulse_bank+i*filtLen), dev_rsig_it, filtLen, sigLen, convLen);
        dev_conv<<<(convLen+C-1)/C, C, 0, s2>>>((dev_convArr+(2*i+1)*convLen), (pulse_bank+i*filtLen), dev_rsig_qt, filtLen, sigLen, convLen);

        //dev_convMult<<<(convLen+C-1)/C, C, 0, s1>>>((dev_convArr+2*i*convLen),     (pulse_bank+i*filtLen), dev_rsig_it, filtLen, sigLen, convLen);
        //dev_convMult<<<(convLen+C-1)/C, C, 0, s2>>>((dev_convArr+(2*i+1)*convLen), (pulse_bank+i*filtLen), dev_rsig_qt, filtLen, sigLen, convLen);

        //Downsampling
        dev_downsample<<<(nsym+M-1)/M,M,0,s1>>>(dev_isyms, (dev_convArr+2*i*convLen), nsym, offs, N);
        dev_downsample<<<(nsym+M-1)/M,M,0,s2>>>(dev_qsyms, (dev_convArr+(2*i+1)*convLen), nsym, offs, N);

        //Calculate complex kurtosis
        dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_isyms, dev_qsyms, nsym);
        dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
        dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
        dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurtArr+i, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
        //kurtosis array of length nfilt; find the minimum
    }

    //Get minimum kurtosis
    dev_getMin<<<1, threadsPerBlk>>>(dev_minArr, dev_idxArr, dev_kurtArr, nfilt);
    cudaMemcpy(&minIdx, dev_idxArr, sizeof(int), cudaMemcpyDeviceToHost);

    //Select filtered signal with the appropriate timing offset
    dev_ups_it = dev_convArr + 2*minIdx*convLen;
    dev_ups_qt = dev_ups_it + convLen;

    cudaMemcpy(dev_offs, dev_ups_it, convLen*2*sizeof(float), cudaMemcpyDeviceToDevice);

    //Carrier frequency offset correction
    assert(nfft>=convLen);
    dev_cmplxPow4<<<(convLen+M-1)/M, M>>>(dev_cmplxData, dev_ups_it, dev_ups_qt, convLen);
    
    //Find the frequency peak of y^4
    dev_initArr<<<(nfft+M-1)/M, M>>>(dev_fftData, nfft);                            //initialize data array with 0s
    dev_initArr<<<(id+M-1)/M,   M>>>(dev_maxArr, M);
    cudaMemcpy(dev_fftData, dev_cmplxData, convLen*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

    res = cufftExecC2C(plan, dev_fftData, dev_fftData, CUFFT_FORWARD); //**
    if(res != 0) fprintf(stderr, "Error in fft execution %d\n", res);
    dev_magComplx<<<(nfft+M-1)/M, M>>>(dev_y4mag, dev_fftData, nfft);
    ind_cfo = -1; max = -1, max1 = -1;

    //Get maximum value between 0 and .05
    dev_getMax<<<(id+M-1)/M,M>>>(dev_maxArr, dev_indArr, dev_y4mag, id);
    dev_getMax<<<    1     ,M>>>(dev_max, dev_ind, dev_maxArr, M);

    cudaMemcpy(&max, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ind, dev_ind, sizeof(int),  cudaMemcpyDeviceToHost);

    //Get maximum value between -.05 and 0
    dev_getMax<<<(id+M-1)/M,M>>>(dev_maxArr, dev_indArr1, (dev_y4mag+i2), id);
    dev_getMax<<<    1     ,M>>>(dev_max, dev_ind, dev_maxArr, M);

    cudaMemcpy(&max1, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ind1, dev_ind, sizeof(int),   cudaMemcpyDeviceToHost);

    //Take the largest of the maximums
    if(max > max1){
        cudaMemcpy(&ind_cfo, (dev_indArr+ind), sizeof(int), cudaMemcpyDeviceToHost);
    }else{
        cudaMemcpy(&ind_cfo, (dev_indArr1+ind1), sizeof(int), cudaMemcpyDeviceToHost);
        ind_cfo = ind_cfo + i2 - nfft;
    }

    //Perform the frequency offset correction
    cfo = -(float)ind_cfo*(float)N/(4.0*nfft); 
    arg_cfo = 2*PI*T*cfo/(float)N;
    dev_cfo<<<(convLen+M-1)/M, M, 0, s1>>>(dev_ups_it, dev_ups_qt, dev_it_offs, dev_qt_offs, arg_cfo, convLen); //de-mix

    //Downsample
    dev_downsample<<<(nsym+M-1)/M, M, 0, s1>>>(dev_isyms, dev_ups_it, nsym, offs, N);
    dev_downsample<<<(nsym+M-1)/M, M, 0, s2>>>(dev_qsyms, dev_ups_qt, nsym, offs, N);

    //Phase offset correction (multiple methods)
    if(rot_method == 0){                                //0: kurtosis
        float k=0, kmin;
        kmin = 1e4; ind_rot = -1;
        for(int i=0; i<np; i++){
            phi = -PI/4 + i*PI/(2*(np-1));
            //Perform the rotation
            dev_xrot = dev_rotArr + 2*i*nsym;
            dev_yrot = dev_rotArr + (2*i+1)*nsym;
            dev_rotate<<<(nsym+M-1)/M,M>>>(dev_xrot, dev_yrot, dev_isyms, dev_qsyms, phi, nsym);

            //Real kurtosis on in-phase branch
            dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_xrot, dev_zeros, nsym);
            dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
            dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurt, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
            cudaMemcpy(&kx, dev_kurt, sizeof(float), cudaMemcpyDeviceToHost);

            //Real kurtosis on quad-phase branch
            dev_cmplxKSums<<<blocksPerGrid, threadsPerBlk>>>(dev_ym4, dev_ym2, dev_y2r, dev_y2i, dev_yrot, dev_zeros, nsym);
            dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_ym4Sum, dev_ym4, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_ym2Sum, dev_ym2, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s1>>>(dev_y2rSum, dev_y2r, blocksPerGrid);
            dev_getSum<<<1, threadsPerBlk, 0, s2>>>(dev_y2iSum, dev_y2i, blocksPerGrid);
            dev_cmplxKurt<<<1, threadsPerBlk>>>(dev_kurt, dev_ym4Sum, dev_ym2Sum, dev_y2rSum, dev_y2iSum, nsym);
            cudaMemcpy(&ky, dev_kurt, sizeof(float), cudaMemcpyDeviceToHost);

            k = kx + ky;
            //assert(k<0);
            if(k >= 0) fprintf(stderr, "kurtosis not < 0: %f\n", k);
            if(k < kmin){
                kmin = k;
                ind_rot = i;
            }
        }
        phi = -PI/4+ind_rot*PI/(2*(np-1));
        cudaMemcpy(rot, dev_rotArr+2*ind_rot*nsym, nsym*2*sizeof(float), cudaMemcpyDeviceToHost);
    } else {                                            //no correction
        phi = 0;
        cudaMemcpy(rot, dev_syms, 2*nsym*sizeof(float), cudaMemcpyDeviceToHost);
    }

//---------------------------------------------------------------------------------//
//  Push data back to CPU

    //Symbol decisions
    decisionBlk(deltaOut, xr, yr, lut, nsym, bps);
    diffdec(outbits, deltaOut, nsym*bps);
    cudaEventRecord(stop);
    end = clock();
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    //Count bit errors
    int diff = 0;
    int total = nbits - 2;
    for(int i=2; i<nbits; i++){
        diff += (outbits[i] != inbits[i]);
        if(print_FLAG>0)printf("%d ", outbits[i]);
    }
    if(write_FLAG == 1){
        FILE *fsym;
        if(NULL==(fsym = fopen("sym.bin", "wb"))){
            printf("ERROR opening %s for output. :(\n", "sym.bin");
            exit(1);
        }
        assert(yr == xr + nsym);
        fwrite(&nsym, sizeof(int), 1, fsym);
        fwrite(xr, sizeof(float), nsym, fsym);
        fwrite(yr, sizeof(float), nsym, fsym);
        fclose(fsym);
    }
    if(print_FLAG == 1){
        printf("\n");
        printf("-----------------\n");
        printf("id: %d\n", id);
        printf("argmax: %d\n", ind_cfo);
        printf("cfo: %1.6f\n", cfo);
        printf("nfft: %d\n", nfft);
    }
    printf("-----------------\n");
    tau = -N*T/2.0 + N*(float)T*minIdx/(float)(nfilt-1);
    printf("Est. delay: %f/T \n", tau/(float)N);
    printf("Est. freq offset: %f\n", -cfo);
    printf("Act. freq offset: %f\n", v);
    printf("Est. phase offset: %f\n", rad2deg(phi));
    printf("SNR: \t%f\n", snr);
    printf("Bit Errors: %d/%d\n", diff, total);
    printf("BER: %e\n", (float)diff/total);
    printf("-----------------\n");

    float cpu_time_used = ((double)(end - begin))/CLOCKS_PER_SEC; 
    printf("CPU Execution Time: %f ms\n", cpu_time_used*1000);
    printf("GPU Execution Time: %f ms\n", ms);

//---------------------------------------------------------------------------------//
//  Cleanup
    cufftDestroy(plan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(dev_sig);
    cudaFree(dev_rsig);
    cudaFree(dev_filt);
    cudaFree(dev_offs);
    cudaFree(dev_syms);
    cudaFree(dev_convArr);
    cudaFree(dev_cmplxData);
    cudaFree(dev_fftData);
    cudaFree(dev_ym4);
    cudaFree(dev_ym2);
    cudaFree(dev_y2);
    cudaFree(dev_kurt);
    cudaFree(dev_ySum);
    cudaFree(dev_minArr);
    cudaFree(dev_idxArr);
    cudaFree(dev_kurtArr);
    cudaFree(dev_min);
    cudaFree(dev_idx);
    cudaFree(dev_y4mag);
    cudaFree(dev_max);
    cudaFree(dev_maxArr);
    cudaFree(dev_indArr);
    cudaFree(dev_indArr1);
    cudaFree(dev_ind);
    cudaFree(dev_zeros);
    cudaFree(dev_rotArr);
    cudaFree(pulse_bank);
    free(s_r);
    free(noise);
    free(rot);
    free(inbits);
    free(outbits);
    free(deltaIn);
    free(deltaOut);
    free(syms);
    free(ups_it);
    free(ups_qt);
    free(pulse);
    free(sig_it);
    free(sig_qt);
    free(cos_mix);
    free(sin_mix);
    free(s);
}
