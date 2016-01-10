#include "main.h"
int gPeakChangeFreq;
int gInJobArrayIndex;
int gInEpoch;

int inReadParams(int inJobArrayIndex, int *inStrategy, double *kSigma, int *totNumChanges, double *constParam, double *vLen, double *hSev, double *wSev, int *inPeakNum, int *inInitSamples, int *inPSMPRecursionLevel, double *universalMeanPrior, double *rSeed);
void vdInitSamples(history *H, gsl_rng *rSamples, int n, double *bestNextGenotype, char* szFileName, int samplesToTake);
void backupPrGP(history *H, int n, int inStrategy, int maxRecursionLevel);
void evolveMovPeaks(gsl_rng *rSamples, double vLen, double hS, int inPeakNum);

#ifdef LOG_ERRORS_2_FILE
	FILE * pErrorFile;
#endif

int main(int argc, char* argv[]){

	int inJobArrayIndex;
	if(argc>1)
		inJobArrayIndex = atoi(argv[1]);
	else{
		printf("Invalid number of arguments\n");
		return -1;
	}
	gInJobArrayIndex = inJobArrayIndex;	//Only for log purposes (plotGpRegFunction)

	inDivergeCount = 0;
	time_t iniSeconds = time (NULL);
	time_t endSeconds;

	/* Read parameters from config file (or arguments) */
	int inStrategy=0;
	double kSigma=0;
	int totNumChanges=1;
	double rSeed=0;
	double constParam;
	double vLen=0;
	double hSev=0;
	double wSev=0;
	int inPeakNum = 0;
	int inInitSamples = 1;
	int inPSMPRecursionLevel = 0;
	double universalMeanPrior = 0;
	FILE * pFile;	//Output file

	if(inReadParams(inJobArrayIndex, &inStrategy, &kSigma, &totNumChanges, &constParam, &vLen, &hSev, &wSev, &inPeakNum, &inInitSamples, &inPSMPRecursionLevel, &universalMeanPrior, &rSeed) != 0){
		pdebug(__FILE__,__LINE__,__FUNCTION__,"Inside in readPArams, returning\n");
		return -1;
	}



	/* Initialize random stream */
	gsl_rng * r0 = initRandGen(rSeed);
	unsigned long int seedR = (unsigned long int) 10000*gsl_rng_uniform(r0);
	unsigned long int seedRSamples = (unsigned long int) 10000*gsl_rng_uniform(r0);
	unsigned long int seedMovPeaks = (unsigned long int) 10000*gsl_rng_uniform(r0);
	gsl_rng * r = initRandGen(seedR);
	gsl_rng * rSamples = initRandGen(seedRSamples);	
	gsl_rng_free (r0);

	/* initialize peaks */
	init_peaks(seedMovPeaks, vLen, hSev, wSev, inPeakNum);
	
	/* Evolve the moving peaks function EVOLVE_MOV_PEAKS number of times */
#ifdef EVOLVE_MOV_PEAKS
	evolveMovPeaks(rSamples, vLen, hSev, inPeakNum);
	return 0;
#endif

#ifdef LOG_ERRORS_2_FILE
	char szFileName[200];
	sprintf(szFileName,"errorLog/gpErr_%dd_%05d_%d_%1.1f_%1.1f.dat",D,inJobArrayIndex,inStrategy,vLen,hSev);
	if((pErrorFile = fopen(szFileName,"w"))==NULL){
		printf("Unable to open file for writing\n");
		return -1;
	}
	fprintf(pErrorFile,"Index\tOfflineError\tAvgError\tCurrentError\tBestKnown(performance)\tCurrentPerformance\trightPeak\tMaxPeak\n");
	fclose(pErrorFile);
#endif
	char szFileNameOut[200];

	/* Create structure to hold all the sampled points */
	history H;
	H.maxN = gPeakChangeFreq * totNumChanges;
	H.maxVals = (double *) malloc(H.maxN * sizeof(double));
	H.inInitSamples = inInitSamples;
	H.X = gsl_matrix_calloc(H.maxN, D);
	H.y = gsl_matrix_calloc(H.maxN, 1);
	H.w = (kernelParams*) malloc(H.maxN * sizeof(kernelParams));
	H.usePoint = gsl_vector_calloc(H.maxN);
	H.constParam = constParam;
	H.prevGP = (respSurf*) malloc(sizeof(respSurf));	//starts empty for the first epoch
	H.prevGP->epochId = -1;
	if(inStrategy==ST_CONST_MEAN_PRIOR)
		H.prevGP->constScalarMean = constParam;
	else
		H.prevGP->constScalarMean = universalMeanPrior;

	H.prevGP->X = NULL;
	H.prevGP->y = NULL;
	H.prevGP->w = NULL;
	H.prevGP->invKY = NULL;
	H.prevGP->meanPr = NULL;

	double bestNextGenotype[D];

	int n = 0;
	for(int ch=0; ch<totNumChanges; ch++){
		H.currPeriod = ch;
		gInEpoch = ch;	//Only for log purposes (plotGpRegFunction)

		if(inStrategy == ST_RANDOM){
			vdInitSamples(&H,rSamples,n,bestNextGenotype, szFileName, gPeakChangeFreq);
			n += (gPeakChangeFreq);
		}
		else{//Everything but random strategy
			if(ch==0 || inStrategy == ST_RESET || inStrategy == ST_CONST_MEAN_PRIOR){//take random initial samples.
				vdInitSamples(&H,rSamples,n,bestNextGenotype, szFileName, H.inInitSamples);
				n += (H.inInitSamples);
			}
			else if(inStrategy!=ST_IGNORE){//Ignore dismisses changes, so would not know it is a new epoch... plus has enough data samples to carry on.
				//for  ST_CONST_MEAN_PRIOR, no need to backup, so it is always "the first epoch"
				if(inStrategy==ST_PREV_SURF_MEAN_PRIOR){//Backup before taking new sample.
//					printXYMatrices("XY before backing up", H.X, H.y, 0, n);
//					printVector(H.usePoint,0,n);
					backupPrGP(&H, n-1, inStrategy, inPSMPRecursionLevel);	//At this point, we have just switched epoch, but need to backup the previous.
/*
					printf("After backing-up, it looks like:\n");
					printf("Previous X:\n");
					printMatrix(H.prevGP->X);
					printf("Previous Y:\n");
					printMatrix(H.prevGP->y);
					printf("H.prevGP->meanPr %p:\tH.prevGP->meanPr->X %p:\n",H.prevGP->meanPr,H.prevGP->meanPr->X);
*/
//					getchar();
				}

				getPreviousBest(bestNextGenotype, &H, n, gPeakChangeFreq);//First sample of new epoch
				for(int d=0; d<D; d++)
					gsl_matrix_set(H.X,n,d,bestNextGenotype[d]);

				double currVal = eval_movpeaks(bestNextGenotype);
				gsl_matrix_set(H.y,n,0,currVal);
				if(currVal > H.maxVals[n-1])
					H.maxVals[n] = currVal;
				else
					H.maxVals[n] = H.maxVals[n-1];
				gsl_vector_set(H.usePoint,n,USE_POINT);
#ifdef LOG_ERRORS_2_FILE
				if((pErrorFile = fopen(szFileName,"a"))==NULL){
					printf("Unable to open file for appending\n");
					return -1;
				}
				fprintf(pErrorFile,"%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d\n",n,get_offline_error(),get_avg_error(),get_current_error(),get_current_maximum(),dummy_eval(bestNextGenotype),get_right_peak(),get_maximum_peak());
				fclose(pErrorFile);
#endif
				n++;
			
#ifdef INIT_RAND_SAMPLES_PER_EPOCH			//Take all the remaining H.inInitSamples-1 randomly
				vdInitSamples(&H,rSamples,n,bestNextGenotype, szFileName, H.inInitSamples-1);
				n += (H.inInitSamples-1);
				gsl_rng_uniform(rSamples);	//Draw an additional random sample to use the exact same as RESET strategy and increase the power of the comparison.
#endif


			}

			do
			{
				getNextSample(&H, r, n, bestNextGenotype, inStrategy, kSigma);
#ifdef LOG_ERRORS_2_FILE
				if((pErrorFile = fopen(szFileName,"a"))==NULL){
					printf("Unable to open file for appending\n");
					return -1;
				}
				fprintf(pErrorFile,"%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d\n",n,get_offline_error(),get_avg_error(), \
				get_current_error(),get_current_maximum(),dummy_eval(bestNextGenotype),get_right_peak(),get_maximum_peak());
				fclose(pErrorFile);
#endif

/*
				printf("n=%d\toffline=%f\tavg=%f\tcurrError=%f\tcurrentBestKnown=%f\tdummyEval=%f\trightPeak=%d\n",n, \
				get_offline_error(),get_avg_error(),get_current_error(),get_current_maximum(), \
				dummy_eval(bestNextGenotype),get_right_peak());
				getchar();
//*/				
				n++;
//				dumpLog(&H,n,inJobArrayIndex,inStrategy, vLen, hSev);

//				if(n>25)
//					getchar();

//		dumpLog(&H,n,inJobArrayIndex,inStrategy, vLen, hSev);
//		printf("LogDumped\tn=%d\n",n);
//		getchar();
			
			}while((n)%gPeakChangeFreq);
		}//end:	else(Everything but random strategy)

		
//		plotLandscape(ch, inJobArrayIndex, inStrategy);
//		dumpLog(&H,n,inJobArrayIndex,inStrategy, vLen, hSev);
//		printf("LogDumped\n");
//		getchar();


		/* Track performance after every epoch */
		memset(szFileNameOut,0x00,sizeof(szFileNameOut));
		sprintf(szFileNameOut,"%s%dd_%d.dat",PATH_OUT,D,ch);
		if((pFile = fopen(szFileNameOut,"a+"))==NULL){
			printf("Unable to open file for writing\n");
			return -1;
		}
		endSeconds = time (NULL);
		fprintf(pFile,"%d\t%ld\t%d\t%d\t%2.0f\t%d\t%d\t%f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%f\t%10.0f\t%f\t%f\n",inJobArrayIndex,(endSeconds-iniSeconds),D,inStrategy, kSigma, gPeakChangeFreq, totNumChanges, constParam, vLen, hSev, wSev, inPeakNum, inInitSamples, inPSMPRecursionLevel, universalMeanPrior, rSeed, get_offline_error(),get_avg_error());
		fclose(pFile);
		pFile=NULL;
		change_peaks();
	}//end for totNumChanges


	dumpLog(&H,n,inJobArrayIndex,inStrategy, vLen, hSev);
/*
	if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam == LEARN_TIMESCALE)
		dumpParams(&H,n,inJobArrayIndex,D+1);
	else
		dumpParams(&H,n,inJobArrayIndex,D);
*/
//		printXYMatrices("All samples", H.X, H.y, 0, n);

	
	//Write output to file
	memset(szFileName,0x00,sizeof(szFileName));
	sprintf(szFileName,"%s%dd.dat",PATH_OUT,D);
	if((pFile = fopen(szFileName,"a+"))==NULL){
		printf("Unable to open file for writing\n");
		return -1;
	}
	endSeconds = time (NULL);
	fprintf(pFile,"%d\t%ld\t%d\t%d\t%2.0f\t%d\t%d\t%f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%f\t%10.0f\t%f\t%f\n",inJobArrayIndex,(endSeconds-iniSeconds),D,inStrategy, kSigma, gPeakChangeFreq, totNumChanges, constParam, vLen, hSev, wSev, inPeakNum, inInitSamples, inPSMPRecursionLevel, universalMeanPrior, rSeed, get_offline_error(),get_avg_error());
	fclose(pFile);




	gsl_rng_free (r);
	gsl_rng_free (rSamples);
	gsl_matrix_free(H.X);
	gsl_matrix_free(H.y);
	free(H.w);
	gsl_vector_free(H.usePoint);

}

/**********************************************************************************************************************************************************************
Reads parameters from 1 same file, different line according to the input.
Arguments in the file:
	inStrategy:	{0->Reset,1->Ignore,2->Inrease Noise,3->Temporal Discount... refer to main.h} (GP Inference)
	kSigma: 	{-1->EGO, 0->Mean, >0->kSigma} (Next best sample Strategy: EGO or how far away from the mean to be considered)
	gPeakChangeFreq:How often the peaks change
	totNumChanges:	Number of times the peaks will change over time
	constParam:	Fixed learning timescale to avoid the extra dimension learning for the D+1 model. LEARN_TIMESCALE (=-3) to ignore it and learn it from data. Use it
			to provide noise level to be used for ST_INCREASE_NOISE, or to provide a constant mean for ST_CONST_MEAN_PRIOR model. Ignored otherwise.
	rSeed:		Random seed for this run

	
Returns:
	 0 if parameters OK
	-1 on error
************************************************************************************************************************************************************************/
int inReadParams(int inJobArrayIndex, int *inStrategy, double *kSigma, int *totNumChanges, double *constParam, double *vLen, double *hSev, double *wSev, int *inPeakNum, int *inInitSamples, int *inPSMPRecursionLevel, double *universalMeanPrior, double *rSeed){
	char szFileName[200];

	sprintf(szFileName,"%s%dd.dat",PATH_IN,D);
	int inLine = 0; int inFStatus=0;
	FILE * pFile;
	if((pFile = fopen(szFileName,"r"))==NULL){
		printf("Unable to open file %s\n",szFileName);
		return -1;
	}

	//Get to the desired line
	while(inLine < inJobArrayIndex && inFStatus != EOF){
		inFStatus = fscanf(pFile,"%d\t%lf\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%d\t%d\t%d\t%lf\t%lf\n",inStrategy, kSigma, &gPeakChangeFreq, totNumChanges, constParam, vLen, hSev, wSev, inPeakNum, inInitSamples, inPSMPRecursionLevel, universalMeanPrior, rSeed);
		inLine++;
	}
	inFStatus = fscanf(pFile,"%d\t%lf\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%d\t%d\t%d\t%lf\t%lf\n",inStrategy, kSigma, &gPeakChangeFreq, totNumChanges, constParam, vLen, hSev, wSev, inPeakNum, inInitSamples, inPSMPRecursionLevel, universalMeanPrior, rSeed);
	fclose(pFile);
	if(inFStatus == EOF)
		return -1;
	else
		return 0;
}

void vdInitSamples(history *H, gsl_rng *rSamples, int n, double *bestNextGenotype, char* szFileName, int samplesToTake){

	/* Generate some (initSamples) initial samples for the first epoch (or each epoch when using RESET) */
	double tmpGenotype[D];
	double u, currMax=0, currVal;

	for(int i=n; i<n+samplesToTake; i++){
		for(int d=0; d<D; d++){
			u = (MAX_COORD - MIN_COORD)*gsl_rng_uniform(rSamples) + MIN_COORD;
			gsl_matrix_set(H->X,i,d,u);
			tmpGenotype[d] = u;
		}
		currVal = eval_movpeaks(tmpGenotype);
#ifdef LOG_ERRORS_2_FILE
		if((pErrorFile = fopen(szFileName,"a"))==NULL){
				printf("Unable to open file for appending\n");
				return;
		}
		fprintf(pErrorFile,"%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d\n",i,get_offline_error(),get_avg_error(),get_current_error(),get_current_maximum(),dummy_eval(tmpGenotype),get_right_peak(),get_maximum_peak());
		fclose(pErrorFile);
#endif
		if(currVal >= currMax){
			currMax = currVal;
			memcpy(bestNextGenotype,tmpGenotype,sizeof(double)*D);
		}
		gsl_matrix_set(H->y,i,0,currVal);
		H->maxVals[i] = currMax;
		gsl_vector_set(H->usePoint,i,USE_POINT);
//			printf("\n[%i]\tReal Max: %2.3f\tCurrent Error:%2.3f\tBest found:%2.3f\tAvg Err:%2.3f\n",i,global_max,get_current_error(),dummy_eval(bestNextGenotype),get_avg_error());
	}
	return;
}


/*************************************************
Take the latest state of the GP (X,Y,W,K)
and move it to H.prevGP
*************************************************/
void backupPrGP(history *H, int n, int inStrategy, int maxRecursionLevel){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

//No need to do this, read comment below
//	respSurf tmpRespSurf;
//	backup first what will be required to substract the mean from Y (even if it's only null values for the first time)		
//	memcpy(&tmpRespSurf,H->prevGP,sizeof(respSurf));


	//Check if this is (not) the first time
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

/* Never free!!!
	if(H->prevGP!=NULL){
		//is this enough or do we need to free each element inside prevGP?????	
		free(H->prevGP);
		pdebug(__FILE__,__LINE__,__FUNCTION__,"H->prevGP is now free!!!");
	}
*/




	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	//Count how many usefull datapoints we have for this ending epoch
	int inDummy=0;
	gsl_matrix  *vDummy=NULL;
	int startN = (H->currPeriod-1) * gPeakChangeFreq;
	int endN = (n % gPeakChangeFreq) + startN + 1;	//We want to consider up to the most recent point (included)
	int inCount = getViewCount(H, startN, endN, &inDummy);
//	printf("n=%d\tstartN=%d\tendN=%d\tinCount=%d\tinDummy=%d\n",n,startN,endN,inCount,inDummy);
	if(inDummy)
		vDummy = gsl_matrix_calloc(inDummy,2*D);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");


		//Allocate enough memory to hold the required data to define the response surface
		respSurf *tmp;
		tmp = (respSurf*) malloc(sizeof(respSurf));
		tmp->epochId = H->currPeriod - 1;
		tmp->X = gsl_matrix_calloc(inCount,D);
		tmp->y = gsl_matrix_calloc(inCount,2);	//2-D for compatibility with getView, although we don't need the noise
		tmp->w = (kernelParams*) malloc(sizeof(kernelParams));
		tmp->invKY = gsl_matrix_calloc(inCount,1);
		pdebug(__FILE__,__LINE__,__FUNCTION__,"");

		//Make the switch!
		tmp->meanPr = H->prevGP;	
		H->prevGP = tmp;

		//Depending on the desired recursion level, make the r-th level null to stop it.
		tmp = H->prevGP;
		int rLevel = 0;
//		printf("tmp=%p\ttmp->X=%p\ttmp->meanPr=%p\trLevel=%d\n",tmp,tmp->X,tmp->meanPr,rLevel);
		while(tmp->X!=NULL && rLevel<maxRecursionLevel){
			tmp = tmp->meanPr;
			rLevel++;
//			printf("tmp=%p\ttmp->X=%p\ttmp->meanPr=%p\trLevel=%d\n",tmp,tmp->X,tmp->meanPr,rLevel);
		}

		if(tmp->meanPr !=NULL){
			tmp->meanPr->X = NULL;			//just maxRecursionLevel Levels
			tmp->meanPr->w = NULL;
			tmp->meanPr->constScalarMean = 0;	//just maxRecursionLevel Levels
		}
//		printf("Outside:\ttmp=%p\ttmp->X=%p\ttmp->meanPr=%p\trLevel=%d\n",tmp,tmp->X,tmp->meanPr,rLevel);
//		getchar();

/*
		if(H->prevGP->meanPr->meanPr !=NULL){
			H->prevGP->meanPr->X = NULL;	//just 1 level
			H->prevGP->meanPr->w = NULL;
			H->prevGP->meanPr->constScalarMean = 0;	//just 1 level
		}
*/

	getView(H, inDummy, startN, endN, H->prevGP->X, H->prevGP->y, inStrategy, vDummy);

//No need to back up y with mean removed. Otherwise, it would be necessary to remember as well what values were removed (which can only be done by storing the whole respSurf model).
/*
	gsl_vector *vPrM=NULL;
	vPrM = gsl_vector_calloc(inCount);
	vdGetPrevGP(vPrM,&tmpRespSurf, H->prevGP->X, H->prevGP->y);

	vdRemoveMean(vPrM, H->prevGP->y);
	gsl_vector_free(vPrM);
*/
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	if(inDummy)
		gsl_matrix_free(vDummy);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	memcpy(H->prevGP->w,&H->w[endN-1],sizeof(kernelParams));	//endN points to an element of H->w[] for which maxKernelMLE was never run. Ideally, the parameters should be 										recalculated with all the available samples (but after so many, the MLE parameters shouldn't change much, so reuse 										the previous ones)
/*
	printf("H->%p\n",H);
	printf("H->prevGP->%p\n",H->prevGP);
	printf("H->prevGP->w%p\n",H->prevGP->w);

	printf("H->prevGP->w%p\n",H->prevGP->w);

	printf("H->w[endN]->s_f=%f\n",H->w[endN].s_f);
	printf("H->w[endN]->s_noise=%f\n",H->w[endN].s_noise);
	printf("H->w[endN]->lengthScale[0]=%f\n",H->w[endN].lengthScale[0]);
	printf("H->w[endN]->lengthScale[1]=%f\n",H->w[endN].lengthScale[1]);

	printf("H->prevGP->w->lengthScale[0]=%f\n",H->prevGP->w->lengthScale[0]);
	printf("H->prevGP->w->lengthScale[0]=%f\n",H->prevGP->w->lengthScale[1]);
	printf("H->prevGP->w->s_f=%f\n",H->prevGP->w->s_f);
	printf("H->prevGP->w->s_noise=%f\n",H->prevGP->w->s_noise);

	getchar();
*/

	//Store the inverse of the Kernel times Y (invKY)


//	printf("inCount=%d\n",inCount);

	gsl_matrix *K = gsl_matrix_calloc (inCount, inCount);
	gsl_matrix *invK = gsl_matrix_calloc (inCount, inCount);
	/* Calculate main Kernel */
	double maxCov = pow( H->prevGP->w->s_f,2);
	double scale[D];
	for(int d=0; d<D; d++)
		scale[d] = 1/(2*pow( H->prevGP->w->lengthScale[d] ,2));

	for (int i = 0; i < inCount; i++){
		for (int j = 0; j < inCount; j++){
			double sum=0.0;
			for(int d=0; d<D; d++){
				sum += -scale[d] * pow(gsl_matrix_get(H->prevGP->X, i, d) - gsl_matrix_get(H->prevGP->X, j, d), 2);
			}
			gsl_matrix_set (K, i, j, maxCov* exp(sum) );
		}
	}

	/* Invert K */
	int sign;
	gsl_permutation * perm = gsl_permutation_alloc (inCount);
	gsl_linalg_LU_decomp (K, perm, &sign);

	gsl_linalg_LU_det (K, sign);
	gsl_linalg_LU_invert (K, perm, invK);
	gsl_permutation_free(perm);

	/* Calculate K^-1 * Y **/
	for(int i=0; i<inCount; i++){
		double colTmp = 0.0;
		for(int j=0; j<inCount; j++)
			colTmp += gsl_matrix_get(invK,i,j) * gsl_matrix_get(H->prevGP->y,j,0);
		gsl_matrix_set (H->prevGP->invKY, i, 0, colTmp);
	}

//	printf("Substracting mean, invKY:\n");
//	printMatrix(H->prevGP->invKY);

	gsl_matrix_free(K);
	gsl_matrix_free(invK);

}


void evolveMovPeaks(gsl_rng *rSamples, double vLen, double hS, int inPeakNum){
#ifdef EVOLVE_MOV_PEAKS
/*
	//Check normality of double movnrand() 
	FILE * fWevol;
	char szFileWevol[200];
	sprintf(szFileWevol,"movNRand.dat");
	if((fWevol = fopen(szFileWevol,"a"))==NULL){
		printf("Unable to open file for writing\n");
		return;
	}
	double u;	
	for(int i=0; i<100000; i++){
		u = movnrand();
		fprintf(fWevol,"%f\n",u);	
	}
	fclose(fWevol);
	return;
//*/		

///*
	//Analyse how the global maxima changes from peak to peak
	double u;
	double tmpGenotype[D];	
	double sumW=0;

	int inPrevPeak = get_maximum_peak();
	change_peaks();
	int inCurrentPeak = get_maximum_peak();
	double pcStay=0;
	double pcPrev=0;
	double pcOther=0;

	for(int iMov=0; iMov<EVOLVE_MOV_PEAKS; iMov++){
		for(int d=0; d<D; d++){
			u = (MAX_COORD - MIN_COORD)*gsl_rng_uniform(rSamples) + MIN_COORD;
			tmpGenotype[d] = u;
			sumW += get_w(d);
		}
		eval_movpeaks(tmpGenotype);
		change_peaks();
//		printf("Before\tinCurrentPeak=%d\tinPrevPeak=%d\tpcStay=%f\tpcPrev=%f\tpcOther=%f\n",inCurrentPeak,inPrevPeak,pcStay,pcPrev,pcOther);
//		getchar();
		if(inCurrentPeak==get_maximum_peak())
			pcStay = pcStay + 1;
		else{//Maxima shifted to other peak
			pcOther = pcOther + 1;
			if(inPrevPeak==get_maximum_peak()) //And this other peak is the one of same as in the epoch before the previous.
				pcPrev = pcPrev + 1;
		}
		
//		printf("After\tinCurrentPeak=%d\tinPrevPeak=%d\tpcStay=%f\tpcPrev=%f\tpcOther=%f\tget_max=%d\n",inCurrentPeak,inPrevPeak,pcStay,pcPrev,pcOther,get_maximum_peak());
//		getchar();

		inPrevPeak = inCurrentPeak;
		inCurrentPeak = get_maximum_peak();		 

	}
	sumW = sumW / (double)(D * EVOLVE_MOV_PEAKS);
	pcStay = pcStay / (double) EVOLVE_MOV_PEAKS;
	pcPrev = pcPrev / (double) EVOLVE_MOV_PEAKS;
	pcOther = pcOther / (double) EVOLVE_MOV_PEAKS;
	
	FILE * fWevol;
	char szFileWevol[200];
	sprintf(szFileWevol,"wEvol_%dd.dat",D);
	if((fWevol = fopen(szFileWevol,"a"))==NULL){
		printf("Unable to open file for writing\n");
		return;
	}
	fprintf(fWevol,"%1.1f\t%1.1f\t%1.4f\t%1.4f\t%1.4f\t%1.4f\t%d\n",vLen,hS,pcStay,pcPrev,pcOther,sumW,inPeakNum);
	fclose(fWevol);
	return;
//*/
/*
	//Get a uniform random sample. 100 points per epoch, for 100 epochs.
	double sum=0; 
	double u;
	double tmpGenotype[D];
	FILE * fWevol;
	char szFileWevol[200];
	sprintf(szFileWevol,"wEvol_%dd.dat",D);
	if((fWevol = fopen(szFileWevol,"a"))==NULL){
		printf("Unable to open file for writing\n");
		return -1;
	}
	for(int iMov=0; iMov<EVOLVE_MOV_PEAKS; iMov++){
		for(int iRSample=0; iRSample<10000; iRSample++){
			for(int d=0; d<D; d++){
				u = (MAX_COORD - MIN_COORD)*gsl_rng_uniform(rSamples) + MIN_COORD;
				tmpGenotype[d] = u;
			}
			sum += eval_movpeaks(tmpGenotype);	
		}
		fprintf(fWevol,"%f\t",get_w(0));
		change_peaks();	
	}
	fprintf(fWevol,"\n");
	fclose(fWevol);

	//Write the output to a file
	FILE * pMeanPrior;
	char szFileNameMP[200];
	sprintf(szFileNameMP,"gpMeanPrior_%dd.dat",D);
	if((pMeanPrior = fopen(szFileNameMP,"a"))==NULL){
		printf("Unable to open file for writing\n");
		return -1;
	}
	printf("mean accross %d epochs with %d samples each = %f\n",EVOLVE_MOV_PEAKS,100,sum/(EVOLVE_MOV_PEAKS*10000));
	fprintf(pMeanPrior,"%f\n",sum/(EVOLVE_MOV_PEAKS*10000));
	fclose(pMeanPrior);
	return 0;
*/
#endif
}













