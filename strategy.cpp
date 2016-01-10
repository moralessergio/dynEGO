/* Tracking global optima in 
* dynamic environments with efficient 
* global optimization --- 2010-2014 
*/

/* strategy.cpp
 * Copyright (C) 2014 Sergio Morales.
 * This is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License.
 */
#include "main.h"

bool avoidBubbleCloseSample(gsl_matrix *X, double *bestGenotype, gsl_rng *r);
void avoidDuplicateSample(gsl_matrix *X, double *bestGenotype, gsl_rng *r);
void retainBestInBubble(history *H, int n);
int selectBestModel(gsl_matrix *X, gsl_matrix *y, kernelParams *W, float constParam, int inActualD, int inStrategy, int *inBestModel);
int inDivergeCount;
/************************************************************************************************************************
history *H:		History structure: contains all the previous samples.						*
gsl_rng *r:		Random stream pointer.										*
int n:			Current sample number to be taken.								*
double *bestNextGenotype: Used to get the previous best place to sample and to return the next best sample.		*
int inStrategy:	How to build the response surface (GP) [ST_RESET|ST_IGNORE|ST_INCREASE_NOISE|ST_TEMPORAL_DISCOUNT]	*
	ST_RESET:		Just start over looking for a new optimal (from cold).					*
	ST_IGNORE:		Don't do anything. Ignore the change and use all the available information.		*
	ST_INCREASE_NOISE:	Use old information in a noisy way (s_n increasing for old samples).			*
	ST_TEMPORAL_DISCOUNT:	Use old information by taking into account the difference in time (how old samples are)	* 
				inside the kernel.									*
double kSigma:	Specifies how to use the response surface once it is available. [kSigma=0->Mean|kSigma>0->how many std	*
		away from mean|kSigma=-1->EGO(EXPECTED improvement)|kSigma=-2->EGO_PROB(probability of improvement)]	*
************************************************************************************************************************/
bool getNextSample(history *H, gsl_rng *r, int n, double *bestNextGenotype, int inStrategy, double kSigma){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	int startN;
	int endN;
	if(inStrategy == ST_RESET || inStrategy == ST_RESET_STAR){
		startN = H->currPeriod * gPeakChangeFreq;
		endN = n % gPeakChangeFreq + startN;
	}
	else if(inStrategy==ST_PREV_SURF_MEAN_PRIOR || inStrategy==ST_CONST_MEAN_PRIOR){
		/* Take datapoins from current epoch only */
		startN = H->currPeriod * gPeakChangeFreq;
		endN = n % gPeakChangeFreq + startN;

		//No need to take any other points, since we already have 
		//saved the last K matrix and the corresponding parameters
		//this is possible because we are assuming that with enough samples
		//there is no need to include the mean prior to make predictions.
		
	}
	else{
		startN = max(0,H->currPeriod * gPeakChangeFreq - (EPOCHS_IN_MEM-1)*gPeakChangeFreq);	//Use only the last EPOCHS_IN_MEM epochs as historic data.
		//startN = 0;
		endN = n;
	}
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
//	printf("startN = %d\tendN = %d\nn=%d\tnMOD25=%d\n",startN,endN,n,(n+1)%25);
	/* Get a matrix view of the current state */
	gsl_matrix *vX, *vy;
	gsl_matrix  *vTabu=NULL;
	int inTabuCount=0;
	int inCount = getViewCount(H, startN, endN, &inTabuCount);
	int inActualD;
	if(inStrategy == ST_TEMPORAL_DISCOUNT && H->constParam == LEARN_TIMESCALE)
		inActualD = D+1;		//To hold the time dimension
	else
		inActualD = D;			//No time dimension required

	vX = gsl_matrix_calloc(inCount,inActualD);
	vy = gsl_matrix_calloc(inCount,2);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	if(inTabuCount){
		vTabu = gsl_matrix_calloc(inTabuCount,2*D);
	}

	getView(H, n, startN, endN, vX, vy, inStrategy, vTabu);

/*//	if((n+1)%25==0){
	if(n>25){
		printf("**********\nn=%d\n",n);
		printXYMatrices("XY matrices original", vX, vy, 0, inCount);
		printVector(H->usePoint, startN, endN);
		getchar();
	}
//*/
	gsl_vector *vPrM=NULL;
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
//	if(inStrategy==ST_PREV_SURF_MEAN_PRIOR || inStrategy==ST_CONST_MEAN_PRIOR){
		pdebug(__FILE__,__LINE__,__FUNCTION__,"");
		/* Get the (previous epoch) mean prior to be removed from each sample */		
		vPrM = gsl_vector_calloc(vy->size1);	//values initialized to zero!		
//Never learn the prior except for the ST_CONST_MEAN_PRIOR
		vdGetPrevGP(vPrM,H->prevGP, vX, vy, H->prevGP->constScalarMean, H->inInitSamples, inStrategy);

/*		if(n>49){
			printf("**********\nn=%d\n",n);
			printXYMatrices("XY matrices original", vX, vy, 0, inCount);
			printf("\nvPrM:\n");
			printVector(vPrM, 0, vPrM->size);
		}
*/

		/* Remove the prior mean */
		vdRemoveMean(vPrM, vy);
//	}
	gsl_vector_free(vPrM);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"vPrM freed");


	/* Make GP regression to estimate the function (ie: Optimize parameters) */
	kernelParams W[LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD];
	memset(W,0x00,sizeof(kernelParams)*LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD);


	//Big mistake, don't initialize at the previous starting point. If there is definitely not a way of finding a new best, just take a random sample (but only after exploring all the different models)
	if(n>0){	//Initialize the starting point at the previous MLE (besides, if no new MLE parameters found, then use the previous ones)
		memcpy(&W[0],&H->w[n-1],sizeof(kernelParams));
		W[0].maxEGO = -1;	//Only initialize MLE parameters, not historic values
		W[0].detK = 0;
		W[0].logLike = 0;
	}

//	printf("Number of datapoints %d\t(n Mod gPeakChangeFreq)=%d\n",n,n%gPeakChangeFreq);
	int maxKernelStatus = GSL_FAILURE;
	if( (inStrategy==ST_PREV_SURF_MEAN_PRIOR || inStrategy==ST_RESET_STAR) && (n%gPeakChangeFreq == 1) ){
		//Do not run the Likelihood maximization (maxKernelMLE), just copy the previous one.
		memcpy(&W[0],&H->w[n-2],sizeof(kernelParams));
		//check what to do when only 1 data point is available (copy previous parameters W and do not enter into maxKernelMLE. Only perform the maximization (maxGPReg).
//		printf("********Not running logLikeMaximization****\n");

	}
	else{
		//Sometimes, the allocated number of iterations is not enough for the Kernel maximizer to find a feasible solution. To avoid wasting too much computational effort
		//by increasing the iterations for this always, just do it whenever it is required. Keep looking until a solution is found...
		int inTryHarder = 0;
//		printf("********Running logLikeMaximization\n");
		while(maxKernelStatus != GSL_SUCCESS && inTryHarder < 10){
			maxKernelStatus = maxKernelMLE(vX, vy, W, r, inStrategy, H->constParam);
//			printf("Max kernel status: %d\tinTryHarder=%d\n",maxKernelStatus,inTryHarder);
			inTryHarder++;
		}
		if(maxKernelStatus != GSL_SUCCESS){//if still not finding good parameters, use the previous ones...provided they're not zero (which happens for the first point of each epoch)
			if(H->w[n-1].logLike > 0)	//If previous was not the first point of this epoch
				memcpy(&W[0],&H->w[n-1],sizeof(kernelParams));	
			else				//else, take the last one from last epoch (2 points ago)
				memcpy(&W[0],&H->w[n-2],sizeof(kernelParams));
		}

	}


	int inOrderedModels[LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD];
	bool bExploring = false;

	selectBestModel(vX, vy, W, H->constParam, inActualD, inStrategy, inOrderedModels); //Only from a list of LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS ~ 20**inActualD


	int kModel=0;
	memcpy(&H->w[n],&W[inOrderedModels[kModel]],sizeof(kernelParams));
	bExploring = maxGPReg(vX, vy, &H->w[n], bestNextGenotype,kSigma,r,H->maxVals[n-1], inStrategy, H->constParam, vTabu, H->prevGP);
		

	for(int d=0; d<D; d++){	
		gsl_matrix_set(H->X,n,d,bestNextGenotype[d]);
//		printf("bestNextGenotype[%d]=%f\t",d,bestNextGenotype[d]);
	}
//	printf("\n");

	/* Get a new sample: Evaluate the function at the next best coordinates */
	double currVal = eval_movpeaks(bestNextGenotype);
	gsl_matrix_set(H->y,n,0,currVal);


	if(currVal > H->maxVals[n-1])
		H->maxVals[n] = currVal;
	else
		H->maxVals[n] = H->maxVals[n-1];

#ifdef NO_EXPLOIT_AFTER	//Avoids just exploiting. Exploitation in this case is just an exploration using all the current samples except those too close to each others.
	gsl_vector_set(H->usePoint,n,USE_POINT);
//printf("before retainBestInBubble\n");
	retainBestInBubble(H, n);
//printf("after retainBestInBubble\n");
	bExploring = true;	
#endif

	gsl_matrix_free(vX);
	gsl_matrix_free(vy);
	gsl_matrix_free(vTabu);

	return bExploring;
}

/* Returns the number of points to be used. */
int getViewCount(history *H, int startN, int endN, int *tabuCount){
	int inCount=0;
	*tabuCount = 0;
	for(int i=startN; i<endN; i++){
		if(gsl_vector_get(H->usePoint,i)==USE_POINT){
			inCount++;
		}
		else if(gsl_vector_get(H->usePoint,i) > USE_POINT){
			(*tabuCount)++;
		}
	}
//	printf("tabuCount==inDummy==%d\n",*tabuCount);
//	getchar();
	return inCount;
}

/*Returns the number of points to be used. The points are stored in vX and the responses in vY*/
int getView(history *H, int n, int startN, int endN, gsl_matrix *vX, gsl_matrix *vy, int inStrategy, gsl_matrix *vTabu){
	int j=0;
	int inTabuCount=0;
	for(int i=startN; i<endN; i++){
		if(gsl_vector_get(H->usePoint,i)==USE_POINT){
			for(int d=0; d<D; d++){
				gsl_matrix_set(vX,j,d,gsl_matrix_get(H->X,i,d));
			}
			gsl_matrix_set(vy,j,0,gsl_matrix_get(H->y,i,0));
			if(inStrategy == ST_INCREASE_NOISE){
				gsl_matrix_set(vy,j,1,H->currPeriod - floor(i/gPeakChangeFreq));
//				printf("ST_INCREASE_NOISE [%d]\n",inStrategy);
			}
			else if(inStrategy == ST_TEMPORAL_DISCOUNT){
				if(H->constParam == LEARN_TIMESCALE){
					gsl_matrix_set(vX,j,D,H->currPeriod - floor(i/gPeakChangeFreq));
					gsl_matrix_set(vy,j,1,0);
				}
				else{
					gsl_matrix_set(vy,j,1,H->currPeriod - floor(i/gPeakChangeFreq));
				}
//				printf("ST_TEMPORAL_DISCOUNT [%d]\n",inStrategy);
			}
			else{
				gsl_matrix_set(vy,j,1,0);	//No need to increase the epoch
//				printf("else [%d]\n",inStrategy);
			}
			j++;
		}
		else if(gsl_vector_get(H->usePoint,i) > USE_POINT && vTabu!=NULL){
			//Define a tabu region by providing an interval, for each dimension individually. Use one vector to concatenate the to vectors defining each point. Coords of second point are in the 					same order as for the first point but shifted by D positions
			//Keep the low coordinate in the lower part of the array and the high coordinate in the higher part of the array.
			double xIni, xEnd;
			int inEndTabuRegion = gsl_vector_get(H->usePoint, i);	//Points to the index (in History->X) of the point together with wich this points defines a tabu region
			for(int d=0; d<D; d++){
				xIni = gsl_matrix_get(H->X,i,d);
				xEnd = gsl_matrix_get(H->X,inEndTabuRegion,d);
				//Floor and ceil correspondigly in order to avoid very close sampling. 
				if(xIni <= xEnd){
					xIni = floor(xIni/EPSILON_BUBBLE_RADIOUS)*EPSILON_BUBBLE_RADIOUS;
					xEnd = ceil(xEnd/EPSILON_BUBBLE_RADIOUS)*EPSILON_BUBBLE_RADIOUS;
					gsl_matrix_set(vTabu,inTabuCount,d,xIni);
					gsl_matrix_set(vTabu,inTabuCount,d+D,xEnd);
				}
				else{
					xIni = ceil(xIni/EPSILON_BUBBLE_RADIOUS)*EPSILON_BUBBLE_RADIOUS;
					xEnd = floor(xEnd/EPSILON_BUBBLE_RADIOUS)*EPSILON_BUBBLE_RADIOUS;
					gsl_matrix_set(vTabu,inTabuCount,d,xEnd);
					gsl_matrix_set(vTabu,inTabuCount,d+D,xIni);
				}
				
			}
			inTabuCount++;
		}
	}
//	printf("H->currPeriod[%d] - floor(n[%d]/gPeakChangeFreq[%d],[%d])[%f] = %f\n",H->currPeriod,n,gPeakChangeFreq,n/gPeakChangeFreq,floor(n/gPeakChangeFreq),H->currPeriod - floor(n/gPeakChangeFreq));
	return j;
}

void avoidDuplicateSample(gsl_matrix *X, double *bestGenotype, gsl_rng *r){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	int n = X->size1;

	for(int i=n-1; i>=0; i--){
		int inCount = 0;
		for(int d=0; d<D; d++){
			if(gsl_matrix_get(X,i,d) == bestGenotype[d])
				inCount++;
		}
		if(inCount==D){
			double u;
			for(int d=0; d<D; d++){
				u = (gsl_rng_uniform(r) - 0.5)*EPSILON_SAMPLING_SHIFT;	//Introduce a +/-epsilon small random shift in each dimension
				bestGenotype[d] += u;
			}
			return;
		}
	}
}

bool avoidBubbleCloseSample(gsl_matrix *X, double *bestGenotype, gsl_rng *r){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	int n = X->size1;
	for(int i=n-1; i>=0; i--){
		//Calculate euclidian distance
		double dist = 0;
		for(int d=0; d<D; d++)
			 dist += pow(gsl_matrix_get(X,i,d) - bestGenotype[d],2);

		if(sqrt(dist) < EPSILON_BUBBLE_RADIOUS){
			double u;
			for(int d=0; d<D; d++){
				u = (gsl_rng_uniform(r) - 0.5)*2*EPSILON_SAMPLING_SHIFT;	//Introduce a +/-epsilon small random shift in each dimension
				bestGenotype[d] += u;
			}
			pdebug(__FILE__,__LINE__,__FUNCTION__,"Sampling too close => Increasing exploration counter.");
			return false;
		}
	}
	return true;

}

void retainBestInBubble(history *H, int n){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	//Consider only the current period.
		//int startN = H->currPeriod * gPeakChangeFreq;
		//int endN = n % gPeakChangeFreq + startN;
		//int inActualD = H->X->size2;
	//Or consider all the historic data
		int startN = 0;
		int endN = n;
		int iPeriod = 0;
	//In fact, it should consider only points which are candidates, i.e.: Points from the current epoch (0) until the first sample of (EPOCHS_IN_MEM - 1) back.
/*
if(n>13){
	printf("endN=%d\tn=%d\n",endN,n);
	printXYMatrices("X\tY:", H->X, H->y, 0, endN+2);
	printf("usePoint:\n");
	printVector(H->usePoint,0,endN+2);
//	getchar();
}
//*/
	for(int i=startN; i<endN; i++){
		iPeriod = H->currPeriod - floor(i/gPeakChangeFreq);
		if(gsl_vector_get(H->usePoint,i) == USE_POINT){
			double dist = 0;
			for(int d=0; d<D; d++){//Do not consider time dimension. We are looking for same points in space, and will always keep the newest.
				dist += pow(gsl_matrix_get(H->X, i, d) - gsl_matrix_get(H->X, n, d),2);
			}
			if(sqrt(dist) < (D * EPSILON_BUBBLE_RADIOUS)){
//				printf("Removing!!!!*****************(i=%d \t n=%d)\n",i,n);
//				getchar();
				if(iPeriod > 0){
//					printf("Disabling point %d: Close sample from new epoch found (%d).\n",i,n);
//					getchar();
					gsl_vector_set(H->usePoint, i, DISCARD_POINT);	//Keep only the point belonging to the newest epoch, regardless of the value.
				}
				else if(gsl_matrix_get(H->y, i, 0) <= gsl_matrix_get(H->y, n, 0)){//Strictly better, otherwise we would accept exactly the same configuration of points.
//					printf("Disabling point %d: Strictly better (close) sample from the same (current) epoch found (%d).\n",i,n);
//					getchar();
					gsl_vector_set(H->usePoint, i, DISCARD_POINT);	//Don't use this point anymore (since a new one with better solution was found in a very close region for the same epoch)
				}
				else{
//					getchar();
					//Getting stuck??? Discarding a new point (under a fair basis, since it is worse than another already known), but leads the sampling history unchanged, hence damed to 							repeat history! Instead, should make sure we learn from this, and do not sample in this region anymore => Tabu the region.
					pdebug(__FILE__,__LINE__,__FUNCTION__,"Getting stuck??? Data is not changing when discarding a new sample.");
//					gsl_vector_set(H->usePoint, n, 0);	//Don't use the new point (too close and worse)
//					printf("Setting H->usePoint[%d]=%d\n",n,i);
//					getchar();
					gsl_vector_set(H->usePoint, n, i);	//Don't use the new point (too close and worse). Instead, define a tabu region (hyper-cube) from the new point (n), to the 												"close point" i in which the expected improvement shall be zero.
				}
			}
		}
	}
}

/***********************************
Returns the index of the best k-th model
***********************************/
int selectBestModel(gsl_matrix *X, gsl_matrix *y, kernelParams *W, float constParam, int inActualD, int inStrategy, int *inBestModel){

//	int inBestModel = -1;
//	int inBestModel[LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS];
	kernelParams tmpW;
//	double minRMSE = DBL_MAX;
	double vRMSE[LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD];
	double tmpRMSE;

	for(int i=0; i<LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD; i++){
		inBestModel[i]=-1;
		vRMSE[i] = DBL_MAX;
	}
//	printf("RMSE of models in disorder:\n");
	int i=0;
	while(W[i].logLike != 0 && i<LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD){	

		memcpy(&tmpW,&W[i],sizeof(kernelParams));
		tmpRMSE = getRMSE(X, y, &W[i], constParam, inStrategy);
//		printf("[%d]\t%E\n",i,tmpRMSE);

		int j=0;
		//while(tmpRMSE > vRMSE[j] && vRMSE[j] < DBL_MAX){
		while(j<=i){//Find where to insert
			if(tmpRMSE < vRMSE[j])
				break;
			j++;
		}

		//Insert here
		for(int k=i; k>j; k--){
			vRMSE[k] = vRMSE[k-1];
			inBestModel[k] = inBestModel[k-1];
		}
		vRMSE[j] = tmpRMSE;
		inBestModel[j] = i;

/*		else if(minRMSE == tmpRMSE){//Un-tie by best logLike ??????????
			if(inBestModel >=0 && W[i].logLike > W[inBestModel].logLike){//Check inBestModel is a valid index (ie: at least pointing to a valid model)
				inBestModel = i;
//				printf("*****New best found by untie = %d\n",inBestModel);
			}
		}
*/
		i++;
	}
/*	printf("Models in increasing order according to RMSE...:\n");
	for(int kOrder=0;kOrder<LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD; kOrder++){
		printf("%d   ",inBestModel[kOrder]);
	}
*/
//	printf("\nminRMSE found = %2.20f \n", minRMSE);
//	return inBestModel;
	return i;
}

void getPreviousBest(double *bestNextGenotype, history *H, int n, int gPeakChangeFreq){
	
	int startN = (H->currPeriod -1) * gPeakChangeFreq;
	int endN = n-1;
	int iMax = startN;
	double maxPrev = gsl_matrix_get(H->y, startN, 0);

	for(int i=startN+1; i<=endN; i++){
		if(gsl_matrix_get(H->y, i, 0) > maxPrev){
			maxPrev = gsl_matrix_get(H->y, i, 0);
			iMax = i;
		}
	}

//	printf("Looking for the best of previous epoch. Starting at n=%d, to n=%d.\niMax=%d\t",startN, endN, iMax);
	for(int d=0; d<D; d++){
		bestNextGenotype[d] = gsl_matrix_get(H->X, iMax, d);
//		printf("%f\t",bestNextGenotype[d]);
	}

	/* Once it has been found, discard the old one, because we are going to get an updated value for that same location */
	gsl_vector_set(H->usePoint, iMax, DISCARD_POINT);
	//...Perhaps not the best idea for TasD+1??? check results and come back to modify this.

//	printf("\n With value: yMax=%f\n",gsl_matrix_get(H->y, iMax, 0));

}


/*************************************************************************
Receives :
	All the necessary infomration to recreate the response surface 
	as it was known at the end of the previous epoch. This is the mean 
	prior (default value to be taken where no new observations are 
	available)
	prevGP->X,Y,W,K
	A list of points where the values will be tested, which correspond
	to the observations available made at the current epoch.
	vy

Returns a list of values to be removed to the response variable (Y) in:
	vPrM (same size as vy).
**************************************************************************/
void vdGetPrevGP(gsl_vector *vPrM, respSurf *prevGP, gsl_matrix *vX, gsl_matrix *vy, double constParam, int inInitSamples, int inStrategy){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	//Check if H->prevGP is null. This would mean that this is the first epoch and only the mean of Y shall be 
	//returned for each test data point.
	int inSampleSize = vy->size1;
	double meanY=0;
	pdebug(__FILE__,__LINE__,__FUNCTION__,"here?");

	if(prevGP->w==NULL){//First time of PSMP or any other strategy: set everything to the mean of first inInitSamples elements of Y
		if(inStrategy==ST_CONST_MEAN_PRIOR && constParam == LEARN_TIMESCALE){//Learn constant mean from initial random samples.
			for(int i=0; i<inInitSamples;i++)
				meanY += gsl_matrix_get(vy,i,0);
			meanY = meanY/inInitSamples;
		}
		else
			meanY = constParam;

		pdebug(__FILE__,__LINE__,__FUNCTION__,"NULL");
		gsl_vector_set_all(vPrM,meanY);
		prevGP->constScalarMean = meanY;
//		printVector(vPrM, 0, vPrM->size);
		return;
	}

	pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");
	/* Recursive call */
	vdGetPrevGP(vPrM, prevGP->meanPr, vX, vy, constParam, inInitSamples, inStrategy);

	int inOldSampleSize = prevGP->y->size1;
	//Fill vPrM[inSampleSize] with the predictions made at each point of vX

	/* Calculate the kernel from each test point (current genotype) to each sampled point */
	double dblSample[D];
	double maxCov = pow( prevGP->w->s_f ,2);
	double scale[D];
	for(int d=0; d<D; d++)
		scale[d] = 1/(2*pow( prevGP->w->lengthScale[d] ,2));




	for(int i=0; i<inSampleSize; i++){//For each point...(take one by one from vX and vY)

		//Copy this sample to dblSample
		for(int d=0; d<D; d++)
			dblSample[d] = gsl_matrix_get(vX,i,d);

		pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");
		gsl_vector *k_new = gsl_vector_alloc (inOldSampleSize);
		gsl_vector *invK_kNew = gsl_vector_alloc (inOldSampleSize);
		pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");
		/* Calculate the kernel from this test point (current genotype) to each sampled point from the old epoch (i.e. predict the value according to previous epoch model) */
		for(int j=0; j<inOldSampleSize; j++){
			double sum = 0.0;

			for(int d=0; d<D; d++){
				sum +=  -scale[d]*pow(gsl_matrix_get(prevGP->X, j, d) - dblSample[d], 2);
			}
			gsl_vector_set(k_new, j, maxCov * exp(sum) );

		}


	pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");
//	printf("invKY used:");
//	printMatrix(prevGP->invKY);

		/* Calculate mean(i) */
		double YinvKY = 0.0;
		for(int j=0; j<inOldSampleSize; j++)
		    YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(prevGP->invKY,j,0);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");	
		gsl_vector_set(vPrM,i,YinvKY + gsl_vector_get(vPrM,i));	//set the mean calculated at this point + the mean prior from the previous model.
		gsl_vector_free(k_new);
		gsl_vector_free(invK_kNew);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"or here?");
	}
}

/* Remove the prior mean vPrM from vy, and return the result in vy */
void vdRemoveMean(gsl_vector *vPrM, gsl_matrix *vy){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	for(unsigned int i=0;i<vy->size1;i++)
		gsl_matrix_set(vy,i,0,gsl_matrix_get(vy,i,0)-gsl_vector_get(vPrM,i));
}

