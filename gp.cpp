/* Tracking global optima in 
* dynamic environments with efficient 
* global optimization --- 2010-2014 
*/

/* gp.cpp
 * Copyright (C) 2014 Sergio Morales.
 * This is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License.
 */
#include "main.h"
void plotEGO(void *Data);
void plotEGO2D(void *Data);
void plotEGOAtSamples2D(void *Data);
void getBoxes(gsl_matrix *X, gsl_matrix *startP, gsl_matrix *steps, respSurf *prevGP, int N);
void crossProd(gsl_matrix *M, gsl_vector *v);
double priorMean(respSurf *prevGP, const gsl_vector *pGenotype);
double logLikelihood(const gsl_vector *pw, void *Data){
	/* Restore the data to the original form */
	gsl_matrix **Samples = (gsl_matrix **)Data;
	gsl_matrix *X = Samples[0];
	gsl_matrix *y = Samples[1];
	gsl_matrix *model = Samples[2];

	int inSampleSize = X->size1;
	int inActualD = X->size2;
	float inStrategy = gsl_matrix_get(model,0,0);
	float constParam = gsl_matrix_get(model,1,0);

	kernelParams w;
	w.s_f = gsl_vector_get (pw, 0);
	if(inStrategy == ST_INCREASE_NOISE)
		w.s_noise = constParam;	//Set the approriate level of measurement noise to be added
	else
		w.s_noise = 0;	//Set to 0 for deterministic functions

	for(int d=0; d<inActualD; d++)
		w.lengthScale[d] = gsl_vector_get (pw, d+1);

	//CAllocate the kernel matrix
	gsl_matrix *k = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *s_noise = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *invK = gsl_matrix_calloc (inSampleSize, inSampleSize);
	double detK;

	double maxCov = pow( w.s_f,2);
	double scale[inActualD];
	for(int d=0; d<inActualD; d++)
		scale[d] = 1/(2*pow( w.lengthScale[d] ,2));

	/* Calculate main Kernel */
	for (int i = 0; i < inSampleSize; i++){
		for (int j = 0; j < inSampleSize; j++){
			double sum = 0.0;
			for(int d=0; d<inActualD; d++){
				sum+= -scale[d]*pow(gsl_matrix_get(X, i, d) - gsl_matrix_get(X, j, d), 2);
			}
			//Should this be included here? Should not affect since it is a constant? Answer: Yes, improves a lot when included! (Learning the real environment where it will be tested perhaps)
			if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)		
				sum = exp( sum - constParam*pow(gsl_matrix_get(y, i, 1) - gsl_matrix_get(y, j, 1), 2) );
			else
				sum = exp( sum );

        		gsl_matrix_set (k, i, j, maxCov*sum );
		}
	}

	/* Replace the kronecker delta --which is 1 only for the diagonal-- by: */
	//gsl_matrix_set_identity (eye);
	for(int i=0; i<inSampleSize; i++){
		gsl_matrix_set(s_noise,i,i,gsl_matrix_get(y,i,1)*pow(w.s_noise,2)); //Time index <=> epoch: So recent measurements have time 0, old measurements have time > 0 (integer times for now).

	}
	//gsl_matrix_scale (eye, pow(w.s_noise,2));
	gsl_matrix_add (k, s_noise);

	/* Calculate log likelihood in several steps, using LU decompositions to avoid direct inversion */
	int sign;
	gsl_permutation * perm = gsl_permutation_alloc (inSampleSize);
	gsl_linalg_LU_decomp (k, perm, &sign);

	detK = gsl_linalg_LU_det (k, sign);
//	if(detK==0)
	if(detK < pow(MIN_TRUSTED_DET_K,5) ){
		gsl_matrix_free(k);
		gsl_matrix_free(s_noise);
		gsl_matrix_free(invK);
		gsl_permutation_free(perm);
		return DBL_MAX;
	}
	gsl_linalg_LU_invert (k, perm, invK);
	gsl_permutation_free(perm);

	gsl_vector *yTmp = gsl_vector_alloc (inSampleSize);
	for(int i=0; i<inSampleSize; i++){
		double colTmp = 0.0;
		for(int j=0; j<inSampleSize; j++)
			colTmp += gsl_matrix_get(y,j,0) * gsl_matrix_get(invK,i,j);
		gsl_vector_set (yTmp, i, colTmp);
	}

	double YkY = 0.0;
	for(int i=0; i<inSampleSize; i++)
		YkY += gsl_matrix_get(y,i,0) * gsl_vector_get(yTmp,i);

	double logLikelihood = (-0.5)*YkY-(0.5)*log(detK) - (inSampleSize/2.0)*log(2.0*PI);
	//Avoid getting infinity (division by 0)
	if(isinf(logLikelihood) || isnan(logLikelihood)){
		logLikelihood = -DBL_MAX;
	}
//	printf("inStrategy=%f\tconstPara=%f\tnoise=%f\tdetK=%f\tlogLike=%f\tinSampleSize=%d\n",inStrategy,constParam,w.s_noise,detK,logLikelihood,inSampleSize);
	gsl_matrix_free(k);
	gsl_matrix_free(s_noise);
	gsl_matrix_free(invK);
	gsl_vector_free(yTmp);

	return (-logLikelihood);
}

int maxKernelMLE(gsl_matrix *X, gsl_matrix *y, kernelParams *wOut,  gsl_rng *r, int inStrategy, float constParam)
{
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	gsl_matrix *model = gsl_matrix_alloc(2,1);
	gsl_matrix_set(model, 0,0,(float)inStrategy);
	gsl_matrix_set(model, 1,0,constParam);

	/* Encapsulate the arguments for the likelihood */
	gsl_matrix *Data[3] = {X,y,model};

	/* save original handler, turn off error handling temporarily */
	gsl_error_handler_t *old_handler = gsl_set_error_handler_off ();
	double u;
	double tmpMaxLikelihood=-DBL_MAX;
	int overallStatus = GSL_ERANGE;
	int iRobust=0;
	int inConverged=0;
	int inActualD = X->size2;


	while( iRobust++ < LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS * inActualD ){
		/* Instantiate and configure minimizer */
//		const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
		const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;//Consider using the random initialized version
		gsl_multimin_fminimizer *s = NULL;
		gsl_vector *ss, *w;
		gsl_multimin_function minex_func;

		/* Set initial step sizes to 1 */
		ss = gsl_vector_alloc (1 + inActualD);
		gsl_vector_set_all (ss, 1.0);

		/* Create a starting point */
		w = gsl_vector_alloc (1 + inActualD);	//s_f, D length scales (or D+1 if time is considered a dimension)

		if(iRobust==1){//For the first iteration don't start at random, but at the previously best known location
			//printf("***IN!\t\tiRobust=[%d]\n",iRobust);
			gsl_vector_set (w, 0, wOut[0].s_f);	//s_f
			for(int d=1; d<(1+inActualD); d++)
				gsl_vector_set (w, d, wOut[0].lengthScale[d]);	//lengthscale
		}
		else{
			u = MAX_LIKELIHOOD_PARAM_SEARCH*gsl_rng_uniform(r);
			gsl_vector_set (w, 0, u);	//s_f
			for(int d=1; d<(1+inActualD); d++){
				u = MAX_LIKELIHOOD_PARAM_SEARCH*gsl_rng_uniform(r);
				gsl_vector_set (w, d, u);	//lengthscale
			}
		}

		size_t iter = 0;
		int status;
		double size;

		/* Initialize method and iterate */
		minex_func.n = 1+inActualD;
		minex_func.f = logLikelihood;
		minex_func.params = Data;
		s = gsl_multimin_fminimizer_alloc (T, 1+inActualD);
		gsl_multimin_fminimizer_set (s, &minex_func, w, ss);

		do{
			iter++;
			status = gsl_multimin_fminimizer_iterate(s);

			if (status){
				char szMessage[200];
				sprintf(szMessage,"Status: %s.\tBreaking...\n", gsl_strerror(status));
				pdebug(__FILE__,__LINE__,__FUNCTION__,szMessage);
				printf("***Breaking***\t");
				break;
			}

			size = gsl_multimin_fminimizer_size (s);
			status = gsl_multimin_test_size (size, GSL_MINIMIZER_LIKELIHOOD_PRECISION);
#ifdef LOG_MIN
			if (status == GSL_SUCCESS)
			{
				printf ("converged to minimum at %2.5f\n",s->fval);


				char szMessage[200];
				char szTmp[50];
				sprintf(szMessage,"%5d %10.5e ", (int)iter, gsl_vector_get (s->x, 0));
				for(int d=1; d<(inActualD+1); d++){
					sprintf(szTmp,"%10.5e ",gsl_vector_get (s->x, d));
					strcat(szMessage,szTmp);
				}
				sprintf(szTmp," f() = %10.8f size = %.3f\n", s->fval, size);
				strcat(szMessage,szTmp);
				printf("%s",szMessage);
			}
#endif
		}while (status == GSL_CONTINUE && iter < (unsigned int) (LOGLIKE_MAXIMIZATION_MAX_ITERATIONS * inActualD));

//		if(status == GSL_SUCCESS && tmpMaxLikelihood < s->fval){//then keep new answer
//		printf("iRobust=%d just finished\n",iRobust);
		if(status == GSL_SUCCESS){//then keep new answer
//			printf("New best solution found at iRobust=[%d], Converged for the =%d time\n",iRobust,inConverged);
			tmpMaxLikelihood = s->fval;
			wOut[inConverged].s_f 		= gsl_vector_get (s->x, 0);
			if(inStrategy == ST_INCREASE_NOISE)
				wOut[inConverged].s_noise 	= constParam;
			else
				wOut[inConverged].s_noise 	= 0;
			wOut[inConverged].logLike		= s->fval;

			for(int d=1; d<(inActualD+1); d++){
				wOut[inConverged].lengthScale[d-1] 	= gsl_vector_get (s->x, d);
			}


			if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)
				wOut[inConverged].lengthScale[inActualD] 	= constParam;
			overallStatus = GSL_SUCCESS;	//at least one good solution found
			inConverged++;
		}
//		printf("Deallocationg w, ss, and s\n");
		gsl_vector_free(w);
		gsl_vector_free(ss);
		gsl_multimin_fminimizer_free (s);

	}//end while iRobust

	gsl_matrix_free(model);
	/* restore original handler */
	gsl_set_error_handler (old_handler);
	return overallStatus;
}

/*Receives:
	mu: expected mean according to GP regression
	sigma: standard deviation of the normal distrinution of our belief according to GP regression
	currMax: best known value so far
Returns: The expected improvement	*/
double egoImprovement(double mu, double sigma, double currMax, double kSigma){
//	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
#ifdef SLOW
	double resolution=EGO_RESOLUTION;
	double expImprovement_Int = 0;

	gsl_vector *z = gsl_vector_calloc(resolution);
	gsl_vector *I = gsl_vector_calloc(resolution);
	gsl_vector *probI = gsl_vector_calloc(resolution);
	gsl_vector *expectedI = gsl_vector_calloc(resolution);
	double dz = linSpace(MIN_HEIGHT, MAX_HEIGHT, resolution, z);
	//double dz = linSpace(0, mu + 2*sigma, resolution, z);

	for(unsigned int i=0; i<I->size; i++){
		gsl_vector_set(I,i, max(gsl_vector_get(z,i) - currMax,0));
		gsl_vector_set(probI,i, gsl_ran_gaussian_pdf(gsl_vector_get(z,i)-mu, sigma));
		gsl_vector_set(expectedI,i, gsl_vector_get(I,i)*gsl_vector_get(probI,i));
		expImprovement_Int += gsl_vector_get(expectedI,i);
	}
	gsl_vector_free(z);
	gsl_vector_free(I);
	gsl_vector_free(probI);
	gsl_vector_free(expectedI);
	expImprovement_Int = expImprovement_Int * dz;
	return -expImprovement_Int;
#else
	if(kSigma == EGO){	//EGO as described in the paper
/*
		double expImprovement_Int = 0;
		double dz = (MAX_HEIGHT - MIN_HEIGHT)/(EGO_RESOLUTION - 1);
		for(int i=0; i<EGO_RESOLUTION; i++)
			expImprovement_Int += max(i*dz - currMax,0)*gsl_ran_gaussian_pdf(i*dz-mu, sigma);
		expImprovement_Int = expImprovement_Int * dz;
		return -expImprovement_Int;
*/
		if(sigma <= 0.01 /*EPSILON_BUBBLE_RADIOUS*/){	//Too close to zero or negative value (due to numerical error)
//			printf("Returning EGO=0, since sigma = %f\n",sigma);
			//getchar();
			return 0;
		}
		else{
			double normalizedArgument = (mu - currMax) / sigma;
//			printf("currMax=%f\tEGO=%2.20f\n",currMax,-1*( (mu - currMax) * gsl_cdf_ugaussian_P(normalizedArgument) + sigma* gsl_ran_ugaussian_pdf(normalizedArgument)));
			double egoImp = -1*( (mu - currMax) * gsl_cdf_ugaussian_P(normalizedArgument) + sigma* gsl_ran_ugaussian_pdf(normalizedArgument));
			if(isinf(egoImp) || isnan(egoImp)){
//				printf("mu=%f\tcurrMax=%f\tnormalizedArgument=%f\tsigma=%20.20f\tgsl_cdf_ugaussian_P(normalizedArgument)=%f\tgsl_ran_ugaussian_pdf(normalizedArgument)=%f\n",mu, currMax, normalizedArgument, sigma, gsl_cdf_ugaussian_P(normalizedArgument), gsl_ran_ugaussian_pdf(normalizedArgument));
//				getchar();
			}
//			if(egoImp == -1)
//				printf("***Here it is!***\n");
			return egoImp;
//			return -1*( (mu - currMax) * gsl_cdf_ugaussian_P(normalizedArgument) + sigma* gsl_ran_ugaussian_pdf(normalizedArgument));
		}


	}
	else if(kSigma == EGO_PROB){//Robust as opposed to greedy?
		//probImprovement = errorFunction(mu, sigma, x>currMax)??? Then return this... Check exactly how to calculate the probability.
		double probImprovement = gsl_cdf_gaussian_Q(currMax - mu,sigma);
		return -probImprovement;
	}
	return -1;
#endif

//	printf("expImprovement_Int=%5.40f\n",expImprovement_Int);

}

bool inTabuRegion(const gsl_vector *pGenotype, gsl_matrix *Tabu){

	int n = Tabu->size1;
	int insideCount;
	
//	printf("TabuSize: %d\n",n);

	//Check each tabu region
	for(int i=0; i<n; i++){	//Change to whiles and use breaks to make it faster (exit as soon as it is outside the region in 1 of the dimensions)
		insideCount = 0;
		for(int d=0; d<D; d++){
			if( gsl_matrix_get(Tabu,i,d) <= gsl_vector_get(pGenotype,d) && gsl_vector_get(pGenotype,d) <= gsl_matrix_get(Tabu,i,d+D) ){
//			printf("Comparing: tabu[%d][%d] <= pGenotype(%f) && tabu[%d][%d] <= pGenotype(%f)\n",i,d,gsl_vector_get(pGenotype,d),i,d+D,gsl_vector_get(pGenotype,d));
//			printf("%f <=? %f \t&&\t %f <=%f",gsl_matrix_get(Tabu,i,d),gsl_vector_get(pGenotype,d),gsl_vector_get(pGenotype,d),gsl_matrix_get(Tabu,i,d+D));

				insideCount++;
			}
		}
//		printf("insideCount=%d\n",insideCount);
		if(insideCount == D)
			return true;
		
	}
	return false;

}

double gpRegFunction(const gsl_vector *pGenotype, void *Data){
//	pdebug(__FILE__,__LINE__,__FUNCTION__,"");


	/* Restore the data to the original form */
	gsl_matrix **Params = (gsl_matrix **)Data;
	gsl_matrix *X = Params[0];
	gsl_matrix *invKY = Params[1];
	gsl_matrix *invK = Params[2];
	gsl_matrix *scale = Params[3];
	gsl_matrix *y = Params[4];
	gsl_matrix *Tabu = Params[5];
	respSurf *prevGP = (respSurf*)Params[6];


							//Holds (in this order):  {inStrategy,constParam,s_f,s_noise,d_0,d_1, ...,  d_D|d_{D+1},      kSigma,      currMax} (where inActualD=(D or D+1) depending on wheather time is considered as an additional dimension)
							//Indices:		  {        0,         1,   2,      3,4+d,4+d, ...,4+inActualD-1, 4+inActualD,4+inActualD+1} 


	//only for EGO? Or for any sampling method?
	if(Tabu!=NULL && inTabuRegion(pGenotype,Tabu)){
//		printf("In Tabu region!!! Tabu matrix:\n");
//		printMatrix(Tabu);
//		printf("In Tabu region!!!... pGenotype=%f\n",gsl_vector_get(pGenotype,0));
//		//getchar();
//		printf("returning all this!: %f\n",DBL_MAX);
		return DBL_MAX;	//We are minimizing, so a bad value is +DBL_MAX. 
	}


	int inSampleSize = X->size1;
	int inActualD = X->size2;
	
	gsl_vector *k_new = gsl_vector_alloc (inSampleSize);
	gsl_vector *invK_kNew = gsl_vector_alloc (inSampleSize);

	double inStrategy = gsl_matrix_get(scale,0,0);
	double timeScale = gsl_matrix_get(scale,1,0);
	if(inStrategy == ST_TEMPORAL_DISCOUNT && timeScale != LEARN_TIMESCALE)
		inActualD++;
	double maxCov = pow( gsl_matrix_get(scale,2,0),2);
	double kDiag = maxCov; 					//kDiag is the kernel from the new point to itself => S_n = 0 (since current epoch) and k(x,x)=1 by definition. So, kDiag = maxCov;
	double kSigma = gsl_matrix_get(scale,4+inActualD,0);
	double currMax = -1;
	
//	printf("kSigma=%f\n",kSigma);
	if(kSigma==EGO || kSigma==EGO_PROB){
		currMax = gsl_matrix_get(scale,4+inActualD+1,0);
//		printf("currMax=%5.20f\n",currMax);
		if(outOfBounds(pGenotype)){
			gsl_vector_free(k_new);
			gsl_vector_free(invK_kNew);
			return DBL_MAX;
		}
	}

	/* Calculate the kernel from this test point (current genotype) to each sampled point */
	for(int j=0; j<inSampleSize; j++){
		double sum = 0.0;

		for(int d=0; d<D; d++){
//			printf("scale[2+%d]=%5.5f; X(%d,%d)=%5.5f; Gen(%d)=%5.5f\t",d,gsl_matrix_get(scale,4+d,0),j,d,gsl_matrix_get(X, j, d),d,gsl_vector_get(pGenotype, d));
			sum +=  -gsl_matrix_get(scale,4+d,0)*pow(gsl_matrix_get(X, j, d) - gsl_vector_get(pGenotype, d), 2);
		}

		if(inStrategy == ST_TEMPORAL_DISCOUNT && timeScale != LEARN_TIMESCALE)
			sum = exp( sum - timeScale*pow(gsl_matrix_get(y, j, 1) - 0, 2) );			//New sample being taken at the current time, the value for the D+1 dimension is zero.
		else if(inActualD > D)
			sum = exp( sum - gsl_matrix_get(scale,4+inActualD-1,0)*pow(gsl_matrix_get(X, j, D) - 0, 2) );	//New sample being taken at the current time, the value for the D+1 dimension is zero.
		else
			sum = exp( sum );

		gsl_vector_set(k_new, j, maxCov*sum );

	}


	/* Calculate mean(i) */
	double YinvKY = 0.0;
	for(int j=0; j<inSampleSize; j++)
		YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(invKY,j,0);

/*	
	printf("\n Evaluating at genotype:\n");
	printVector((gsl_vector *)pGenotype,0,pGenotype->size);
	printf("\n k_new:\n");
	printVector(k_new,0,k_new->size);
	printf("\n mean:\t%f\n",YinvKY);
//*/

//	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
//	printf("\n Y values outside:\n");
//	printMatrix(y);


//Modify priorMean to receive only the pointer --extracted from the 1x1 matrix-- to prevGP.

//	if(inStrategy==ST_PREV_SURF_MEAN_PRIOR)
		YinvKY += priorMean(prevGP,pGenotype);	//calculate the value at pGenotype according to the previous model, then remove (or add???) this value from YinvKY (which is the predicted mean).
/*
	printf("\ninvKY\n");
	printMatrix(invKY);
	printf("\nk_new\n");
	printVector(k_new,0,k_new->size);
	printf("\n mean + prior:\t%f\n",YinvKY);
//*/

	//Calculate var(i) (K^-1 * k_new )
	for(int j=0; j<inSampleSize; j++){
		double colTmp = 0.0;
		for(int jj=0; jj<inSampleSize; jj++)
			colTmp += gsl_matrix_get(invK,j,jj) * gsl_vector_get(k_new,jj);
		gsl_vector_set (invK_kNew, j, colTmp);
	}

	double kNew_invK_kNew = 0.0;
	for(int j=0; j<inSampleSize; j++)
		kNew_invK_kNew += gsl_vector_get(k_new,j) * gsl_vector_get(invK_kNew,j);
	//gsl_vector_set(gpStd, i, sqrt(kDiag - kNew_invK_kNew));	//If wanted to store mean and var, provide a pointer to a matrix (inside the encapsulated data) and an index
	double gpStd = 0;
	if(kDiag > kNew_invK_kNew)
		gpStd = sqrt(kDiag - kNew_invK_kNew);	

//	printf("\n Evaluating at genotype:\n");
//	printVector((gsl_vector *)pGenotype,0,pGenotype->size);
//	printf("\negoImprovement(%f\t%f\t%f\t%f\t)=%f\n",YinvKY, gpStd, currMax, kSigma,egoImprovement(YinvKY, gpStd, currMax, kSigma));
//	getchar();
//	printf("k_new->size=%d\n",(int)k_new->size);


	gsl_vector_free(k_new);
	gsl_vector_free(invK_kNew);




	if(kSigma==EGO || kSigma==EGO_PROB)
		return egoImprovement(YinvKY, gpStd, currMax, kSigma);
	else
		return -YinvKY - kSigma*gpStd;	//Return mean + kSigma * std
}

//Returns 1 if a new maximum was found (still exploring) or 0 if no better solution was found.
bool maxGPReg(gsl_matrix *X, gsl_matrix *y, kernelParams *w, double *bestGenotype, double kSigma, gsl_rng *r, double lastMax, int inStrategy, float constParam, gsl_matrix *Tabu, respSurf *prevGP){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

//	printf("lastMax=%f\n",lastMax);

	int inSampleSize = X->size1;
	int inActualD;
	if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)
		inActualD = X->size2 + 1;
	else
		inActualD = X->size2;	//Since X->size2 doesn't take into account the additional dimension in this case.
	double maxCov = pow( w->s_f,2);
	double scale[inActualD];
//	gsl_matrix *scale_M;			//Holds (in this order): {s_f, s_noise, d_0, d_1,...,d_D-1	  , kSigma     , currMax      , inStrategy   ,constParam   } (where inActualD=(D or D+1) depending on wheather time is considered as an additional dimension)
//	scale_M = gsl_matrix_alloc (2+inActualD+4,1);		//Indices:{0 , 1,       2,   3  ,...,2+inActualD-1, 2+inActualD, 2+inActualD+1, 2+inActualD+2,2+inActualD+3} 

	gsl_matrix *scale_M;			//Holds (in this order):  {inStrategy,constParam,s_f,s_noise,d_0,d_1, ...,  d_D|d_{D+1},      kSigma,      currMax} (where inActualD=(D or D+1) depending on wheather time is considered as an additional dimension)
	scale_M = gsl_matrix_alloc (2+inActualD+4,1);		//Indices:{        0,         1,   2,      3,4+d,4+d, ...,4+inActualD-1, 4+inActualD,4+inActualD+1} 


	double maxImprovement = -DBL_MAX;
	gsl_matrix_set(scale_M, 0, 0, inStrategy);			//Put the Strategy type in the first position
	gsl_matrix_set(scale_M, 1, 0, constParam);			//Put the constant parameter (timeScale) type in the second
	gsl_matrix_set(scale_M,2,0, w->s_f);
	gsl_matrix_set(scale_M,3,0, w->s_noise);
	gsl_matrix_set(scale_M,4+inActualD,0,kSigma);
	if(kSigma==EGO || kSigma==EGO_PROB)
		gsl_matrix_set(scale_M,4+inActualD+1,0,lastMax);
	else
		gsl_matrix_set(scale_M,4+inActualD+1,0,0);

	if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE){	
		gsl_matrix_set(scale_M, 4+inActualD-1, 0, constParam);		//Put the constParam in the lengthscale place.		
		inActualD--;							//Return inActualD to its original size
	}

	for(int d=0; d<inActualD; d++){
		scale[d] = 1/(2*pow( w->lengthScale[d] ,2));
		gsl_matrix_set(scale_M,4+d,0,scale[d]);
	}

	gsl_matrix *K = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *invK = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *invKY = gsl_matrix_calloc (inSampleSize,1);

	/* Calculate main Kernel */
	for (int i = 0; i < inSampleSize; i++){
		for (int j = 0; j < inSampleSize; j++){
			double sum=0.0;
			for(int d=0; d<inActualD; d++){
				sum += -scale[d]*pow(gsl_matrix_get(X, i, d) - gsl_matrix_get(X, j, d), 2);
			}

			if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)	//Consider the additional dimension using the constant time scale
				sum = maxCov*exp( sum - constParam*pow(gsl_matrix_get(y, i, 1) - gsl_matrix_get(y, j, 1), 2) );
			else
				sum = maxCov*exp( sum );

			if(i==j)
				sum += gsl_matrix_get(y,i,1)*pow(w->s_noise,2);	//matrix y contains in it's second colum the time at which each sample was taken (or zero if no increasing s_noise is being used)

			gsl_matrix_set (K, i, j, sum );	//matrix y contains in it's second colum the time when each sample was taken
		}
	}

	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	/* Invert K */
	double detK;
	int sign;
	gsl_permutation * perm = gsl_permutation_alloc (inSampleSize);
	gsl_linalg_LU_decomp (K, perm, &sign);

	detK = gsl_linalg_LU_det (K, sign);
	gsl_linalg_LU_invert (K, perm, invK);
	gsl_permutation_free(perm);

	w->detK = detK;

	/* Calculate K^-1 * Y **/
	for(int i=0; i<inSampleSize; i++){
		double colTmp = 0.0;
		for(int j=0; j<inSampleSize; j++)
			colTmp += gsl_matrix_get(invK,i,j) * gsl_matrix_get(y,j,0);
		gsl_matrix_set (invKY, i, 0, colTmp);
	}
pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	/* Encapsulate Parameters to send */
	//gsl_matrix *Data[9];
	
//For the recursive version we need to pass the pointer to prevGP to the maximizer, so that it can extract the required parameters and datapoints when needed. Otherwise, if they are extracted here, they would have to be passed, each, as a matrix through the Data array, which will have variable length (not allowed). So, create a 1x1 matrix to hold the pointer to prevGP.
//Nonetheless, at least the previous data points subsets (of each epoch)... called oldX for the 1 level previous epoch implementation, must be extracted here as well in order to calculate the expected improvement starting points and steps.
//Since the "extraction/conversion" from structure to matrix will be done many times, it might be worth changing the way in which the data is stored (perhaps use a matrix from the beginning to avoid conversin every time).


	if(inStrategy==ST_PREV_SURF_MEAN_PRIOR){
		respSurf *tmp = prevGP;
		while(tmp->X!=NULL){
			inSampleSize += tmp->X->size1;
			tmp = tmp->meanPr;
		}
	}


//	if(prevGP->X!=NULL)
//		inSampleSize +=  prevGP->X->size1;	//Add the number of datapoints used for the prior mean surface, so that they are taken into account for calculating the maxEgo 								starting points.
						//This should actually take into account all the previous data samples for the PSMP case. Modify this when the getBoxes function is ready 							to include recursive data.
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
	gsl_matrix *Data[7] = {X,invKY,invK,scale_M,y,Tabu,(gsl_matrix*)prevGP};
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
/*
	printf("invKY\n");
	printMatrix(invKY);

	printf("X\n");
	printMatrix(X);
	printf("invKY\n");
	printMatrix(invKY);
	printf("invK\n");
	printMatrix(invK);
	printf("scale_M\n");
	printMatrix(scale_M);
	printf("y\n");
	printMatrix(y);
//*/

//	plotGpRegFunction(Data);
//	getchar();
//	plotEGO(Data);
//	plotEGO2D(Data);
//	plotEGOAtSamples2D(Data);

//Try some simulated annealing instead of random initial points?
	int iRobust=0;
	bool bChanged = false;

#ifdef TRACK_EGO
	FILE * pFile;
	pFile = fopen("improvement.dat","w");
#endif
	/* save original handler, turn off error handling temporarily */
	gsl_error_handler_t *old_handler = gsl_set_error_handler_off ();
	int inContractionFailed = 0;
//printf("inActualD=%d\n",inActualD);
	//Even for the ST_TEMPORAL_DISCOUNT model, only a genotype of size D is required, because the time component (D+1 dimension) is fixed. We can't choose at what time to sample...

	//Instead of randomly starting the initial searches, generate a set of (n+1)^D points to be used as starting points. This number corresponds to the number of regions created by the samples within which the expected improvement is convex. 

pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	
	int inConvexBoxes = pow(inSampleSize+1,D);

//	if(inSampleSize > 100)
//		printf("inSampleSize=%d\tinConvexBoxes=%d\n",inSampleSize,inConvexBoxes);

	gsl_matrix *startP, *steps;
	startP = gsl_matrix_alloc(inConvexBoxes,D);
	steps = gsl_matrix_alloc(inConvexBoxes,D);
	getBoxes(X,startP,steps,prevGP, inSampleSize);
pdebug(__FILE__,__LINE__,__FUNCTION__,"");
//Note: For ST_PREV_SURF_MEAN_PRIOR, it might be worth starting the search not only at the centre of the regions created by the current data set, but also at the one used for the prior.
/*
	printf("Maximization starting points:\n");
	printMatrix(startP);
	printf("\nMaximization steps:\n");
	printMatrix(steps);
//*/

	while( iRobust < inConvexBoxes){
		/* Instantiate and configure minimizer */
		const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
		gsl_multimin_fminimizer *s = NULL;
		gsl_vector *ss, *genotype;
		gsl_multimin_function minex_func;

		/* Set initial step sizes to something which will fall within the box enclosing the ego convex region as defined by getBoxes */
		ss = gsl_vector_alloc (D);
		//gsl_vector_set_all (ss, 1.0);
		for(int d=0; d<D; d++)
			gsl_vector_set(ss, d, gsl_matrix_get(steps, iRobust, d));

//		printf("inConvexBoxes=%d\tiRobust=%d\n",inConvexBoxes,iRobust);
//		printVector(ss,0,ss->size);

		/* Create a starting point */
		genotype = gsl_vector_alloc (D);	//genotype of size D

		for(int d=0; d<D; d++)
			gsl_vector_set(genotype, d, gsl_matrix_get(startP, iRobust, d));

//		printVector(genotype,0,genotype->size);
/*
		if(iRobust < inSampleSize){//Start by searching for maxImprovement around the known samples
			for(int d=0; d<D; d++){
				double u = gsl_matrix_get(X,iRobust,d);
				gsl_vector_set (genotype, d, u);
			}
		}
		else{//Continue looking in random locations
			for(int d=0; d<D; d++){
				double u = (MAX_COORD - MIN_COORD)*gsl_rng_uniform(r) + MIN_COORD;
				gsl_vector_set (genotype, d, u);
//				gsl_vector_set (genotype, d, 30.0);	//try random? //Now starting at the middle point
			}
		}
*/
		size_t iter = 0;
		int status;
		double size;

		/* Initialize method and iterate */
		minex_func.n = D;
		minex_func.f = gpRegFunction;
		minex_func.params = Data;
		s = gsl_multimin_fminimizer_alloc (T, D);
		gsl_multimin_fminimizer_set (s, &minex_func, genotype, ss);

//		printf("Starting Regressed Function Maximization\n");
		do{
			iter++;
			status = gsl_multimin_fminimizer_iterate(s);

			if (status){
				char szMessage[200];
				sprintf(szMessage,"Status: %s.\tContractions failed: %d",gsl_strerror (status),++inContractionFailed);
				pdebug(__FILE__,__LINE__,__FUNCTION__,szMessage);
				break;
			}

			size = gsl_multimin_fminimizer_size (s);
			status = gsl_multimin_test_size (size, GSL_MINIMIZER_GPREG_PRECISION);
#ifdef LOG_MAX
			if (status == GSL_SUCCESS)
			{
				printf ("\t[%i]\tconverged to minimum at f() = %10.8f\n",iRobust,s->fval);
				char szMessage[200];
				char szTmp[50];
				memset(szMessage,0x00,strlen(szMessage));
				//sprintf(szMessage,"%5d ", (int)iter);
				for(int d=0; d<D; d++){
					sprintf(szTmp,"%10.5e ",gsl_vector_get (s->x, d));
					strcat(szMessage,szTmp);
				}
				sprintf(szTmp," f() = %10.8f size = %.3f\n", s->fval, size);
				strcat(szMessage,szTmp);
				printf("%s",szMessage);

			}
			else{
				char szMessage[500];
				char szTmp[500];
				sprintf(szMessage,"%5d ", (int)iter);
				for(int d=0; d<D; d++){
					sprintf(szTmp,"%10.5e ",gsl_vector_get (s->x, d));
					strcat(szMessage,szTmp);
				}
				sprintf(szTmp," f() = %10.8f size = %.3f\n", s->fval, size);
				strcat(szMessage,szTmp);
				printf("%s",szMessage);
			}
#endif
#ifdef TRACK_EGO
			fprintf(pFile,"%d\t%d\t%2.20f\t%2.20f\t%2.20f\n",iRobust,(int)iter,gsl_vector_get (s->x, 0),gsl_vector_get (s->x, 1), s->fval);
#endif			
		}while (status == GSL_CONTINUE && iter < (unsigned int)(500 * inActualD));
//For EGO, we need to keep track of the maximum improvement, but still keep tracking the lastMax, because it is needed

//Proposal: The comparison of the IF condition can be done using probImprovement, but switch both if successful (lastMAx and best probImprovement).

//		printf("next x=%f\t with improv value=%f\n",gsl_vector_get (s->x, 0), s->fval);

		if(kSigma==EGO || kSigma==EGO_PROB){
			if(-s->fval >= maxImprovement && status==GSL_SUCCESS){//Due to a saturation effect in the confidence intervals, EGO surface gets flat (saturates) to maxCov in regions without samples. So it is good practice to keep moving even if the expected improvement is the same, just to explore new regions.
				//printf("Changing because probImprovement (%3.7f>%3.7f) maxImprovement. Expecting to find something better but can't tell what :S\t RegStatus=%d (%s)\n",-s->fval,maxImprovement,status,gsl_strerror (status));
				maxImprovement = -s->fval;
				for(int d=0; d<D; d++){
					bestGenotype[d] = gsl_vector_get (s->x, d);
				}
				w->maxEGO = maxImprovement;
				//bChanged = true;
			}
			else if(-s->fval >= maxImprovement && status!=GSL_SUCCESS){	//In case we found a better solution but the algorithm didn't converge...
				if(!(isinf(-s->fval) || isnan(-s->fval))){		//just check that the better solution is a number (not inf nor nan), and use it.
					maxImprovement = -s->fval;
					for(int d=0; d<D; d++){
						bestGenotype[d] = gsl_vector_get (s->x, d);
					}
					w->maxEGO = maxImprovement;
				}
			}
		}
		else{
			if(-s->fval > lastMax && status==GSL_SUCCESS){
				for(int d=0; d<D; d++)
					bestGenotype[d] = gsl_vector_get (s->x, d);
				//lastMax = -s->fval;
				bChanged = true;
//				printf("Changing! (%3.7f>?%3.7f)Expecting to find: %f\t RegStatus=%d (%s)\n",-s->fval,lastMax,-s->fval,status,gsl_strerror (status));
			}
		}

		gsl_vector_free(genotype);
		gsl_vector_free(ss);
		gsl_multimin_fminimizer_free (s);
		iRobust++;
	}

	/* restore original handler */
	gsl_set_error_handler (old_handler);
//		printf("Max ExpectedImprovement %3.7f \n",maxImprovement); getchar();
#ifdef TRACK_EGO
		fclose(pFile);
//		printf("Max ExpectedImprovement (%3.7f) maxImprovement. Expecting to find something better but can't tell what :S\n",maxImprovement);
#endif

	gsl_matrix_free(scale_M);
	gsl_matrix_free(K);
	gsl_matrix_free(invK);
	gsl_matrix_free(invKY);
	gsl_matrix_free(startP);
	gsl_matrix_free(steps);

	if(kSigma==EGO){
		if(maxImprovement / lastMax >= EGO_EXP_IMPROV_STOP)//If we expect a large enough improvement
			bChanged = true;
		else
			bChanged = false;
	}
	/*Double check this stopping criteria!!!!*/
	if(kSigma==EGO_PROB){
		if(maxImprovement >= EGO_MIN_PROB_IMPROV_STOP)//If we expect a large enough improvement
			bChanged = true;
		else
			bChanged = false;
	}

	return bChanged;
}

//Function built from gpRegFunction, modified to be able to plot EGO.
void plotEGO2D(void *Data){

    pdebug(__FILE__,__LINE__,__FUNCTION__,"A");

	/* Restore the data to the original form */
	gsl_matrix **Params = (gsl_matrix **)Data;
	gsl_matrix *X = Params[0];
	gsl_matrix *invKY = Params[1];
	gsl_matrix *invK = Params[2];
	gsl_matrix *scale = Params[3];
/*
	printf("X\n");
	printMatrix(X);
	printf("invKY\n");
	printMatrix(invKY);
	printf("invK\n");
	printMatrix(invK);
	printf("scale_M\n");
	printMatrix(scale);
*/
	int inSampleSize = X->size1;
	int inActualD = X->size2;
	gsl_vector *k_new = gsl_vector_alloc (inSampleSize);
	gsl_vector *invK_kNew = gsl_vector_alloc (inSampleSize);

	double maxCov = pow( gsl_matrix_get(scale,2,0),2);
	double kDiag = maxCov; //kDiag is the kernel from the new point to itself => S_n = 0 (since current epoch) and k(x,x)=1 by definition. So, kDiag = maxCov;
	double kSigma = gsl_matrix_get(scale,4+inActualD,0);
	double currMax = -1;
	if(kSigma==EGO || kSigma==EGO_PROB){
		currMax = gsl_matrix_get(scale,4+inActualD+1,0);
	//	printf("kSigma=%f\n",kSigma);
	}


	#define EGOPlotPoints 100
    	pdebug(__FILE__,__LINE__,__FUNCTION__,"B");
    	gsl_vector *pGenotype = gsl_vector_alloc(inActualD);
	double egoPlot[EGOPlotPoints][EGOPlotPoints][5];//[x,y,predMean,predVar,expImp]
	double delta = (MAX_COORD - MIN_COORD + 1)/EGOPlotPoints;

	/* Calculate the kernel for all possible test points to each sampled point */
	for(int iPlot=0; iPlot<EGOPlotPoints; iPlot++){
		for(int jPlot=0; jPlot<EGOPlotPoints; jPlot++){

			egoPlot[iPlot][jPlot][0] = delta * iPlot;
			egoPlot[iPlot][jPlot][1] = delta * jPlot;
			gsl_vector_set(pGenotype, 0, egoPlot[iPlot][jPlot][0]);
			gsl_vector_set(pGenotype, 1, egoPlot[iPlot][jPlot][1]);
	
			for(int j=0; j<inSampleSize; j++){
			    double sum = 0.0;
			    for(int d=0; d<D; d++){
				sum +=  -gsl_matrix_get(scale,4+d,0)*pow(gsl_matrix_get(X, j, d) - gsl_vector_get(pGenotype, d), 2);
			    }
			    if(inActualD > D)
				sum +=  -gsl_matrix_get(scale,4+inActualD-1,0)*pow(gsl_matrix_get(X, j, D) - 0, 2);	//New sample being taken at the current time, the value for the D+1 dimension is zero.
	
			    gsl_vector_set(k_new, j, maxCov*exp( sum ));
			}

			/* Calculate mean(i) */
			double YinvKY = 0.0;
			for(int j=0; j<inSampleSize; j++)
			    YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(invKY,j,0);


			//Calculate var(i) (K^-1 * k_new )
			for(int j=0; j<inSampleSize; j++){
			    double colTmp = 0.0;
			    for(int jj=0; jj<inSampleSize; jj++)
				colTmp += gsl_matrix_get(invK,j,jj) * gsl_vector_get(k_new,jj);
			    gsl_vector_set (invK_kNew, j, colTmp);
			}

			double kNew_invK_kNew = 0.0;
			for(int j=0; j<inSampleSize; j++)
			    kNew_invK_kNew += gsl_vector_get(k_new,j) * gsl_vector_get(invK_kNew,j);
			//gsl_vector_set(gpStd, i, sqrt(kDiag - kNew_invK_kNew));	//If wanted to store mean and var, provide a pointer to a matrix (inside the encapsulated data) and an index
			double gpStd = sqrt(kDiag - kNew_invK_kNew);
//			printf("YinvKY=%f\t gpStd=%f\t currMax=%f\t, kSigma=%f\n",YinvKY, gpStd, currMax, kSigma);
			egoPlot[iPlot][jPlot][2] = YinvKY;
			egoPlot[iPlot][jPlot][3] = gpStd;
			egoPlot[iPlot][jPlot][4] = egoImprovement(YinvKY, gpStd, currMax, kSigma);
			//printf("egoPlot=%10.10f\n",egoPlot[iPlot][jPlot][2]);
		}
	}




		gsl_vector_free(k_new);
		gsl_vector_free(invK_kNew);
		gsl_vector_free(pGenotype);


	FILE * pFile;
	if ((pFile = fopen("egoPlot.dat","w"))==NULL) {
	        printf("Cannot open file.\n");
        	exit(1);
      	}
	pdebug(__FILE__,__LINE__,__FUNCTION__,"File");

	char szMessage[200];
	for(int iPlot=0; iPlot<EGOPlotPoints; iPlot++){
		for(int jPlot=0; jPlot<EGOPlotPoints; jPlot++){
			memset(szMessage,0x00,sizeof(szMessage));
		        sprintf(szMessage,"%10.10f\t%10.10f\t%10.10f\t%10.10f\t%10.10f\n",egoPlot[iPlot][jPlot][0],egoPlot[iPlot][jPlot][1],egoPlot[iPlot][jPlot][2],egoPlot[iPlot][jPlot][3],egoPlot[iPlot][jPlot][4]);
//      	  	printf("%10.10f\t%10.10f\t%i\n",egoPlot[iPlot][0],egoPlot[iPlot][1],(int) strlen(szMessage));
        		fprintf(pFile,"%s",szMessage);
		}
	}
	//sleep(1);
	fclose(pFile);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"File");

}

/***********************************************************************************************************************
Returns the root mean squared error (RMSE) that a given model (W) has when trying to predict the training points.
(In theory should be zero (in absence of measurement noise) but it is not due to numerical errors)
************************************************************************************************************************/
double getRMSE(gsl_matrix *X, gsl_matrix *y, kernelParams *W, float constParam, int inStrategy){

	int inSampleSize = X->size1;
	int inActualD;
	if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)
		inActualD = X->size2 + 1;
	else
		inActualD = X->size2;	//Since X->size2 doesn't take into account the additional dimension in this case.
	double maxCov = pow( W->s_f,2);	
	double scale[inActualD];

	for(int d=0; d<inActualD; d++){
		scale[d] = 1/(2*pow( W->lengthScale[d] ,2));
	}

	gsl_matrix *K = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *invK = gsl_matrix_calloc (inSampleSize, inSampleSize);
	gsl_matrix *invKY = gsl_matrix_calloc (inSampleSize,1);

	gsl_vector *k_new = gsl_vector_alloc (inSampleSize);
	gsl_vector *invK_kNew = gsl_vector_alloc (inSampleSize);

	/* Calculate main Kernel */
	for (int i = 0; i < inSampleSize; i++){
		for (int j = 0; j < inSampleSize; j++){
			double sum=0.0;
			for(int d=0; d<inActualD; d++){
				sum += -scale[d]*pow(gsl_matrix_get(X, i, d) - gsl_matrix_get(X, j, d), 2);
			}

			if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)	//Consider the additional dimension using the constant time scale
				sum = maxCov*exp( sum - constParam*pow(gsl_matrix_get(y, i, 1) - gsl_matrix_get(y, j, 1), 2) );
			else
				sum = maxCov*exp( sum );

			if(i==j)
				sum += gsl_matrix_get(y,i,1)*pow(W->s_noise,2);	//matrix y contains in it's second colum the time at which each sample was taken (or zero if no increasing s_noise is being used)

			gsl_matrix_set (K, i, j, sum );	
		}
	}

	/* Invert K */
	double detK;
	int sign;
	gsl_permutation * perm = gsl_permutation_alloc (inSampleSize);
	gsl_linalg_LU_decomp (K, perm, &sign);
	detK = gsl_linalg_LU_det (K, sign);
	gsl_linalg_LU_invert (K, perm, invK);
	gsl_permutation_free(perm);

	W->detK = log(detK);

	/* Calculate K^-1 * Y **/
	for(int i=0; i<inSampleSize; i++){
		double colTmp = 0.0;
		for(int j=0; j<inSampleSize; j++)
			colTmp += gsl_matrix_get(invK,i,j) * gsl_matrix_get(y,j,0);
		gsl_matrix_set (invKY, i, 0, colTmp);
	}


	/* Having calculated the constant parameters, try to predict each of the samples */
	double dblSE = 0;
	for(int inTest=0; inTest < inSampleSize; inTest++){

		/* Calculate the kernel from this test point (current genotype) to each sampled point */
		for(int j=0; j<inSampleSize; j++){
			double sum = 0.0;

			for(int d=0; d<D; d++){
//				printf("scale[2+%d]=%5.5f; X(%d,%d)=%5.5f; Gen(%d)=%5.5f\t",d,gsl_matrix_get(scale,4+d,0),j,d,gsl_matrix_get(X, j, d),d,gsl_vector_get(pGenotype, d));
				sum +=  -scale[d]*pow(gsl_matrix_get(X, j, d) - gsl_matrix_get(X, inTest, d), 2);
			}

			if(inStrategy == ST_TEMPORAL_DISCOUNT && constParam != LEARN_TIMESCALE)
				sum = exp( sum - constParam*pow(gsl_matrix_get(y, j, 1) - 0, 2) );			//New sample being taken at the current time, the value for the D+1 dimension is zero.
			else if(inActualD > D)
				sum = exp( sum - scale[D]*pow(gsl_matrix_get(X, j, D) - 0, 2) );			//New sample being taken at the current time, the value for the D+1 dimension is zero.
			else
				sum = exp( sum );

			gsl_vector_set(k_new, j, maxCov*sum );

		}
		/* Calculate mean(i) */
		double YinvKY = 0.0;
		for(int j=0; j<inSampleSize; j++)
			YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(invKY,j,0);
//		printf("predicted mean=%f \t vs \t sample=%f \t \t SqError = %E\t",YinvKY, gsl_matrix_get(y,inTest,0),pow(YinvKY - gsl_matrix_get(y,inTest,0),2));
		

		/* Calculate var(i) (K^-1 * k_new ) */
		for(int j=0; j<inSampleSize; j++){
			double colTmp = 0.0;
			for(int jj=0; jj<inSampleSize; jj++)
				colTmp += gsl_matrix_get(invK,j,jj) * gsl_vector_get(k_new,jj);
			gsl_vector_set (invK_kNew, j, colTmp);
		}
		
		double kNew_invK_kNew = 0.0;
		for(int j=0; j<inSampleSize; j++)
			kNew_invK_kNew += gsl_vector_get(k_new,j) * gsl_vector_get(invK_kNew,j);

		double gpStd = 0;
		if(maxCov > kNew_invK_kNew)
			gpStd = sqrt(maxCov - kNew_invK_kNew);		
//		printf("predicted var=%f \t  predicted std=%f\n",gpStd, sqrt(maxCov - kNew_invK_kNew));

		if(isinf(gpStd) || isnan(gpStd))//Use this to penalize the model each time there is a singularity...
			dblSE = dblSE + 1*pow(YinvKY - gsl_matrix_get(y,inTest,0),2);
		else
			dblSE = dblSE + pow(YinvKY - gsl_matrix_get(y,inTest,0),2);
			

	}
	gsl_matrix_free(K);
	gsl_matrix_free(invK);
	gsl_matrix_free(invKY);
	gsl_vector_free(k_new);
	gsl_vector_free(invK_kNew);
	
	return sqrt(dblSE/inSampleSize);
}

//Function built from plotEGO2D build from gpRegFunction, modified to be able to plot EGO.
void plotEGOAtSamples2D(void *Data){

	/* Restore the data to the original form */
	gsl_matrix **Params = (gsl_matrix **)Data;
	gsl_matrix *X = Params[0];
	gsl_matrix *invKY = Params[1];
	gsl_matrix *invK = Params[2];
	gsl_matrix *scale = Params[3];

	int inSampleSize = X->size1;
	int inActualD = X->size2;
	gsl_vector *k_new = gsl_vector_alloc (inSampleSize);
	gsl_vector *invK_kNew = gsl_vector_alloc (inSampleSize);

	double maxCov = pow( gsl_matrix_get(scale,2,0),2);
	double kDiag = maxCov; //kDiag is the kernel from the new point to itself => S_n = 0 (since current epoch) and k(x,x)=1 by definition. So, kDiag = maxCov;
	double kSigma = gsl_matrix_get(scale,4+inActualD,0);
	double currMax = -1;

	double egoPlot[100][5];//[x,y,predMean,predVar,expImp]

	if(kSigma==EGO || kSigma==EGO_PROB){
		currMax = gsl_matrix_get(scale,4+inActualD+1,0);
	}

    	gsl_vector *pGenotype = gsl_vector_alloc(inActualD);


	/* Calculate the kernel for all the samples to see if we are actually predicting the mean correctly */
	for(int iPlot=0; iPlot<inSampleSize; iPlot++){

			egoPlot[iPlot][0] = gsl_matrix_get(X, iPlot, 0);
			egoPlot[iPlot][1] = gsl_matrix_get(X, iPlot, 1);
			gsl_vector_set(pGenotype, 0, egoPlot[iPlot][0]);
			gsl_vector_set(pGenotype, 1, egoPlot[iPlot][1]);
	
			for(int j=0; j<inSampleSize; j++){
			    double sum = 0.0;
			    for(int d=0; d<D; d++){
				sum +=  -gsl_matrix_get(scale,4+d,0)*pow(gsl_matrix_get(X, j, d) - gsl_vector_get(pGenotype, d), 2);
			    }
			    if(inActualD > D)
				sum +=  -gsl_matrix_get(scale,4+inActualD-1,0)*pow(gsl_matrix_get(X, j, D) - 0, 2);	//New sample being taken at the current time, the value for the D+1 dimension is zero.
	
			    gsl_vector_set(k_new, j, maxCov*exp( sum ));
			}

			/* Calculate mean(i) */
			double YinvKY = 0.0;
			for(int j=0; j<inSampleSize; j++)
			    YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(invKY,j,0);


			//Calculate var(i) (K^-1 * k_new )
			for(int j=0; j<inSampleSize; j++){
			    double colTmp = 0.0;
			    for(int jj=0; jj<inSampleSize; jj++)
				colTmp += gsl_matrix_get(invK,j,jj) * gsl_vector_get(k_new,jj);
			    gsl_vector_set (invK_kNew, j, colTmp);
			}

			double kNew_invK_kNew = 0.0;
			for(int j=0; j<inSampleSize; j++)
			    kNew_invK_kNew += gsl_vector_get(k_new,j) * gsl_vector_get(invK_kNew,j);

			double gpStd = 0;
			if(kDiag > kNew_invK_kNew) 
				gpStd = sqrt(kDiag - kNew_invK_kNew);
//			printf("YinvKY=%f\t gpStd=%f\t currMax=%f\t, kSigma=%f\n",YinvKY, gpStd, currMax, kSigma);
			egoPlot[iPlot][2] = YinvKY;
			egoPlot[iPlot][3] = gpStd;
			egoPlot[iPlot][4] = egoImprovement(YinvKY, gpStd, currMax, kSigma);
			//printf("egoPlot=%10.10f\n",egoPlot[iPlot][jPlot][2]);
	}




	gsl_vector_free(k_new);
	gsl_vector_free(invK_kNew);
	gsl_vector_free(pGenotype);


	FILE * pFile;
	if ((pFile = fopen("egoPlot_Samples.dat","w"))==NULL) {
	        printf("Cannot open file.\n");
        	exit(1);
      	}
	pdebug(__FILE__,__LINE__,__FUNCTION__,"File");

	char szMessage[200];
	for(int iPlot=0; iPlot<inSampleSize; iPlot++){
		memset(szMessage,0x00,sizeof(szMessage));
	        sprintf(szMessage,"%10.10f\t%10.10f\t%10.10f\t%E\t%E\n",egoPlot[iPlot][0],egoPlot[iPlot][1],egoPlot[iPlot][2],egoPlot[iPlot][3],egoPlot[iPlot][4]);
//     	  	printf("%10.10f\t%10.10f\t%i\n",egoPlot[iPlot][0],egoPlot[iPlot][1],(int) strlen(szMessage));
       		fprintf(pFile,"%s",szMessage);
	}
	fprintf(pFile,"\n\nParameters used: \nmaxCov=%10.10f\tl(0)=%10.10f\tl(1)=%10.10f\n",sqrt(maxCov),sqrt(1/(2*gsl_matrix_get(scale,4+0,0))),sqrt(1/(2*gsl_matrix_get(scale,4+1,0))));

	//sleep(1);
	fclose(pFile);
	pdebug(__FILE__,__LINE__,__FUNCTION__,"File");

}

//Needs attention. All the points of all previous epochs must be considered for the recursive strategy, currently only 1 previous epoch is being used.
void getBoxes(gsl_matrix *X, gsl_matrix *startP, gsl_matrix *steps, respSurf *prevGP, int N){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");
#define LAMBDA_STEP 5

	gsl_matrix *pX;
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	if(prevGP->X == NULL){
		pdebug(__FILE__,__LINE__,__FUNCTION__,"");
		N = N+2;	//To include left and right bounds as datapoints
		pX = X;
	}
	else{
		pdebug(__FILE__,__LINE__,__FUNCTION__,"");
		N = N + 2;
		//Create a new matrix containing all datapoints, those in X and those in prevGP->X (recursively)
		pX = gsl_matrix_calloc(N-2,D);	

		//Copy those from X
		for(unsigned int i=0; i<X->size1; i++)
			for(int d=0;d<D;d++)
				gsl_matrix_set(pX,i,d,gsl_matrix_get(X,i,d));
		
		//Now copy all the datapoints stored in X at each recurssion level
		int inSampleCounter = X->size1;
		respSurf *tmp = prevGP;
		while(tmp->X!=NULL){
			for(unsigned int i=0; i<tmp->X->size1; i++){
				for(int d=0;d<D;d++)
					gsl_matrix_set(pX,inSampleCounter,d,gsl_matrix_get(tmp->X,i,d));
				inSampleCounter++;
			}
			tmp = tmp->meanPr;
		}
	}

	//Claculate the guiding lines to be used for the crossing points
	gsl_vector *orderedX[D];
	gsl_vector *middlePointsV[D];
	gsl_vector *stepsV[D];


	for(int d=0;d<D;d++){
		orderedX[d] = gsl_vector_calloc(N);
		middlePointsV[d] = gsl_vector_calloc(N-1);
		stepsV[d] = gsl_vector_calloc(N-1);

		gsl_vector_set(orderedX[d],0,MIN_COORD);
		for(int n=1; n<N-1; n++)
			gsl_vector_set(orderedX[d],n,gsl_matrix_get(pX,n-1,d));
		gsl_vector_set(orderedX[d],N-1,MAX_COORD);


		// Order the data in each dimension
		gsl_sort_vector (orderedX[d]);

//		printf("\nMiddle points: [%d]\n",d);	
		//Calculate the middle points
		for(int n=0; n<N-1; n++){
			gsl_vector_set(middlePointsV[d],n, gsl_vector_get(orderedX[d],n) + (gsl_vector_get(orderedX[d],n+1) - gsl_vector_get(orderedX[d],n))/2  );
//			printf("%f\n",gsl_vector_get(middlePointsV[d],n));
		}

//		printf("\nInitial steps [%d]:\n",d);
		//Calculate the initial steps, which should be (LAMBDA_STEP times) smaller than half of the width of the box in each dimension
		for(int n=0; n<N-1; n++){
			gsl_vector_set(stepsV[d],n, (gsl_vector_get(orderedX[d],n+1) - gsl_vector_get(orderedX[d],n))/(2*LAMBDA_STEP)  );
//			printf("%f\n",gsl_vector_get(stepsV[d],n));
		}
//		getchar();
	}
	if(pX!=X)
		gsl_matrix_free(pX);

//Put everything in the same loop to avoid vectors of pointers to gsl_array	

	//Fill the matrix containing all the crossing points, along with the corresponding initial step sizes.
	gsl_matrix_set_all(startP,-1);
	gsl_matrix_set_all(steps,-1);
	for(int d=0;d<D;d++){
		crossProd(startP,middlePointsV[d]);
		crossProd(steps,stepsV[d]);

		gsl_vector_free(orderedX[d]);
		gsl_vector_free(middlePointsV[d]);
		gsl_vector_free(stepsV[d]);
	}

/*	printf("\nInside getBoxes: Maximization starting points:\n");
	printMatrix(startP);
	printf("\nMaximization steps:\n");
	printMatrix(steps);
//*/
	
}

/*************************************************************************
M is the input matrix filled with -1 wherever there are no datapoints
M serves as output matrix, where the new dimension is appended
v is the vector to be appended. For each element of v, a copy of M is created
	and the additional dimension is filled with that element of v.
M is expected to be able to contain already all the copies (result of cross product).
*************************************************************************/
void crossProd(gsl_matrix *M, gsl_vector *v){
	int usedD=0;
	int usedN=0;
	while(gsl_matrix_get(M,usedN,usedD)!=-1)
		usedD++;
//	printf("usedD=%d\n",usedD);

	while(gsl_matrix_get(M,usedN,0)!=-1)
		usedN++;
//	printf("usedN=%d\n",usedN);


	for(unsigned int n=0; n<v->size; n++){
		if(usedN==0){
			gsl_matrix_set(M, n, usedD, gsl_vector_get(v,n) );//Set the last dimension of the matrix to the (Get) current element from new vector
		}
		else{
			for(int oldN=0; oldN < usedN; oldN++){
				for(int d=0;d<usedD;d++){//Get element from matrix
					gsl_matrix_set(M, n*usedN+oldN, d, gsl_matrix_get(M,oldN,d) );
				}
	
					gsl_matrix_set(M, n*usedN+oldN, usedD, gsl_vector_get(v,n) );//Set the last dimension of the matrix to the (Get) current element from new vector
			}
		}
	}

//	printf("Matrix M after filling:\n");
//	printMatrix(M);
//	getchar();
}


//Change priorMean to accept a pointer to prevGP, and each time extract the necessary parameters and data from the corresponding recurssion level in prevGP.
//The returning value must be not only the prediction of the model at this level, but should also include the corresponding prior mean. So, return something like
// return YinvKY + priorMean(prevGP->prevGP)... (or meanPrior???)

double priorMean(respSurf *prevGP, const gsl_vector *pGenotype){
//	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	if(prevGP->X==NULL){//First epoch, so return mean only. The mean shall not be calculated again from y, since it has been removed already and the result would be zero. 
//		pdebug(__FILE__,__LINE__,__FUNCTION__,"oldW==NULL\n");
		return prevGP->constScalarMean;
	}
	int inOldSampleSize = prevGP->X->size1;

	gsl_vector *k_new = gsl_vector_alloc (inOldSampleSize);


	double maxCov = pow(prevGP->w->s_f,2);

	//oldW holds (in this order):  {s_f,s_noise,d_0,d_1, ...,  d_D,constScalarMean}, where constScalarMean is used for constant mean prior.
	double scale[D];
	for(int d=0; d<D; d++){
		scale[d] = 1/(2*pow( prevGP->w->lengthScale[d] ,2));
//		printf("for calculating oldk_new, \t scale[%d]=%f\n",d,scale[d]);
	}


	/* Calculate the kernel from this test point (current genotype) to each sampled point from the old epoch (i.e. predict the value according to previous epoch model) */
	for(int j=0; j<inOldSampleSize; j++){
		double sum = 0.0;

		for(int d=0; d<D; d++){
			sum +=  -scale[d] * pow(gsl_matrix_get(prevGP->X, j, d) - gsl_vector_get(pGenotype,d), 2);
//			printf("sum=%f\tscale[d]=%f\toldX=%f\tpGenotype=%f\tpow(oldX-pGenotype)=%f\tmaxCov*exp(sum)=%f\n",sum,scale[d],gsl_matrix_get(oldX, j, d),gsl_vector_get(pGenotype,d),pow(gsl_matrix_get(oldX, j, d) - gsl_vector_get(pGenotype,d), 2),maxCov * exp(sum));
		}

		gsl_vector_set(k_new, j, maxCov * exp(sum) );

	}

	/* Calculate mean(i) */
	double YinvKY = 0.0;
	for(int j=0; j<inOldSampleSize; j++)
	    YinvKY += gsl_vector_get(k_new,j) * gsl_matrix_get(prevGP->invKY,j,0);


/*
	printf("***Calculating the prior mean***\n");
	printf("pGenotype=%f\n",gsl_vector_get(pGenotype,0));
	printf("oldX:\n");
	printMatrix(oldX);
	printf("old_S_F=%f\n",gsl_matrix_get(oldW,0,0));
	printf("old_S_N=%f\n",gsl_matrix_get(oldW,1,0));
	printf("old_lengthScale[0]=%f\n",gsl_matrix_get(oldW,2,0));

	printf("maxCov=%f\n",maxCov);
	printf("scale[0]=%f\n",scale[0]);


	printf("old_invKY\n");
	printMatrix(invKY);
	printf("old_k_new\n");
	printVector(k_new,0,k_new->size);
	printf("Prior (YinvKY)=%f\n",YinvKY);
	getchar();
//*/

	gsl_vector_free(k_new);


	
/* No! there is already a stopping condition at the beginning of the function, just return the sum!
	if(prevGP->meanPr!=NULL) //Or... another condition to prematurely stop the recursion depth
		return YinvKY + priorMean(prevGP->meanPr,pGenotype);
	else
*/
		return YinvKY + priorMean(prevGP->meanPr,pGenotype);
//		return YinvKY;	//This is the prediction of the previous model at the genotype, which is to be used as a prior mean.



}















