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

double gsl_linalg_SV_invert(gsl_matrix *U, gsl_matrix *V, gsl_vector *S, gsl_matrix *pseudoInvU){	//http://en.wikipedia.org/wiki/Singular_value_decomposition
	int n = S->size;
	double pseudoDet = 1;	//http://en.wikipedia.org/wiki/Pseudo-determinant

	//pseudoInvU = V pseudoInvS U_trans
	//Calculate the pseudo inverse of S by which is formed by replacing every nonzero diagonal entry by its reciprocal and transposing the resulting matrix.
	for(int i=0; i<n; i++){
		if(gsl_vector_get(S,i)!=0){
			pseudoDet *= gsl_vector_get(S,i);
			gsl_vector_set(S,i,1/gsl_vector_get(S,i));
		}
	}

	//Multiply: V*pseudoInvS
	gsl_matrix *VpsInvU = gsl_matrix_calloc(n,n);
	for(int i=0; i<n;i++)
		for(int j=0; j<n;j++)
			gsl_matrix_set(VpsInvU,i,j, gsl_matrix_get(V,i,j)*gsl_vector_get(S,j));

	//Multiply: (V*pseudoInvS)*U_trans [transposing directly by swapping k,j -> j,k]
	for(int i=0; i<n;i++){
		for(int j=0; j<n;j++){
			double sum = 0;
			for(int k=0; k<n;k++){
				sum+= gsl_matrix_get(VpsInvU,i,k) * gsl_matrix_get(U,j,k);
			}
			gsl_matrix_set(pseudoInvU,i,j, sum);
		}
	}

	gsl_matrix_free(VpsInvU);
	return pseudoDet;
	
}
gsl_rng *initRandGen(double seed){

	const gsl_rng_type * T;
	gsl_rng * r;
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r,seed);
	return r;
}

double linSpace(double lBound, double uBound, int gridSize, gsl_vector *v){

	double step = (double)(uBound - lBound)/(gridSize - 1);
	gsl_vector_set (v, 0, lBound);

	for(int i=1; i<gridSize; i++)
		gsl_vector_set (v, i, gsl_vector_get (v, i-1) + step);

	return step;
}

void printMatrix(gsl_matrix *out){

	for(unsigned int i=0; i<out->size1; i++){
		printf("%i\t",i);
		for(unsigned int j=0; j<out->size2; j++){
			printf("%1.8f\t",gsl_matrix_get(out,i,j));
		}
		printf("\n");
	}
	return;
}

void printVector(gsl_vector *out, int start, int end){

	for( int i=start; i<end; i++){
		printf("%i\t",i);
		printf("%1.8f\t",gsl_vector_get(out,i));
		printf("\n");
	}
	return;
}

void printXYMatrices(const char* szMessage, gsl_matrix *x, gsl_matrix *y, int fromRow, int toRow){
	printf("\n***%s***\n",szMessage);
	if(x->size1 != y->size1){
		printf("Matrices length mismatch!\n");
		return;
	}
	if((unsigned int)toRow > x->size1){
		printf("Dimension exceeded!\n");
		return;
	}
	if(fromRow == toRow && toRow==0){//Show all
		fromRow = 0;
		toRow = x->size1;
	}
	if(toRow >= fromRow  && fromRow >=0){
		for(int i=fromRow; i<toRow; i++){
			printf("%i\t",i);
			for(unsigned int j=0; j<x->size2; j++){
				printf("%1.28f\t",gsl_matrix_get(x,i,j));
			}
			printf("\t\t\t");
			for(unsigned int j=0; j<y->size2; j++){
				printf("%1.28f\t",gsl_matrix_get(y,i,j));
			}
			printf("\n");
		}
	}
}

double peaks(double x, double y){
	return 3*pow(1-x,2)*exp( -pow(x,2) - pow(y+1,2) )  -  10*(x/5 - pow(x,3) - pow(y,5))*exp( -pow(x,2) - pow(y,2) ) - (1/3)*exp( -pow(x+1,2) - pow(y,2) );
}

/* 
Receives:
	gsl_matrix *ranges:	Dx3 Matrix specifying xMin, xMax, and gridSize for each dimension
	gsl_matrix *outM:	pointer to resulting MxD matrix containing the D coordinates for each of the resulting P points, P = prod_{i=1}^D(gridSize_i)
Returns:
	int P:			Resulting length of outM = prod_{i=1}^D(gridSize_i)
*/
/* 
Perform the cross product of matrices a and b
I.e: returns all the possible combinations of elements from sets a and b
*/
void crossProduct(gsl_matrix *a, gsl_vector *b, gsl_matrix *out){

	int n = a->size1;
	int m = b->size;
	int d1 = a->size2;

	for(int i=0; i<n; i++){
		for(int j=0; j<m; j++){
			for(int d=0; d<d1+1; d++){
				if(d < d1){
					gsl_matrix_set(out, i*m+j, d, gsl_matrix_get(a,i,d));				
				}
				else{
					gsl_matrix_set(out, i*m+j, d, gsl_vector_get(b,j));
				}
			}			
		}
	}
	return;
}

int linSpaceRN(gsl_matrix *ranges, gsl_matrix *outM){

	int dim = ranges->size1;

	//Create vector for first dimension
	double lBound = gsl_matrix_get(ranges,0,0);
	double uBound = gsl_matrix_get(ranges,0,1);
	int gridSize = gsl_matrix_get(ranges,0,2);
	gsl_vector *vTmp = gsl_vector_calloc(gridSize);
	linSpace(lBound,uBound,gridSize,vTmp);

	//Convert first dimension to a matrix:
	gsl_matrix *tmpM = gsl_matrix_calloc(vTmp->size,1);
	for(unsigned int i=0; i<vTmp->size; i++)
		gsl_matrix_set(tmpM,i,0,gsl_vector_get(vTmp,i));
	gsl_vector_free(vTmp);
	
	//Extract and Combine all the linspaced vectors in the output matrix, providing all the possible combinations of the vector points in order to define the grid
	//by calculating the cross products of each dimension and the resulting of the previous cross products
	gsl_matrix *outTmp=NULL;
	for(int d=1; d < dim; d++){
		//Extract information of ranges for each dimension
		lBound = gsl_matrix_get(ranges,d,0);
		uBound = gsl_matrix_get(ranges,d,1);
		gridSize = gsl_matrix_get(ranges,d,2);
		gsl_vector *vTmp = gsl_vector_calloc(gridSize);

		//Create the equally spaced vector according to the extracted information
		linSpace(lBound,uBound,gridSize,vTmp);
		
		//Combine (cross product) the previously obtained matrix with the new vector
		outTmp = gsl_matrix_calloc(tmpM->size1*vTmp->size,d+1);
		crossProduct(tmpM, vTmp, outTmp);

		gsl_matrix_free(tmpM);
		gsl_vector_free(vTmp);
		tmpM = outTmp;
	}

	gsl_matrix_memcpy (outM, outTmp);
	return outM->size1;
}
void getMax(double *maxVal, int *maxIndex, gsl_vector *gpMean){
	for(unsigned int i=0; i<gpMean->size; i++){
		if(gsl_vector_get(gpMean,i) > *maxVal){
			*maxVal = gsl_vector_get(gpMean,i);
			*maxIndex = i;
		}
	}
	return;
}


void pdebug(const char* file, int line, const char* func, const char* szMessage){
#ifdef DEBUG
	printf("%s:%d\t%s\t\t%s\n",file, line, func, szMessage);
#endif
	return;
}

void dumpLog(history *H, int n, int inJobArrayIndex,int inStrategy, double vLen, double hSev){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

	char szMessage[5000];
	char szTmp[1000];
	memset(szMessage,0x00,sizeof(szMessage));
	memset(szTmp,0x00,sizeof(szTmp));
	
	sprintf(szTmp,"histLog/landscapes/gpHist_%0d_%05d_%d_%1.1f_%1.1f.dat",D,inJobArrayIndex,inStrategy,vLen,hSev);

	FILE * pFile;

	if((pFile = fopen(szTmp,"w"))==NULL){
		printf("Unable to open file histLog/gpHist_... for writing\n");
		return;
	}


	int inActualD = H->X->size2;
	
	memset(szMessage,0x00,sizeof(szMessage));
	memset(szTmp,0x00,sizeof(szTmp));

	for(int d=0; d<inActualD; d++){
		sprintf(szTmp,"X(%d)\t\t",d);
		strcat(szMessage,szTmp);
	}

	sprintf(szTmp,"y\t\tEpoch\t\tUsePoint\tlogLike\t\ts_f\t\t");
	strcat(szMessage,szTmp);
	for(int d=0; d<inActualD+1; d++){
		sprintf(szTmp,"lengthScale(%d)\t",d);
		strcat(szMessage,szTmp);
	}
	sprintf(szTmp,"s_noise\t\tmaxEGO\n");
	strcat(szMessage,szTmp);

	fprintf(pFile,"%s",szMessage);

//	printf("******Starting cycle*********\n");
	for(int i=0; i<n; i++){

		memset(szMessage,0x00,sizeof(szMessage));
		memset(szTmp,0x00,sizeof(szTmp));

		for(int d=0; d<inActualD; d++){
			sprintf(szTmp,"%10.10f\t",gsl_matrix_get (H->X, i, d));
			strcat(szMessage,szTmp);
		}
		sprintf(szTmp,"%10.10f\t%f\t%f\t%10.10f\t%10.10f\t",gsl_matrix_get(H->y, i, 0), (H->currPeriod - floor(i/gPeakChangeFreq)), gsl_vector_get(H->usePoint,i), H->w[i].logLike, H->w[i].s_f);
		strcat(szMessage,szTmp);

		for(int d=0; d<inActualD+1; d++){
			sprintf(szTmp,"%10.10f\t", H->w[i].lengthScale[d]);
			strcat(szMessage,szTmp);
		}
		sprintf(szTmp,"%10.10f\t%10.10f\n", H->w[i].s_noise, H->w[i].maxEGO);
		strcat(szMessage,szTmp);

		fprintf(pFile,"%s",szMessage);
	}
	fclose(pFile);
}

void dumpParams(history *H, int n, int inJobArrayIndex, int inActualD){
	pdebug(__FILE__,__LINE__,__FUNCTION__,"");

//	int inActualD = H->X->size2;	//Only the view is modified, not the actual matrix in the history structure.

	char szMessage[500];
	char szTmp[200];
	memset(szMessage,0x00,sizeof(szMessage));
	memset(szTmp,0x00,sizeof(szTmp));

	sprintf(szTmp,"paramsLog/params_%dD_%05d.dat",D,inJobArrayIndex);
	FILE * pFile;
	if ((pFile = fopen(szTmp,"w"))==NULL) {
	        printf("Cannot open file.\n");
        	exit(1);
      	}

	printf("******Starting cycle*********\n");
	for(int i=0; i<n; i++){
//		if(gsl_vector_get(H->usePoint,i)!=1)
//			continue;
		memset(szMessage,0x00,sizeof(szMessage));
		memset(szTmp,0x00,sizeof(szTmp));
		sprintf(szTmp,"%10.20f\t",H->w[i].s_f);
		strcat(szMessage,szTmp);
		for(int d=0; d<inActualD; d++){
			sprintf(szTmp,"%10.20f\t",H->w[i].lengthScale[d]);
			strcat(szMessage,szTmp);
		}
		sprintf(szTmp,"%10.20f\n",H->w[i].s_noise);
		strcat(szMessage,szTmp);
		fprintf(pFile,"%s",szMessage);
	}
	fclose(pFile);
}

bool outOfBounds(const gsl_vector *pGenotype){
	for(int d=0; d<D; d++)
		if(gsl_vector_get(pGenotype,d) > MAX_COORD || gsl_vector_get(pGenotype,d) < MIN_COORD)
			return true;

	return false;
}

void plotLandscape(int inEpoch, int inJobArrayIndex, int inStrategy){

	#define POINTS 2000
	#define SQ_POINTS 10000
	double delta=(MAX_COORD-MIN_COORD)/POINTS;
	double landscape[SQ_POINTS][D+1];
	double gen[D];
	if(D==2){
		for(int i=0; i<POINTS; i++){
			for(int j=0; j<POINTS; j++){
				landscape[i*POINTS+j][0]=i*delta;
				landscape[i*POINTS+j][1]=j*delta;
				gen[0]=landscape[i*POINTS+j][0];
				gen[1]=landscape[i*POINTS+j][1];
				landscape[i*POINTS+j][2]=dummy_eval(gen);
				//printf("i=%d\tj=%d\ti*POINTS+j=[%d]\ti*delta=%f\tj*delta=%f\n",i,j,i*POINTS+j,i*delta,j*delta);
			}
		}
	}
	if(D==1){
		for(int i=0; i<POINTS; i++){
			landscape[i][0]=i*delta;
			gen[0]=landscape[i][0];
			landscape[i][1]=dummy_eval(gen);
		}
	}

	char szTmp[200];
	memset(szTmp,0x00,sizeof(szTmp));
	sprintf(szTmp,"histLog/landscapes/landscape_%dD_%05d_%d_%0d.dat",D,inJobArrayIndex,inStrategy,inEpoch);

	FILE * pFile;

	if ((pFile = fopen(szTmp,"w"))==NULL) {
	        printf("Cannot open file.\n");
        	exit(1);
      	}
	if(D==2){
		for(int i=0; i<SQ_POINTS; i++){
			fprintf(pFile,"%f\t%f\t%f\n",landscape[i][0],landscape[i][1],landscape[i][2]);
		}
	}
	if(D==1){
		for(int i=0; i<POINTS; i++){
			fprintf(pFile,"%f\t%f\n",landscape[i][0],landscape[i][1]);
		}
	}


	fclose(pFile);
	
}
void printFoundParametersList(kernelParams *W, int inActualD){

	printf("\nList of best %d parameters found according to local searches of MLE:\n",LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD);
	printf("\ts_f\t\ts_noise\t\tdetK\t\tlogLike\t\t\tmaxEGO\t\t");
	for(int d=0;d<inActualD;d++)
		printf("lengthscale(%d)\t\t",d);
	int i=0;
	while(W[i].logLike != 0 && i<LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS*inActualD){
		printf("\n[%d]\t%f\t%f\t%f\t%5.20f\t%f\t",i,W[i].s_f,W[i].s_noise,W[i].detK,W[i].logLike,W[i].maxEGO);
		for(int d=0;d<inActualD;d++)
			printf("%f\t",W[i].lengthScale[d]);
		i++;
	}
	printf("\n");
}

double dblGetMiddleOfLargestGap(gsl_matrix *X, int d){

	int N = X->size1 + 2;
//	printf("Starting for dimension %d\tN=%d\n",d,N);

	// Order the data
	gsl_vector *orderedX = gsl_vector_calloc(N);


	gsl_vector_set(orderedX,0,MIN_COORD);
//	printf("orderedX[0]=%f\n",gsl_vector_get(orderedX,0));
	for(int i=1; i<N-1; i++){
		gsl_vector_set(orderedX,i,gsl_matrix_get(X,i-1,d));
//		printf("orderedX[%d]=%f\n",i,gsl_vector_get(orderedX,i));
	}
	gsl_vector_set(orderedX,N-1,MAX_COORD);
//	printf("orderedX[N-1]=%f\n",gsl_vector_get(orderedX,N-1));

	gsl_sort_vector (orderedX);

/*	printf("orderedX\n");
	for(int i=0; i<N; i++){
		printf("[%d]\t%f\n",i,gsl_vector_get(orderedX,i));
	}
*/
	// Select the maximum difference (gap)
	double maxGap = 0;
	int maxGapIndex = 0;
	for(int i=0; i<N-1; i++){
		if((gsl_vector_get(orderedX,i+1) - gsl_vector_get(orderedX,i)) > maxGap){
			maxGap = (gsl_vector_get(orderedX,i+1) - gsl_vector_get(orderedX,i));
			maxGapIndex = i;
		}
	}

	double middleCoord = gsl_vector_get(orderedX,maxGapIndex) + ( (gsl_vector_get(orderedX,maxGapIndex+1) - gsl_vector_get(orderedX,maxGapIndex)) / 2 );
	gsl_vector_free(orderedX);

	return middleCoord;

}


void plotGpRegFunction(void *Data){
	/* Restore the data to the original form */
	gsl_matrix **Params = (gsl_matrix **)Data;
	gsl_matrix *X = Params[0];
	gsl_matrix *invKY = Params[1];
//	gsl_matrix *invK = Params[2];
	gsl_matrix *scale = Params[3];
	gsl_matrix *y = Params[4];
//	gsl_matrix *Tabu = Params[5];
	respSurf *prevGP = (respSurf*)Params[6];


	int inSampleSize = X->size1;
	int inActualD = X->size2;
	
	gsl_vector *k_new = gsl_vector_alloc (inSampleSize);
//	gsl_vector *invK_kNew = gsl_vector_alloc (inSampleSize);

	double inStrategy = gsl_matrix_get(scale,0,0);
	double timeScale = gsl_matrix_get(scale,1,0);
	if(inStrategy == ST_TEMPORAL_DISCOUNT && timeScale != LEARN_TIMESCALE)
		inActualD++;
	double maxCov = pow( gsl_matrix_get(scale,2,0),2);
//	double kDiag = maxCov; 			//kDiag is the kernel from the new point to itself => S_n = 0 (since current epoch) and k(x,x)=1 by definition. So, kDiag = maxCov;

	
	/* Evaluate the kernel at each point */
	#define POINTS 2000
	double landscape[POINTS][D+1];
	double delta=(MAX_COORD-MIN_COORD)/POINTS;
	gsl_vector *pGenotype = gsl_vector_alloc (D);

	for(int i=0; i<POINTS; i++){
		landscape[i][0]=i*delta;
		gsl_vector_set(pGenotype,0,landscape[i][0]);


		/* Calculate the kernel from this test point (current genotype) to each sampled point */
		for(int j=0; j<inSampleSize; j++){
			double sum = 0.0;
	
			for(int d=0; d<D; d++){
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

			YinvKY += priorMean(prevGP,pGenotype);	//calculate the value at pGenotype according to the previous model, then remove (or add???) this value from YinvKY (which is the predicted mean).

	/*
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
	*/

		landscape[i][1]=YinvKY;
	}
	gsl_vector_free(k_new);
//	gsl_vector_free(invK_kNew);


	char szTmp[200];
	memset(szTmp,0x00,sizeof(szTmp));
	sprintf(szTmp,"histLog/landscapes/landscape_%dD_%05d_%d_%0d_reg.dat",D,gInJobArrayIndex,(int)inStrategy,gInEpoch);

	FILE * pFile;

	if ((pFile = fopen(szTmp,"w"))==NULL) {
	        printf("Cannot open file.\n");
        	exit(1);
      	}
	for(int i=0; i<POINTS; i++){
		fprintf(pFile,"%f\t%f\n",landscape[i][0],landscape[i][1]);
	}
	fclose(pFile);


	return;
}
