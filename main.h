/* Tracking global optima in 
* dynamic environments with efficient 
* global optimization --- 2010-2014 
*/

/* main.h
 * Copyright (C) 2014 Sergio Morales.
 * This is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sort_float.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "movpeaks.h"


#define D 		2
//#define INIT_SAMPLES	4	//Specifies the number of random (uniformly distributed) initial samples to take in order to provide some support to start building the response surfaces. 					For strategies with memory (all except ST_RESET), the random initial sampling takes place only for the first epoch. At the begining of all the other 					epochs, the previous best known sample is used (as if no change had happened).
#define INIT_RAND_SAMPLES_PER_EPOCH	//Defines how many total initial samples, at the beginning of each epoch, should be taken. If "best from previous epoch" is already being used, 					then only INIT_RAND_SAMPLES_PER_EPOCH - 1 random samples will be taken. 
#define EPOCHS_IN_MEM	2	//Including current epoch (ex: 2 means current epoch plus 1 before)
#define LINEAR_NOISE	"linNoise"	//Constant times age

//#define EXPLOIT_AFTER	5	//Stop exploring and start exploiting after this number of repeated samples.

#define MIN_COORD	0.0
#define MAX_COORD	100.0
#define MIN_HEIGHT	0.0
#define MAX_HEIGHT	100.0
#define MAX_LIKELIHOOD_PARAM_SEARCH	30.0
//#define EVOLVE_MOV_PEAKS	100


/* Cow requirements: save stuff in: /storage/physics/phrjac */
#define PATH_IN		"parms_"
#define PATH_OUT	"out_linNoise_sqExpT_"
#define PI 3.14159265358979323846


//#define LOG_MLE
//#define LOG_MIN
//#define LOG_MAX
//#define LOG_SAMPLES
//#define DEBUG
//#define SLOW
//#define TRACK_EGO
#define LOG_ERRORS_2_FILE

#define GSL_MINIMIZER_LIKELIHOOD_PRECISION	1e-3
#define GSL_MINIMIZER_GPREG_PRECISION		1e-5
#define EPSILON_SAMPLING_SHIFT 			1e-1
#define EPSILON_BUBBLE_RADIOUS 			5e-2
#define MIN_TRUSTED_DET_K			1.0e-2
//extern double EPSILON_BUBBLE_RADIOUS = 1e-1;
/*
#define EGO_RESOLUTION					1000
#define LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS	20
#define LOGLIKE_MAXIMIZATION_MAX_ITERATIONS		200
*/
#define EGO_RESOLUTION					1000
#define LOGLIKE_MAXIMIZATION_ROBUSTNESS_ITERATIONS	20
#define LOGLIKE_MAXIMIZATION_MAX_ITERATIONS		200

//Sampling strategies
#define ST_RESET		0	//Complete reset (re-sample randomly after each change)
#define ST_IGNORE		1	//Ignore the changes (use points from old epochs, considering them as reliable as new ones)
#define ST_INCREASE_NOISE	2	//Discount old points using measurement noise
#define ST_TEMPORAL_DISCOUNT	3	//Learn how much old points affect new ones, through a time scale (time as additional dimension)
#define ST_RANDOM		4	//Generate random sampling points, uniformly accross each dimension and independently in each dimension.
#define ST_RESET_STAR		5 	//Reset strategy, but after a change, use the previous' epoch best sample as first sample of the current epoch (and re-use the latest parameters to build the first 						model of the current epoch)
#define ST_CONST_MEAN_PRIOR	6	//Take random initial samples to start building the model. If parameter==-1, take their mean and use it as a constant mean prior for the GP (instead of zero). 						Otherwise, use the parameter value as a constant mean.
#define ST_PREV_SURF_MEAN_PRIOR	7	//Use the response surface generated at the end of the previous epoch as a prior mean. For the first sample of the new epoch, re-use the old parameters.
//#define ST_RESET_STAR_RAND	8 	//Reset strategy, but after a change, use the previous' epoch best sample as first sample of the current epoch. Unlike ST_RESET_STAR, the next INIT_PARAMS-1 are sampled 						randomly, to provide a more fair comparison with RESET
//#define ST_COKRIGING		8	//Keep the GP model of the previous epoch, and use it to make predictions. Then use such a prediction as an input for the new model...

#define EGO			-1.0
#define EGO_PROB		-2.0
#define LEARN_TIMESCALE		-3.0
#define LK_DIVERGENCE_RO_BETA	-4.0
#define EGO_EXP_IMPROV_STOP	0.01
#define EGO_MIN_PROB_IMPROV_STOP 0.05
#define USE_POINT		-1
#define DISCARD_POINT		-2
//#define MIN_LENGTHSCALE		0.0001	//Stupid idea
#define NO_EXPLOIT_AFTER		//When not defined, the "Exploit after" parameter is not used. "Exploit after" is used as a stopping criteria (to stop exploring and start exploiting). Whenever the "Exploit after" counter reaches the "Exploit after parameter", 							all further samples for the current epoch are taken at the best known solution. The "Exploit after" counter is increased by one every time a new sample is taken too close from an existing one, or it is increased to the 							maximum if the K matrix starts to become singular according to some arbitrary measure on the determinant being close to zero (under MIN_TRUSTED_DET_K), stopping any further exploration and focusing on exploitation.
					//When NO_EXPLOIT_AFTER IS defined, the exploration carries on indefinitely. This provides a much better OFFLINE ERROR, but a worse AVERAGE ERROR. Besides, it is significantly more slow since the Gaussian process is built from 							many more samples.

#define max(A,B) ((A)>(B) ? (A) : (B))

struct kernelParams {
	double s_f;
	double s_noise;
	double detK;
	double logLike;
	double maxEGO;
	double lengthScale[D+1];	//To hold the additional dimension of time in case it is needed.
};
struct respSurf {		//Holds sufficient information to define a response surface (GP)
	int epochId;
	float constScalarMean;	//Used for ST_CONST_MEAN_PRIOR (including the first epoch for ST_PREV_SURF_MEAN_PRIOR)
	gsl_matrix *X;
	gsl_matrix *y;
	kernelParams *w;
	gsl_matrix *invKY;
	respSurf *meanPr;	//Recurrent definition. 
};				
struct history {
	int maxN;		//Maximum number of samples to take
	int currPeriod;		
	int inInitSamples;	//Specifies the number of random (uniformly distributed) initial samples to take in order to provide some support to start building the response surfaces.
	double *maxVals;	//Maximum function value known up to each time
	double constParam;	//Constant parameter used to hold either the time scale value (when avoiding learning) or the noise level to be added, or the constant mean prior.
	gsl_matrix *X;		//Points where samples are taken (in D dimensions)
	gsl_matrix *y;		//Response obtained for each X
	kernelParams *w;	//History of parameters used (to track evolution and analyze convergence)
	gsl_vector *usePoint;	//-1 Use point; -2 don't use point; >= 0 index to the point (in the History->X matrix) defining the end of a tabu region started at this point (where expected improvement shall 						be zero). Repeated points (or too close to each other) create singularities. Removing the worse point naively might lead to a situation where the algorithm resamples the same point for ever.
	respSurf *prevGP;
};


//Only used for plotGpRegFunction
extern int gInJobArrayIndex;
extern int gInEpoch;


extern int gPeakChangeFreq;
extern unsigned long int movrandseed; /* seed for built-in random number generator */

extern gsl_rng *initRandGen(double seed);
extern double peaks(double x, double y);
extern int maxKernelMLE(gsl_matrix *X, gsl_matrix *y, kernelParams *wOut,  gsl_rng *r, int inStrategy, float constParam);
extern double linSpace(double lBound, double uBound, int gridSize, gsl_vector *v);
extern void getKernelMatrix(gsl_matrix *X, gsl_vector *y, kernelParams w, gsl_matrix *testM, gsl_vector *gpMean, gsl_vector *gpStd);
extern void getMax(double *maxVal, int *maxIndex, gsl_vector *gpMean);
extern double get_current_error();
extern double get_avg_error();
extern double get_offline_error();
extern double get_current_maximum();
extern double get_global_max();
extern void printMatrix(gsl_matrix *out);
extern void printVector(gsl_vector *out, int start, int end);
extern double global_max;
extern bool maxGPReg(gsl_matrix *X, gsl_matrix *y, kernelParams *w, double *bestGenotype, double k, gsl_rng *r, double lastMax, int inStrategy, float constParam, gsl_matrix *Tabu, respSurf *prevGP);
extern double dummy_eval (double *gen);
extern double peak_function1(double *gen, int peak_number);
extern double gsl_linalg_SV_invert(gsl_matrix *U, gsl_matrix *V, gsl_vector *S, gsl_matrix *pseudoInvU);
extern bool getNextSample(history *H, gsl_rng *r, int n, double *bestNextGenotype, int inStrategy, double kSigma);
extern void printXYMatrices(const char* szMessage, gsl_matrix *x, gsl_matrix *y, int fromRow, int toRow);
extern void pdebug(const char* file, int line, const char* func,const char* szMessage);
extern double logLikelihood(const gsl_vector *pw, void *Data);
void dumpLog(history *H, int n, int inJobArrayIndex,int inStrategy, double vLen, double hSev);
extern void dumpParams(history *H, int n, int inJobArrayIndex, int inActualD);
extern bool outOfBounds(const gsl_vector *pGenotype);
extern void change_peaks();
extern void free_peaks();
extern void drasticChange_peaks();
extern double movrand ();
extern double movnrand ();
extern double getBiasedRandMove();
void plotLandscape(int inEpoch, int inJobArrayIndex, int inStrategy);
extern void printFoundParametersList(kernelParams *W, int inActualD);
extern double getRMSE(gsl_matrix *X, gsl_matrix *y, kernelParams *W, float constParam, int inStrategy);
extern double dblGetMiddleOfLargestGap(gsl_matrix *X, int d);
extern int inDivergeCount;
extern void getPreviousBest(double *bestNextGenotype, history *H, int n, int gPeakChangeFreq);
extern int getViewCount(history *H, int startN, int endN, int *inTabuCount);
extern int getView(history *H, int n, int startN, int endN, gsl_matrix *vX, gsl_matrix *vy, int inStrategy, gsl_matrix *vTabu);
extern void vdGetPrevGP(gsl_vector *vPrM, respSurf *prevGP, gsl_matrix *vX, gsl_matrix *vy, double constParam, int inInitSamples, int inStrategy);
extern void vdRemoveMean(gsl_vector *vPrM, gsl_matrix *vy);
extern int get_right_peak();
extern double get_w(int inPeak);
extern int get_maximum_peak();
extern double priorMean(respSurf *prevGP, const gsl_vector *pGenotype);
extern void plotGpRegFunction(void *Data);
