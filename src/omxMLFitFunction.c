 /*
 *  Copyright 2007-2012 The OpenMx Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "omxAlgebraFunctions.h"
#include "omxExpectation.h"
#include "omxMLFitFunction.h"
#include "omxFIMLFitFunction.h"
#include "omxRAMExpectation.h"

void omxDestroyMLFitFunction(omxFitFunction *oo) {

	if(OMX_DEBUG) {Rprintf("Freeing ML Fit Function.");}
	omxMLFitFunction* omlo = ((omxMLFitFunction*)oo->argStruct);

	if(omlo->localCov != NULL)	omxFreeMatrixData(omlo->localCov);
	if(omlo->localProd != NULL)	omxFreeMatrixData(omlo->localProd);
	if(omlo->P != NULL)			omxFreeMatrixData(omlo->P);
	if(omlo->C != NULL)			omxFreeMatrixData(omlo->C);
	if(omlo->I != NULL)			omxFreeMatrixData(omlo->I);
}

omxRListElement* omxSetFinalReturnsMLFitFunction(omxFitFunction *oo, int *numReturns) {
	*numReturns = 3;
	omxRListElement* retVal = (omxRListElement*) R_alloc(*numReturns, sizeof(omxRListElement));
	double det = 0.0;
	omxMatrix* cov = ((omxMLFitFunction*)oo->argStruct)->observedCov;
	int ncols = ((omxMLFitFunction*)oo->argStruct)->observedCov->cols;
    
	retVal[0].numValues = 1;
	retVal[0].values = (double*) R_alloc(1, sizeof(double));
	strncpy(retVal[0].label, "Minus2LogLikelihood", 20);
	retVal[0].values[0] = omxMatrixElement(oo->matrix, 0, 0);

	retVal[1].numValues = 1;
	retVal[1].values = (double*) R_alloc(1, sizeof(double));
	strncpy(retVal[1].label, "SaturatedLikelihood", 20);
	retVal[1].values[0] = (((omxMLFitFunction*)oo->argStruct)->logDetObserved + ncols) * (((omxMLFitFunction*)oo->argStruct)->n - 1);

	retVal[2].numValues = 1;
	retVal[2].values = (double*) R_alloc(1, sizeof(double));
	strncpy(retVal[2].label, "IndependenceLikelihood", 23);
	// Independence model assumes all-zero manifest covariances.
	// (det(expected) + tr(observed * expected^-1)) * (n - 1);
	// expected is the diagonal of the observed.  Inverse expected is 1/each diagonal value.
	// Therefore the diagonal elements of observed * expected^-1 are each 1.
	// So the trace of the matrix is the same as the number of columns.
	// The determinant of a diagonal matrix is the product of the diagonal elements.
	// Since these are the same in the expected as in the observed, we can get 'em direct.
	for(int i = 0; i < ncols; i++) {
		// We sum logs instead of logging the product.
		det += log(omxMatrixElement(cov, i, i));
	}
	if(OMX_DEBUG) { Rprintf("det: %f, tr: %f, n= %d, total:%f\n", det, ncols, ((omxMLFitFunction*)oo->argStruct)->n, (ncols + det) * (((omxMLFitFunction*)oo->argStruct)->n - 1)); }
	if(OMX_DEBUG) { omxPrint(cov, "Observed:"); }
	retVal[2].values[0] = (ncols + det) * (((omxMLFitFunction*)oo->argStruct)->n - 1);

	return retVal;
}

void omxCallMLFitFunction(omxFitFunction *oo) {	// TODO: Figure out how to give access to other per-iteration structures.

	if(OMX_DEBUG) { Rprintf("Beginning ML Evaluation.\n");}
	// Requires: Data, means, covariances.

	double sum = 0.0, det = 0.0;
	char u = 'U';
	char r = 'R';
	int info = 0;
	double oned = 1.0;
	double zerod = 0.0;
	double minusoned = -1.0;
	int onei = 1;
	double fmean = 0.0;

	omxMatrix *scov, *smeans, *cov, *means, *localCov, *localProd, *P, *C;

	omxMLFitFunction *omo = ((omxMLFitFunction*)oo->argStruct);
	
    /* Locals for readability.  Compiler should cut through this. */
	scov 		= omo->observedCov;
	smeans		= omo->observedMeans;
	cov			= omo->expectedCov;
	means 		= omo->expectedMeans;
	localCov 	= omo->localCov;
	localProd 	= omo->localProd;
	P		 	= omo->P;
	C		 	= omo->C;
	double n 	= omo->n;
	double Q	= omo->logDetObserved;
	omxExpectation* expectation = oo->expectation;

    /* Recompute and recopy */
	omxExpectationCompute(expectation);

	omxCopyMatrix(localCov, cov);				// But expected cov is destroyed in inversion

	if(OMX_DEBUG_ALGEBRA) {
		omxPrint(scov, "Observed Covariance is");
		omxPrint(localCov, "Implied Covariance Is");
		omxPrint(cov, "Original Covariance Is");
	}

	/* Calculate |expected| */

//	F77_CALL(dgetrf)(&(localCov->cols), &(localCov->rows), localCov->data, &(localCov->cols), ipiv, &info);
	F77_CALL(dpotrf)(&u, &(localCov->cols), localCov->data, &(localCov->cols), &info);

	if(OMX_DEBUG_ALGEBRA) { Rprintf("Info on LU Decomp: %d\n", info);}
	if(info > 0) {
		char *errstr = calloc(250, sizeof(char));
		sprintf(errstr, "Expected covariance matrix is non-positive-definite");
		if(oo->matrix->currentState->computeCount <= 0) {
			strncat(errstr, " at starting values", 20);
		}
		strncat(errstr, ".\n", 3);
		omxRaiseError(oo->matrix->currentState, -1, errstr);						// Raise error
		free(errstr);
		return;																		// Leave output untouched
	}

	//det = log(det)	// TVO: changed multiplication of det to first log and the summing up; this line should just set det to zero.
	for(info = 0; info < localCov->cols; info++) { 	    	// |cov| is the square of the product of the diagonal elements of U from the LU factorization.
		det += log(fabs(localCov->data[info+localCov->rows*info])); // TVO: changed * to + and added fabs command
	}
	det *= 2.0;		// TVO: instead of det *= det;

	if(OMX_DEBUG_ALGEBRA) { Rprintf("Determinant of Expected Cov: %f\n", exp(det)); }
	// TVO: removed det = log(fabs(det))
	if(OMX_DEBUG_ALGEBRA) { Rprintf("Log of Determinant of Expected Cov: %f\n", det); }

	/* Calculate Expected^(-1) */
//	F77_CALL(dgetri)(&(localCov->rows), localCov->data, &(localCov->cols), ipiv, work, lwork, &info);
	F77_CALL(dpotri)(&u, &(localCov->rows), localCov->data, &(localCov->cols), &info);
	if(OMX_DEBUG_ALGEBRA) { Rprintf("Info on Invert: %d\n", info); }

	if(OMX_DEBUG_ALGEBRA) {omxPrint(cov, "Expected Covariance Matrix:");}
	if(OMX_DEBUG_ALGEBRA) {omxPrint(localCov, "Inverted Matrix:");}

	/* Calculate C = Observed * expected^(-1) */

	if(OMX_DEBUG_ALGEBRA) {Rprintf("Call is: DSYMM(%d, %d, %f, %0x, %d, %0x, %d, %f, %0x, %d)",
					(scov->rows), (localCov->cols), oned, scov->data, (localCov->leading),
					localCov->data, (localCov->leading), zerod, localProd->data, (localProd->leading));}


	// Stop gcc from issuing a warning
	int majority = *(scov->majority) == 'n' ? scov->rows : scov->cols;

	/*  TODO:  Make sure leading edges are being appropriately calculated, and sub them back into this */
	F77_CALL(dsymm)(&r, &u, &(localCov->rows), &(scov->cols),
					&oned, localCov->data, &(majority),
 					scov->data, &(majority),
					&zerod, localProd->data, &(localProd->leading));

    /* And get the trace of the result */

	for(info = 0; info < localCov->cols; info++) {
		sum += localProd->data[info*localCov->cols + info];
	}

//	for(info = 0; info < (localCov->cols * localCov->rows); info++) {
//		sum += localCov->data[info] * scov->data[info];
//	}

	if(OMX_DEBUG_ALGEBRA) {omxPrint(scov, "Observed Covariance Matrix:");}
	if(OMX_DEBUG_ALGEBRA) {omxPrint(localCov, "Inverse Matrix:");}
	if(OMX_DEBUG_ALGEBRA) {omxPrint(localProd, "Product Matrix:");}

	if(means != NULL) {
		if(OMX_DEBUG_ALGEBRA) { Rprintf("Means Likelihood Calculation"); }
		omxRecompute(means);
		omxCopyMatrix(P, means);
		// P = means - smeans
		if(OMX_DEBUG_ALGEBRA) {omxPrint(means, "means");}
		if(OMX_DEBUG_ALGEBRA) {omxPrint(smeans, "smeans");}
		F77_CALL(daxpy)(&(smeans->cols), &minusoned, smeans->data, &onei, P->data, &onei);
		if(OMX_DEBUG_ALGEBRA) {omxPrint(P, "means - smeans");}
		// C = P * Cov
		F77_CALL(dsymv)(&u, &(localCov->rows), &oned, localCov->data, &(localCov->leading), P->data, &onei, &zerod, C->data, &onei);
		// P = C * P'
		fmean = F77_CALL(ddot)(&(C->cols), P->data, &onei, C->data, &onei);

		if(OMX_DEBUG_ALGEBRA) { Rprintf("Mean contribution to likelihood is %f per row.\n", fmean); }
		if(fmean < 0.0) fmean = 0.0;
	}

	oo->matrix->data[0] = (sum + det) * (n - 1) + fmean * (n);

	if(OMX_DEBUG) { Rprintf("MLFitFunction value comes to: %f (Chisq: %f).\n", oo->matrix->data[0], (sum + det) - Q - cov->cols); }

}

void omxPopulateMLAttributes(omxFitFunction *oo, SEXP algebra) {
    if(OMX_DEBUG) { Rprintf("Populating ML Attributes.\n"); }

	omxMLFitFunction *argStruct = ((omxMLFitFunction*)oo->argStruct);
	omxMatrix *expCovInt = argStruct->expectedCov;	    		// Expected covariance
	omxMatrix *expMeanInt = argStruct->expectedMeans;			// Expected means

	SEXP expCovExt, expMeanExt, gradients;
	PROTECT(expCovExt = allocMatrix(REALSXP, expCovInt->rows, expCovInt->cols));
	for(int row = 0; row < expCovInt->rows; row++)
		for(int col = 0; col < expCovInt->cols; col++)
			REAL(expCovExt)[col * expCovInt->rows + row] =
				omxMatrixElement(expCovInt, row, col);

	if (expMeanInt != NULL) {
		PROTECT(expMeanExt = allocMatrix(REALSXP, expMeanInt->rows, expMeanInt->cols));
		for(int row = 0; row < expMeanInt->rows; row++)
			for(int col = 0; col < expMeanInt->cols; col++)
				REAL(expMeanExt)[col * expMeanInt->rows + row] =
					omxMatrixElement(expMeanInt, row, col);
	} else {
		PROTECT(expMeanExt = allocMatrix(REALSXP, 0, 0));		
	}   
	
	if(oo->gradientFun != NULL) {
        int nLocs = oo->matrix->currentState->numFreeParams;
        double gradient[oo->matrix->currentState->numFreeParams];
        for(int loc = 0; loc < nLocs; loc++) {
            gradient[loc] = NA_REAL;
        }
        oo->gradientFun(oo, gradient);
        PROTECT(gradients = allocMatrix(REALSXP, 1, nLocs));

    	for(int loc = 0; loc < nLocs; loc++)
            REAL(gradients)[loc] = gradient[loc];
    } else {
        PROTECT(gradients = allocMatrix(REALSXP, 0, 0));
    }
    
	setAttrib(algebra, install("expCov"), expCovExt);
	setAttrib(algebra, install("expMean"), expMeanExt);
	setAttrib(algebra, install("gradients"), gradients);
	
	UNPROTECT(3);

}

void omxSetMLFitFunctionCalls(omxFitFunction* oo) {
	
	/* Set FitFunction Calls to ML FitFunction Calls */
	oo->fitType = "omxMLFitFunction";
	oo->computeFun = omxCallMLFitFunction;
	oo->destructFun = omxDestroyMLFitFunction;
	oo->setFinalReturns = omxSetFinalReturnsMLFitFunction;
	oo->populateAttrFun = omxPopulateMLAttributes;
	oo->repopulateFun = NULL;	
}


void omxInitMLFitFunction(omxFitFunction* oo, SEXP rObj) {

	if(OMX_DEBUG) { Rprintf("Initializing ML fit function.\n"); }

	int info = 0;
	double det = 1.0;
	char u = 'U';
	
	/* Read and set expectation */
	omxSetMLFitFunctionCalls(oo);

	omxData* dataMat = oo->expectation->data;

	if(!(dataMat == NULL) && strncmp(omxDataType(dataMat), "cov", 3) != 0 && strncmp(omxDataType(dataMat), "cor", 3) != 0) {
		if(strncmp(omxDataType(dataMat), "raw", 3) == 0) {
			if(OMX_DEBUG) { Rprintf("Raw Data: Converting to FIML.\n"); }
			omxInitFIMLFitFunction(oo, rObj);
			return;
		}
		char *errstr = calloc(250, sizeof(char));
		sprintf(errstr, "ML FitFunction unable to handle data type %s.\n", omxDataType(dataMat));
		omxRaiseError(oo->matrix->currentState, -1, errstr);
		free(errstr);
		if(OMX_DEBUG) { Rprintf("ML FitFunction unable to handle data type %s.  Aborting.\n", omxDataType(dataMat)); }
		return;
	}

	omxMLFitFunction *newObj = (omxMLFitFunction*) R_alloc(1, sizeof(omxMLFitFunction));

	if(OMX_DEBUG) { Rprintf("Processing Observed Covariance.\n"); }
	newObj->observedCov = omxDataMatrix(dataMat, NULL);
	if(OMX_DEBUG) { Rprintf("Processing Observed Means.\n"); }
	newObj->observedMeans = omxDataMeans(dataMat, NULL, NULL);
	if(OMX_DEBUG && newObj->observedMeans == NULL) { Rprintf("ML: No Observed Means.\n"); }
	if(OMX_DEBUG) { Rprintf("Processing n.\n"); }
	newObj->n = omxDataNumObs(dataMat);

	newObj->expectedCov = omxGetExpectationComponent(oo->expectation, oo, "cov");
	newObj->expectedMeans = omxGetExpectationComponent(oo->expectation, oo, "means");

	if(newObj->expectedCov == NULL) {
		omxRaiseError(oo->matrix->currentState, OMX_DEVELOPER_ERROR,
			"Developer Error in ML-based fit function object: ML's expectation must specify a model-implied covariance matrix.\nIf you are not developing a new expectation type, you should probably post this to the OpenMx forums.");
		return;
	}

	// Error Checking: Observed/Expected means must agree.  
	// ^ is XOR: true when one is false and the other is not.
	if((newObj->expectedMeans == NULL) ^ (newObj->observedMeans == NULL)) {
		if(newObj->expectedMeans != NULL) {
			omxRaiseError(oo->matrix->currentState, OMX_ERROR,
				"Observed means not detected, but an expected means matrix was specified.\n  If you provide observed means, you must specify a model for the means.\n");
			return;
		} else {
			omxRaiseError(oo->matrix->currentState, OMX_ERROR,
				"Observed means were provided, but an expected means matrix was not specified.\n  If you  wish to model the means, you must provide observed means.\n");
			return;	        
		}
	}

	/* Temporary storage for calculation */
	int rows = newObj->observedCov->rows;
	int cols = newObj->observedCov->cols;
	newObj->localCov = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
	newObj->localProd = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
	newObj->P = omxInitMatrix(NULL, 1, cols, TRUE, oo->matrix->currentState);
	newObj->C = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
	newObj->I = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);

	for(int i = 0; i < rows; i++) omxSetMatrixElement(newObj->I, i, i, 1.0);

	omxCopyMatrix(newObj->localCov, newObj->observedCov);

	newObj->lwork = newObj->expectedCov->rows;
	newObj->work = (double*)R_alloc(newObj->lwork, sizeof(double));

	// TODO: Determine where the saturated model computation should go.

	F77_CALL(dpotrf)(&u, &(newObj->localCov->cols), newObj->localCov->data, &(newObj->localCov->cols), &info);

	if(OMX_DEBUG) { Rprintf("Info on LU Decomp: %d\n", info); }
	if(info != 0) {
		char *errstr = calloc(250, sizeof(char));
		sprintf(errstr, "Observed Covariance Matrix is non-positive-definite.\n");
		omxRaiseError(oo->matrix->currentState, -1, errstr);
		free(errstr);
		return;
	}
	for(info = 0; info < newObj->localCov->cols; info++) {
		det *= omxMatrixElement(newObj->localCov, info, info);
	}
	det *= det;					// Product of squares.

	if(OMX_DEBUG) { Rprintf("Determinant of Observed Cov: %f\n", det); }
	newObj->logDetObserved = log(det);
	if(OMX_DEBUG) { Rprintf("Log Determinant of Observed Cov: %f\n", newObj->logDetObserved); }

	omxCopyMatrix(newObj->localCov, newObj->expectedCov);
    oo->argStruct = (void*)newObj;

}

void omxSetMLFitFunctionGradient(omxFitFunction* oo, void (*derivativeFun)(omxFitFunction*, double*)) {
    if(strncmp("omxMLFitFunction", oo->fitType, 16)) {
        char errorstr[250];
        sprintf(errorstr, "PROGRAMMER ERROR: Using vanilla-ML gradient with FitFunction of type %s", oo->fitType);
        omxRaiseError(oo->matrix->currentState, -2, errorstr);
        return;
    }
    
    if(derivativeFun == NULL) {
        char errorstr[250];
        sprintf(errorstr, "Programmer error: ML gradient given NULL gradient function.");
        omxRaiseError(oo->matrix->currentState, -2, errorstr);
        return;
    }
    
    oo->gradientFun = derivativeFun;
}

void omxSetMLFitFunctionGradientComponents(omxFitFunction* oo, void (*derivativeFun)(omxFitFunction*, omxMatrix**, omxMatrix**, int*)) {
    if(OMX_DEBUG) { Rprintf("Setting up gradient component function for ML FitFunction."); }
    if(!strncmp("omxFIMLFitFunction", oo->fitType, 16)) {
        if(OMX_DEBUG) { Rprintf("FIML FitFunction gradients not yet implemented. Skipping."); }
        return; // ERROR:NYI.
    }
    
    if(derivativeFun == NULL) {
        char errorstr[250];
        sprintf(errorstr, "Programmer error: ML gradient components given NULL gradient function.");
        omxRaiseError(oo->matrix->currentState, -2, errorstr);
        return;
    }
    
    omxMLFitFunction *omo = ((omxMLFitFunction*) oo->argStruct);
    int rows = omo->observedCov->rows;
    int cols = omo->observedCov->cols;
    int nFreeVars = oo->matrix->currentState->numFreeParams;
            
    omo->derivativeFun = derivativeFun;
    omo->X  = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
    omo->Y  = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
    omo->Ms = omxInitMatrix(NULL, 1, cols, TRUE, oo->matrix->currentState);
    omo->Mu = omxInitMatrix(NULL, 1, cols, TRUE, oo->matrix->currentState);
    omo->dSigma = (omxMatrix**) R_alloc(nFreeVars, sizeof(omxMatrix*));
    omo->dMu = (omxMatrix**) R_alloc(nFreeVars, sizeof(omxMatrix*));
    for(int i = 0; i < nFreeVars; i++) {
        omo->dSigma[i] = omxInitMatrix(NULL, rows, cols, TRUE, oo->matrix->currentState);
        omo->dMu[i] = omxInitMatrix(NULL, rows, 1, TRUE, oo->matrix->currentState);
    }
    oo->gradientFun = omxCalculateMLGradient;
}

void omxCalculateMLGradient(omxFitFunction* oo, double* gradient) {

    if(OMX_DEBUG) { Rprintf("Beginning ML Gradient Calculation.\n"); }
    // Rprintf("Beginning ML Gradient Calculation, Iteration %d.%d (%d)\n", 
        // oo->matrix->currentState->majorIteration, oo->matrix->currentState->minorIteration,
        // oo->matrix->currentState->computeCount); //:::DEBUG:::
    // 1) Calculate current Expected Covariance
    // 2) Calculate eCov, the Inverse Expected Covariance matrix 
    // 3) Calculate C = I - eCov D, where D is the observed covariance matrix
    // 4) Calculate b = M - [observed M]
    // 5) For each location in locs:
    //   gradient[loc] = tr(eCov^-1 %*% dEdt %*% C) - (b^T %*% eCov^-1 %*% dEdt + 2 dMdt^T))eCov^-1 b)

    omxMLFitFunction *omo = ((omxMLFitFunction*)oo->argStruct);
    
    /* Locals for readability.  Compiler should cut through this. */
    omxMatrix *scov         = omo->observedCov;
    omxMatrix *smeans       = omo->observedMeans;
    omxMatrix *cov          = omo->expectedCov;
    omxMatrix *M            = omo->expectedMeans;
    omxMatrix *eCov         = omo->localCov;        // TODO: Maybe need to avoid reusing these
    omxMatrix *I            = omo->I;
    omxMatrix *C            = omo->C;
    omxMatrix *X            = omo->X;
    omxMatrix *Y            = omo->Y;
    omxMatrix *Mu           = omo->Mu;
    omxMatrix *Ms           = omo->Ms;
    omxMatrix *P            = omo->P;
    double n                = omo->n;
    omxMatrix** dSigmas     = omo->dSigma;
    omxMatrix** dMus        = omo->dMu;
    
    int gradientSize = oo->matrix->currentState->numFreeParams;
    
    char u = 'U';
    int info;
    double minusoned = -1.0;
    int onei = 1;
    int status[gradientSize];
    int nLocs = oo->matrix->currentState->numFreeParams;
    
    // Calculate current FitFunction values
    // We can safely assume this has been done
    // omxFitFunctionCompute(oo);
    
    // Calculate current eCov
    
    omxCopyMatrix(eCov, cov);				// But expected cov is destroyed in inversion
    
    F77_CALL(dpotrf)(&u, &(eCov->cols), eCov->data, &(eCov->cols), &info);

    if(OMX_DEBUG_ALGEBRA) { Rprintf("Info on LU Decomp: %d\n", info);}
    if(info > 0) {
        char *errstr = calloc(250, sizeof(char));
        sprintf(errstr, "Expected covariance matrix is non-positive-definite");
        if(oo->matrix->currentState->computeCount <= 0) {
            strncat(errstr, " at starting values", 20);
        }
        strncat(errstr, ".\n", 3);
        omxRaiseError(oo->matrix->currentState, -1, errstr);                        // Raise error
        free(errstr);
        return;                                                                     // Leave output untouched
    }
    
    F77_CALL(dpotri)(&u, &(eCov->rows), eCov->data, &(eCov->cols), &info);
    if(info > 0) {
        char *errstr = calloc(250, sizeof(char));
        sprintf(errstr, "Expected covariance matrix is not invertible");
        if(oo->matrix->currentState->computeCount <= 0) {
            strncat(errstr, " at starting values", 20);
        }
        strncat(errstr, ".\n", 3);
        omxRaiseError(oo->matrix->currentState, -1, errstr);                        // Raise error
        free(errstr);
        return;
    }
    // Calculate P = expected means - observed means
    if(M != NULL) {
        omxCopyMatrix(P, smeans);
    	F77_CALL(daxpy)(&(smeans->cols), &minusoned, M->data, &onei, P->data, &onei);
    }
	
	// Reset C and Calculate C = I - eCov * oCov
    omxCopyMatrix(C, I);
    omxDSYMM(TRUE, -1.0, eCov, scov, 1.0, C);
    
    // For means, calculate Ms = eCov-1 %*% P
    if(M != NULL)
        omxDSYMM(FALSE, 1.0, eCov, P, 0.0, Ms);
    
    // Calculate parameter-level derivatives
    // TODO: Parallelize Here.

    if(OMX_DEBUG)  { Rprintf("Calling component function.\n"); }
    omo->derivativeFun(oo, dSigmas, dMus, status);
    
    for(int currentLoc = 0; currentLoc < nLocs; currentLoc++) {
        double meanInfluence, covInfluence;
        if(status[currentLoc] < 0) continue;  // Failure in computation--skip.
        //   gradient[loc] = tr(eCov^-1 %*% dEdt %*% C) - 
        //    (b^T %*% eCov^-1 %*% dEdt + 2 dMdt^T))eCov^-1 b)
        // omxDGEMM(FALSE, FALSE, 1.0, dSigmas[currentLoc], C, 0.0, Y);
        omxDSYMM(TRUE, 1.0, dSigmas[currentLoc], C, 0.0, Y);
        omxDSYMM(TRUE, 1.0, eCov, Y, 0.0, X);
        gradient[currentLoc] = 0;
        covInfluence = 0.0;
        for(int i = 0; i < eCov->cols; i++) 
            covInfluence += omxMatrixElement(X, i, i);
        if(M != NULL) {
            omxCopyMatrix(Mu, dMus[currentLoc]);
            omxDSYMV(1.0, dSigmas[currentLoc], Ms, 2.0, Mu);
            meanInfluence = F77_CALL(ddot)(&(eCov->cols), Mu->data, &onei, Ms->data, &onei);
        } else {
            meanInfluence = 0;
        }
        gradient[currentLoc] = (covInfluence * (n-1)) - (meanInfluence * n);
        if(OMX_DEBUG) { 
            Rprintf("Calculation for Gradient value %d: Cov: %3.9f, Mean: %3.9f, total: %3.9f\n",
            currentLoc, covInfluence, meanInfluence, gradient[currentLoc]); 
        }
    }
}