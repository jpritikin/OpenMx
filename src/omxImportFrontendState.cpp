/*
 *  Copyright 2007-2014 The OpenMx Project
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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <sys/stat.h>
#include <errno.h>

#include "omxDefines.h"
#include "glue.h"
#include "omxState.h"
#include "omxNPSOLSpecific.h"
#include "Compute.h"

/* Outside R Functions */
static int isDir(const char *path);

int matchCaseInsensitive(const char *source, const char *target) {
	return strcasecmp(source, target) == 0;
}

void omxProcessMxDataEntities(SEXP data) {
	SEXP nextLoc;
	if(OMX_DEBUG) { mxLog("Processing %d data source(s).", Rf_length(data));}

	for(int index = 0; index < Rf_length(data); index++) {
		Rf_protect(nextLoc = VECTOR_ELT(data, index));			// Retrieve the data object
		omxNewDataFromMxData(nextLoc);
	}
}

void omxProcessMxMatrixEntities(SEXP matList) {
	if(OMX_DEBUG) { mxLog("Processing %d matrix(ces).", Rf_length(matList));}
	SEXP nextLoc, nextMat;
	globalState->matrixList.clear();
	SEXP matListNames = Rf_getAttrib(matList, R_NamesSymbol);

	for(int index = 0; index < Rf_length(matList); index++) {
		omxManageProtectInsanity protectManager;
		Rf_protect(nextLoc = VECTOR_ELT(matList, index));		// This is the matrix + populations
		Rf_protect(nextMat = VECTOR_ELT(nextLoc, 0));		// The first element of the list is the matrix of values
		omxMatrix *mat = omxNewMatrixFromRPrimitive(nextMat, globalState, 1, -index - 1);
		mat->name = CHAR(STRING_ELT(matListNames, index));
		globalState->matrixList.push_back(mat);

		if(OMX_DEBUG) { omxPrintMatrix(mat, "Imported"); }

		if (isErrorRaised()) return;
	}
}

void omxProcessMxAlgebraEntities(SEXP algList) {
	SEXP nextAlgTuple;
	SEXP algListNames = Rf_getAttrib(algList, R_NamesSymbol);

	if(OMX_DEBUG) { mxLog("Processing %d algebras.", Rf_length(algList)); }

	for(int index = 0; index < Rf_length(algList); index++) {
		globalState->algebraList.push_back(omxInitMatrix(0, 0, TRUE, globalState));
	}

	for(int index = 0; index < Rf_length(algList); index++) {
		omxManageProtectInsanity protectManager;
		Rf_protect(nextAlgTuple = VECTOR_ELT(algList, index));
		if(IS_S4_OBJECT(nextAlgTuple)) {
			// fitfunction, delay until expectations are ready
		} else {								// This is an algebra spec.
			SEXP dimnames, formula;
			omxMatrix *amat = globalState->algebraList[index];
			Rf_protect(dimnames = VECTOR_ELT(nextAlgTuple, 0));
			omxFillMatrixFromRPrimitive(amat, NULL, globalState, 1, index);
			Rf_protect(formula = VECTOR_ELT(nextAlgTuple, 1));
			omxFillMatrixFromMxAlgebra(amat, formula, CHAR(STRING_ELT(algListNames, index)), dimnames);
			omxMarkDirty(amat);
		}
		if (isErrorRaised()) return;
	}
}

void omxProcessMxFitFunction(SEXP algList)
{
	SEXP nextAlgTuple;
	SEXP algListNames = Rf_getAttrib(algList, R_NamesSymbol);

	for(int index = 0; index < Rf_length(algList); index++) {
		Rf_protect(nextAlgTuple = VECTOR_ELT(algList, index));		// The next algebra or fit function to process
		if(IS_S4_OBJECT(nextAlgTuple)) {
			SEXP fitFunctionClass;
			Rf_protect(fitFunctionClass = STRING_ELT(Rf_getAttrib(nextAlgTuple, Rf_install("class")), 0));
			const char *fitType = CHAR(fitFunctionClass);
			omxMatrix *fm = globalState->algebraList[index];
			omxFillMatrixFromMxFitFunction(fm, fitType, index);
			fm->fitFunction->rObj = nextAlgTuple;
			fm->name = CHAR(STRING_ELT(algListNames, index));
			Rf_unprotect(1);	// fitFunctionClass
		}
		if (isErrorRaised()) return;
		Rf_unprotect(1); //nextAlgTuple
	}
}

void omxCompleteMxFitFunction(SEXP algList)
{
	SEXP nextAlgTuple;

	for(int index = 0; index < Rf_length(algList); index++) {
		Rf_protect(nextAlgTuple = VECTOR_ELT(algList, index));             // The next algebra or fit function to process
		if(IS_S4_OBJECT(nextAlgTuple)) {
			omxMatrix *fm = globalState->algebraList[index];
			if (!fm->fitFunction->freeVarGroup) {
				setFreeVarGroup(fm->fitFunction, Global->freeGroup[0]);
			}
			omxCompleteFitFunction(fm);
		}
		Rf_unprotect(1);
	}
}

void omxProcessMxExpectationEntities(SEXP expList) {
	if(OMX_DEBUG) { mxLog("Initializing %d Model Expectation(s).", Rf_length(expList));}
	SEXP nextExp;
	SEXP eNames = Rf_getAttrib(expList, R_NamesSymbol);

	for(int index = 0; index < Rf_length(expList); index++) {
		Rf_protect(nextExp = VECTOR_ELT(expList, index));
		omxExpectation *ex = omxNewIncompleteExpectation(nextExp, index, globalState);
		ex->name = CHAR(STRING_ELT(eNames, index));
		globalState->expectationList.push_back(ex);
		if (isErrorRaised()) return;
	}
}


void omxCompleteMxExpectationEntities() {
	if(OMX_DEBUG) { mxLog("Completing %d Model Expectation(s).", (int) globalState->expectationList.size());}
	
	for(size_t index = 0; index < globalState->expectationList.size(); index++) {
		omxCompleteExpectation(globalState->expectationList[index]);
		if (isErrorRaised()) return;
	}
}

void omxProcessMxComputeEntities(SEXP rObj)
{
	if (Rf_isNull(rObj)) return;

	SEXP s4class;
	Rf_protect(s4class = STRING_ELT(Rf_getAttrib(rObj, Rf_install("class")), 0));
	omxCompute *compute = omxNewCompute(globalState, CHAR(s4class));
	compute->initFromFrontend(rObj);
	Global->computeList.push_back(compute);
}

void omxInitialMatrixAlgebraCompute() {
	size_t numMats = globalState->matrixList.size();
	int numAlgs = globalState->algebraList.size();

	if(OMX_DEBUG) {mxLog("Completed Algebras and Matrices.  Beginning Initial Compute.");}

	for(size_t index = 0; index < numMats; index++) {
		omxRecompute(globalState->matrixList[index]);
	}

	for(int index = 0; index < numAlgs; index++) {
		omxMatrix *matrix = globalState->algebraList[index];
		omxInitialCompute(matrix);
	}
}

/*
checkpointList is a list().  Each element refers to one checkpointing request.
Each interval request is a list of Rf_length 5.
The first element is an integer that specifies type: 0 = file, 1 = socket, 2=R_connection
For a file, the next two are the directory(string)  and file name (string).
For a socket, they are server (string) and port number (int).
For a connection, the next one is the R_connection SEXP object.
After that is an integer <type> specifier.  0 means minutes, 1 means iterations.
The last element is an integer count, indicating the number of <type>s per checkpoint.
*/
void omxProcessCheckpointOptions(SEXP checkpointList)
{
	if(OMX_DEBUG) {mxLog("Found %d checkpoints", Rf_length(checkpointList)); }

	SEXP nextLoc;

	for(int index = 0; index < Rf_length(checkpointList); ++index) {
		omxCheckpoint *oC = new omxCheckpoint;

		const char *pathName, *fileName;

		Rf_protect(nextLoc = VECTOR_ELT(checkpointList, index));
		int next = 0;
		oC->type = (omxCheckpointType) INTEGER(VECTOR_ELT(nextLoc, next++))[0];
		switch(oC->type) {
		case OMX_FILE_CHECKPOINT:{
			pathName = CHAR(STRING_ELT(VECTOR_ELT(nextLoc, next++), 0));
			fileName = CHAR(STRING_ELT(VECTOR_ELT(nextLoc, next++), 0));

			if(!isDir(pathName)) {
				Rf_error("Unable to open directory %s for checkpoint storage.\n", pathName);
			}

			std::string fullname = string_snprintf("%s/%s", pathName, fileName);
			if(OMX_VERBOSE) { mxLog("Opening File: %s", fullname.c_str()); }
			oC->file = fopen(fullname.c_str(), "w");
			if(!oC->file) {
				Rf_error("Unable to open file %s for checkpoint storage: %s.\n",
					 fullname.c_str(), strerror(errno));
			}
			break;}

		case OMX_CONNECTION_CHECKPOINT:{
			Rf_error("Warning NYI: Socket checkpoints Not Yet Implemented.\n");
			break;}
		}

		const char *units = CHAR(STRING_ELT(VECTOR_ELT(nextLoc, next++), 0));
		if (strEQ(units, "iterations")) {
			oC->iterPerCheckpoint = Rf_asInteger(VECTOR_ELT(nextLoc, next++));
		} else if (strEQ(units, "minutes")) {
			oC->timePerCheckpoint = Rf_asReal(VECTOR_ELT(nextLoc, next++)) * 60.0; // Constrained to seconds.
			if(oC->timePerCheckpoint < 1) oC->timePerCheckpoint = 1;
		} else if (strEQ(units, "evaluations")) {
			oC->evalsPerCheckpoint = Rf_asInteger(VECTOR_ELT(nextLoc, next++));
		} else {
			Rf_error("In 'Checkpoint Units' model option, '%s' not recognized", units);
		}
		Global->checkpointList.push_back(oC);
	}
}

void omxProcessFreeVarList(SEXP varList, std::vector<double> *startingValues)
{
	if(OMX_VERBOSE) { mxLog("Processing Free Parameters."); }

	{
		FreeVarGroup *fvg = new FreeVarGroup;
		fvg->id.push_back(FREEVARGROUP_ALL);   // all variables
		Global->freeGroup.push_back(fvg);

		fvg = new FreeVarGroup;
		fvg->id.push_back(FREEVARGROUP_NONE);  // no variables
		Global->freeGroup.push_back(fvg);
	}

	SEXP nextVar, nextLoc;
	int numVars = Rf_length(varList);
	startingValues->resize(numVars);
	for (int fx = 0; fx < numVars; fx++) {
		omxManageProtectInsanity mpi;

		omxFreeVar *fv = new omxFreeVar;
		// default group has free all variables
		Global->freeGroup[0]->vars.push_back(fv);

		fv->id = fx;
		fv->name = CHAR(Rf_asChar(STRING_ELT(Rf_getAttrib(varList, R_NamesSymbol), fx)));
		Rf_protect(nextVar = VECTOR_ELT(varList, fx));

		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 0));
		fv->lbound = REAL(nextLoc)[0];
		if (ISNA(fv->lbound)) fv->lbound = NEG_INF;
		if (fv->lbound == 0.0) fv->lbound = 0.0;

		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 1));
		fv->ubound = REAL(nextLoc)[0];
		if (ISNA(fv->ubound)) fv->ubound = INF;
		if (fv->ubound == 0.0) fv->ubound = -0.0;

		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 2));
		int groupCount = Rf_length(nextLoc);
		for (int gx=0; gx < groupCount; ++gx) {
			int group = INTEGER(nextLoc)[gx];
			if (group == 0) continue;
			Global->findOrCreateVarGroup(group)->vars.push_back(fv);
		}

		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 3));
		int numDeps = LENGTH(nextLoc);
		fv->numDeps = numDeps;
		fv->deps = (int*) R_alloc(numDeps, sizeof(int));
		for (int i = 0; i < numDeps; i++) {
			fv->deps[i] = INTEGER(nextLoc)[i];
		}

		int numLocs = Rf_length(nextVar) - 5;
		if(OMX_DEBUG) { 
			mxLog("Free parameter %d bounded (%f, %f): %d locations", fx, 
			      fv->lbound, fv->ubound, numLocs);
		}
		for(int locIndex = 0; locIndex < numLocs; locIndex++) {
			Rf_protect(nextLoc = VECTOR_ELT(nextVar, locIndex+4));
			int* theVarList = INTEGER(nextLoc);

			omxFreeVarLocation loc;
			loc.matrix = theVarList[0];
			loc.row = theVarList[1];
			loc.col = theVarList[2];

			fv->locations.push_back(loc);
		}
		Rf_protect(nextLoc = VECTOR_ELT(nextVar, Rf_length(nextVar)-1));
		double sv = REAL(nextLoc)[0];
		/*if (sv < fv->lbound) {
			Rf_warning("Moving starting value of parameter '%s' within bounds %f -> %f",
				fv->name, sv, fv->lbound);
			sv = fv->lbound;
		} else if (sv > fv->ubound) {
			Rf_warning("Moving starting value of parameter '%s' within bounds %f -> %f",
				fv->name, sv, fv->ubound);
			sv = fv->ubound;
		}*/
		(*startingValues)[fx] = sv;
	}

	Global->deduplicateVarGroups();
}

/*
	intervalList is a list().  Each element refers to one confidence interval request.
	Each interval request is a Rf_length 5 vector of REAL.
	The first three elements are the matrixPointer, Row, and Column of the element
	for which bounds are to be calculated, and are cast to ints here for speed.
	The last two are the upper and lower boundaries for the confidence space (respectively).
*/
void omxProcessConfidenceIntervals(SEXP intervalList)  {
	SEXP nextVar;
	if(OMX_VERBOSE) { mxLog("Processing Confidence Interval Requests.");}
	Global->numIntervals = Rf_length(intervalList);
	if(OMX_DEBUG) {mxLog("Found %d requests.", Global->numIntervals); }
	Global->intervalList = (omxConfidenceInterval*) R_alloc(Global->numIntervals, sizeof(omxConfidenceInterval));
	for(int index = 0; index < Global->numIntervals; index++) {
		omxConfidenceInterval *oCI = &(Global->intervalList[index]);
		Rf_protect(nextVar = VECTOR_ELT(intervalList, index));
		double* intervalInfo = REAL(nextVar);
		oCI->matrix = omxMatrixLookupFromState1( nextVar, globalState);	// Expects an R object
		oCI->row = (int) intervalInfo[1];		// Cast to int in C to save memory/Protection ops
		oCI->col = (int) intervalInfo[2];		// Cast to int in C to save memory/Protection ops
		oCI->lbound = intervalInfo[3];
		oCI->ubound = intervalInfo[4];
		oCI->max = R_NaReal;					// NAs, in case something goes wrong
		oCI->min = R_NaReal;
	}
	if(OMX_VERBOSE) { mxLog("Processed."); }
	if(OMX_DEBUG) { mxLog("%d intervals requested.", Global->numIntervals); }
}

void omxProcessConstraints(SEXP constraints)  {
	int ncnln = 0; 
	if(OMX_VERBOSE) { mxLog("Processing Constraints.");}
	omxMatrix *arg1, *arg2;
	SEXP nextVar, nextLoc;
	globalState->numConstraints = Rf_length(constraints);
	if(OMX_DEBUG) {mxLog("Found %d constraints.", globalState->numConstraints); }
	globalState->conList = (omxConstraint*) R_alloc(globalState->numConstraints, sizeof(omxConstraint));
	ncnln = 0;
	for(int constraintIndex = 0; constraintIndex < globalState->numConstraints; constraintIndex++) {
		Rf_protect(nextVar = VECTOR_ELT(constraints, constraintIndex));
		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 0));
		arg1 = omxMatrixLookupFromState1(nextLoc, globalState);
		Rf_protect(nextLoc = VECTOR_ELT(nextVar, 1));
		arg2 = omxMatrixLookupFromState1(nextLoc, globalState);
		globalState->conList[constraintIndex].opCode = Rf_asInteger(VECTOR_ELT(nextVar, 2));
		omxMatrix *args[2] = {arg1, arg2};
		globalState->conList[constraintIndex].result = omxNewAlgebraFromOperatorAndArgs(10, args, 2, globalState); // 10 = binary subtract
		omxRecompute(globalState->conList[constraintIndex].result);
		int nrows = globalState->conList[constraintIndex].result->rows;
		int ncols = globalState->conList[constraintIndex].result->cols;
		globalState->conList[constraintIndex].size = nrows * ncols;
		ncnln += globalState->conList[constraintIndex].size;
	}
	if(OMX_VERBOSE) { mxLog("Processed."); }
	if(OMX_DEBUG) { mxLog("%d effective constraints.", ncnln); }
	globalState->ncnln = ncnln;
}

/*
*  Acknowledgement:
*  This function is duplicated from the function of the same name in the R source code.
*  The function appears in src/main/sysutils.c
*  Thanks to Michael Spiegel for finding it.
*  This code is licensed under the terms of the GNU General Public License.
*/
static int isDir(const char *path)
{
    struct stat sb;
    int isdir = 0;
    if(!path) return 0;
    if(stat(path, &sb) == 0) {
        isdir = (sb.st_mode & S_IFDIR) > 0; /* is a directory */
        /* We want to know if the directory is writable by this user,
           which mode does not tell us */
        isdir &= (access(path, W_OK) == 0);
    }
    return isdir;
}

