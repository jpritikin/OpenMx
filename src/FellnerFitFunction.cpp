/*
  Copyright 2015 Joshua Nathaniel Pritikin and contributors

  This is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Named in honor of Fellner (1987) "Sparse matrices, and the
// estimation of variance components by likelihood methods"

#include "glue.h"
#include <Eigen/Core>
#include "omxFitFunction.h"

#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <stan/math/prim/mat/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/mat/fun/quad_form.hpp>

namespace FellnerFitFunction {
	struct state {
		omxMatrix *smallRow;
		int totalNotMissing;
		std::vector<bool> notMissing;
		Eigen::VectorXd data;
		omxMatrix *cov;
		omxMatrix *means;
		omxMatrix *smallCov;
		omxMatrix *smallMeans;
		Eigen::MatrixXd fullCov; // sparse? TODO
		Eigen::VectorXd fullMeans;
	};
	
	static void compute(omxFitFunction *oo, int want, FitContext *fc)
	{
		using stan::math::check_ldlt_factor;
		static const char *function("FellnerFitFunction::compute");

		if (want & (FF_COMPUTE_PREOPTIMIZE)) return;

		state *st                   = (state *) oo->argStruct;
		omxExpectation *expectation = oo->expectation;
		omxData *data               = expectation->data;
		omxMatrix *cov              = st->cov;
		omxMatrix *means            = st->means;
		Eigen::MatrixXd &fullCov    = st->fullCov;
		Eigen::VectorXd &fullMeans  = st->fullMeans;

		Eigen::VectorXi contRemove(cov->cols);
		Eigen::VectorXd oldDefs;
		oldDefs.resize(data->defVars.size());
		oldDefs.setConstant(NA_REAL);

		fullMeans.resize(st->totalNotMissing);
		fullCov.resize(st->totalNotMissing, st->totalNotMissing);
		fullCov.setZero();

		int filteredPos = 0;
		for (int row=0; row < data->rows; ++row) {
			int numVarsFilled = data->handleDefinitionVarList(oo->matrix->currentState, row, oldDefs.data());
			if (row == 0 || numVarsFilled) {
				omxExpectationCompute(expectation, NULL);
			}

			int fullPos = row * cov->rows;
			for (int dx=0; dx < cov->rows; ++dx) contRemove[dx] = !st->notMissing[fullPos + dx];
			omxCopyMatrix(st->smallMeans, means);
			omxRemoveElements(st->smallMeans, contRemove.data());
			omxCopyMatrix(st->smallCov, cov);
			omxRemoveRowsAndColumns(st->smallCov, contRemove.data(), contRemove.data());

			EigenVectorAdaptor smallMeans(st->smallMeans);
			EigenMatrixAdaptor smallCov(st->smallCov);

			fullMeans.segment(filteredPos, st->smallCov->rows) = smallMeans;
			fullCov.block(filteredPos, filteredPos, st->smallCov->rows, st->smallCov->rows) = smallCov;
			filteredPos += st->smallCov->rows;
		}

		//mxPrintMat("mean", fullMeans);
		//mxPrintMat("cov", fullCov);

		double lp;
		try {
			stan::math::LDLT_factor<double,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(fullCov);
			check_ldlt_factor(function, "LDLT_Factor of covariance parameter", ldlt_Sigma);
			Eigen::MatrixXd isigma;
			ldlt_Sigma.inverse(isigma);
			
			lp = log_determinant_ldlt(ldlt_Sigma);
			Eigen::VectorXd resid = st->data - fullMeans;
			lp += stan::math::quad_form(isigma, resid);
			lp += log(2 * M_PI) * st->totalNotMissing;
		} catch (const std::exception& e) {
			lp = NA_REAL;
			if (fc) fc->recordIterationError("%s: %s", oo->name(), e.what());
		}
		oo->matrix->data[0] = lp;
	}

	static void destroy(omxFitFunction *oo)
	{
		state *st = (state*) oo->argStruct;
		omxFreeMatrix(st->smallMeans);
		omxFreeMatrix(st->smallCov);
		omxFreeMatrix(st->smallRow);
		delete st;
	}
};

void InitFellnerFitFunction(omxFitFunction *oo)
{
	omxExpectation* expectation = oo->expectation;
	if(expectation == NULL) {
		omxRaiseErrorf("%s cannot fit without a model expectation", oo->fitType);
		return;
	}
	omxMatrix *cov = omxGetExpectationComponent(expectation, oo, "cov");
	if(cov == NULL) { 
		omxRaiseError("No covariance expectation in FIML evaluation.");
		return;
	}

	omxMatrix *means = omxGetExpectationComponent(expectation, oo, "means");
	if(means == NULL) { 
		omxRaiseError("No means model in FIML evaluation.");
		return;
	}

	// prohibit ordinal for now TODO
	if (expectation->numOrdinal != 0) {
		Rf_error("%s cannot handle ordinal data yet", oo->fitType);
	}

	oo->computeFun = FellnerFitFunction::compute;
	oo->destructFun = FellnerFitFunction::destroy;
	FellnerFitFunction::state *st = new FellnerFitFunction::state;
	oo->argStruct = st;

	st->cov = cov;
	st->means = means;
	st->smallCov   = omxInitMatrix(1, 1, TRUE, oo->matrix->currentState);
	st->smallMeans = omxInitMatrix(1, 1, TRUE, oo->matrix->currentState);
	st->smallRow = omxInitMatrix(1, cov->cols, TRUE, oo->matrix->currentState);
	omxData *data               = expectation->data;
	omxMatrix *dataColumns	    = expectation->dataColumns;

	st->totalNotMissing = 0;
	st->notMissing.reserve(data->rows * cov->cols);
	for (int row=0; row < data->rows; ++row) {
		omxDataRow(data, row, dataColumns, st->smallRow);
		for (int col=0; col < cov->cols; ++col) {
			double val = omxMatrixElement(st->smallRow, 0, col);
			bool yes = std::isfinite(val);
			st->notMissing.push_back(yes);
			if (yes) ++st->totalNotMissing;
		}
	}

	st->data.resize(st->totalNotMissing);
	for (int row=0, dx=0; row < data->rows; ++row) {
		omxDataRow(data, row, dataColumns, st->smallRow);
		for (int col=0; col < cov->cols; ++col) {
			double val = omxMatrixElement(st->smallRow, 0, col);
			if (!std::isfinite(val)) continue;
			st->data[ dx++ ] = val;
		}
	}
}
