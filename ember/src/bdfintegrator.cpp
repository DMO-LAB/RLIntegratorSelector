#include "bdfintegrator.h"
#include "qssintegrator.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <cfloat>

BDFIntegrator::BDFIntegrator()
    : abstol(1e-8)
    , reltol(1e-6)
    , dtmin(1e-15)
    , tn(0)
    , tstart(0)
    , dtmax(1e-6)
    , N(0)
    , dt(0)
    , order(1)  // Start with BDF1 (implicit Euler)
    , maxNewtonIter(5)
    , nSteps(0)
    , nAccept(0)
    , nReject(0)
    , nEval(0)
    , nJac(0)
    , ode_(nullptr)
{
}

void BDFIntegrator::setOde(QssOde* ode)
{
    ode_ = ode;
}

void BDFIntegrator::initialize(size_t N_)
{
    N = N_;
    y.resize(N);
    yHistory.resize(6); // BDF up to order 5 needs 6 history points
    for (auto& yh : yHistory) {
        yh.resize(N);
    }
    tHistory.resize(6);
    
    J.resize(N, N);
    I.setIdentity(N, N);
    f.resize(N);
    d.resize(N);
    dx.resize(N);
    residual.resize(N);
    scratch.resize(N);
}

void BDFIntegrator::setState(const dvec& yIn, double tstart_)
{
    assert(yIn.size() == (int)N);
    y = yIn;
    tstart = tstart_;
    tn = 0.0;
    dt = dtmax;  // Start with maximum step size
    
    // Initialize history with initial condition
    for (auto& yh : yHistory) {
        yh = yIn;
    }
    for (size_t i = 0; i < tHistory.size(); i++) {
        tHistory[i] = tstart;
    }
    
    nSteps = 0;
    nAccept = 0;
    nReject = 0;
    nEval = 0;
    nJac = 0;
    order = 1;  // Start with first order
}

void BDFIntegrator::getBDFCoefficients(std::vector<double>& alpha) const
{
    // BDF coefficients for orders 1-5
    static const std::vector<std::vector<double>> bdfCoeffs = {
        {1.0, -1.0},  // BDF1 (implicit Euler)
        {3.0/2.0, -2.0, 1.0/2.0},  // BDF2 
        {11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0},  // BDF3
        {25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0},  // BDF4
        {137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0}  // BDF5
    };
    
    alpha = bdfCoeffs[order-1];
}

void BDFIntegrator::predict(dvec& ypred)
{
    // Polynomial extrapolation from history points
    ypred = yHistory[0];
    double dt = tn + this->dt - tHistory[0];
    
    for (int i = 1; i < order; i++) {
        double prod = 1.0;
        for (int j = 0; j < i; j++) {
            prod *= dt / (tHistory[0] - tHistory[j+1]);
        }
        ypred += prod * (yHistory[i] - yHistory[i-1]);
    }
}

void BDFIntegrator::computeJacobian(double t, const dvec& y, Eigen::MatrixXd& J)
{
    const double eps = std::sqrt(DBL_EPSILON);
    dvec ytemp = y;
    dvec f1(N), f2(N);
    dvec temp_d(N);
    
    // Compute Jacobian by finite differences
    ode_->odefun(t, y, f1, temp_d);
    
    for (size_t i = 0; i < N; i++) {
        double dy = std::max(1e-5 * std::abs(y[i]), eps);
        ytemp[i] = y[i] + dy;
        ode_->odefun(t, ytemp, f2, temp_d);
        // Convert Array to Matrix for column assignment
        J.col(i) = (f2.matrix() - f1.matrix()) / dy;
        ytemp[i] = y[i];
    }
    
    nJac++;
    nEval += N + 1;
}

int BDFIntegrator::solveNewton(double t, const dvec& ypred, dvec& y)
{
    std::vector<double> alpha;
    getBDFCoefficients(alpha);
    
    // Set up Newton iteration
    y = ypred;
    double gamma = alpha[0] / dt;
    
    // Get initial residual
    dvec f_temp(N);
    dvec d_temp(N);
    ode_->odefun(t, y, f_temp, d_temp);
    nEval++;
    
    // Convert to matrix form for linear algebra operations
    residual = (gamma * y).matrix();
    for (size_t i = 0; i < order; i++) {
        residual += (alpha[i] * yHistory[i] / dt).matrix();
    }
    residual -= f_temp.matrix();
    
    double initialNorm = residual.norm();
    double bestNorm = initialNorm;
    dvec bestY = y;
    
    // Newton iteration
    const double tol = 0.1 * (abstol + reltol * y.matrix().norm());
    
    for (int iter = 0; iter < maxNewtonIter; iter++) {
        if (residual.norm() < tol) {
            return 0;  // Converged
        }
        
        // Update Jacobian if needed
        if (iter == 0 || residual.norm() > 0.1 * bestNorm) {
            computeJacobian(t, y, J);
            J = I * gamma - J;  // Add mass matrix term
        }
        
        // Solve linear system
        dx = J.lu().solve(residual);
        
        // Update solution (converting back to Array) with temperature constraint
        y = (y.matrix() - dx).array();
        
        // Enforce temperature positivity constraint (assuming T is at index 1)
        if (y[1] <= 0) {  // Temperature index (assuming same as in SourceSystem)
            double scale = 0.1;  // Reduce step if temperature goes negative
            y[1] = scale * std::abs(y[1]) + (1 - scale) * bestY[1];
            dx = (bestY.matrix() - y.matrix()).array();  // Adjust dx for the modification
        }
        
        // Limit the maximum change in temperature
        double maxTempChange = 0.2 * std::abs(bestY[1]);  // 20% maximum change per iteration
        if (std::abs(y[1] - bestY[1]) > maxTempChange) {
            double scale = maxTempChange / std::abs(y[1] - bestY[1]);
            y = bestY + scale * (y - bestY);
        }
        
        // Update residual
        ode_->odefun(t, y, f_temp, d_temp);
        nEval++;
        
        residual = (gamma * y).matrix();
        for (size_t i = 0; i < order; i++) {
            residual += (alpha[i] * yHistory[i] / dt).matrix();
        }
        residual -= f_temp.matrix();
        
        // Keep track of best solution
        if (residual.norm() < bestNorm) {
            bestNorm = residual.norm();
            bestY = y;
        }
        
        // Check for convergence
        if (dx.norm() < tol) {
            return 0;
        }
    }
    
    // Failed to converge - use best solution found
    y = bestY;
    return 1;
}

void BDFIntegrator::updateHistory(double t, const dvec& y)
{
    // Shift history
    for (int i = order-1; i > 0; i--) {
        yHistory[i] = yHistory[i-1];
        tHistory[i] = tHistory[i-1];
    }
    yHistory[0] = y;
    tHistory[0] = t;
}

int BDFIntegrator::integrateOneStep(double tf)
{
    // Don't step past tf
    double dtTemp = std::min(dt, tf - tn);
    
    while (true) {
        dvec ypred;
        predict(ypred);
        
        // Verify prediction is physically valid
        if (ypred[1] <= 0) {  // Check temperature
            dtTemp *= 0.5;
            if (dtTemp < dtmin) {
                return -1;
            }
            continue;
        }
        
        // Try to take step
        int ret = solveNewton(tn + dtTemp, ypred, scratch);
        
        // Verify solution is physically valid
        bool physicallyValid = true;
        if (scratch[1] <= 0) {  // Check temperature
            physicallyValid = false;
        }
        
        if (ret == 0 && physicallyValid) {
            // Step accepted
            // Verify the change in temperature is not too large
            double tempChange = std::abs(scratch[1] - y[1]) / y[1];
            if (tempChange > 0.25) {  // More than 25% change
                dtTemp *= 0.5;
                continue;
            }
            
            updateHistory(tn + dtTemp, scratch);
            y = scratch;
            tn += dtTemp;
            
            // More conservative order increase
            if (nSteps > 2*order && order < 5 && nReject == 0) {
                order++;  // Increase order after more steps and no rejections
            }
            
            // More conservative step size increase
            dt = std::min(dtTemp * 1.2, dtmax);  // Limit step size increase
            nSteps++;
            nAccept++;
            return 0;
        } else {
            // Step rejected
            dtTemp *= 0.25;  // More aggressive step size reduction
            if (dtTemp < dtmin) {
                return -1;  // Step size too small
            }
            order = 1;  // Reset to first order on rejection
            nReject++;
        }
    }
}

int BDFIntegrator::integrateToTime(double tf)
{
    while (tn < tf) {
        int ret = integrateOneStep(tf);
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
}