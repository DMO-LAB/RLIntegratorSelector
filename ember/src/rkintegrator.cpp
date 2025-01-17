#include "rkintegrator.h"
#include "qssintegrator.h" // For QssOde definition

// RK23 Bogacki-Shampine method coefficients
const double RK23Integrator::a21 = 1.0/2.0;
const double RK23Integrator::a31 = 0.0;
const double RK23Integrator::a32 = 3.0/4.0;
const double RK23Integrator::a41 = 2.0/9.0;
const double RK23Integrator::a42 = 1.0/3.0;
const double RK23Integrator::a43 = 4.0/9.0;

const double RK23Integrator::b1 = 2.0/9.0;
const double RK23Integrator::b2 = 1.0/3.0;
const double RK23Integrator::b3 = 4.0/9.0;
const double RK23Integrator::b4 = 0.0;

const double RK23Integrator::e1 = -5.0/72.0;
const double RK23Integrator::e2 = 1.0/12.0;
const double RK23Integrator::e3 = -1.0/9.0;
const double RK23Integrator::e4 = 1.0/8.0;

RkIntegratorBase::RkIntegratorBase()
    : abstol(1e-8)
    , reltol(1e-6)
    , dtmin(1e-15)
    , dtmax(1e-6)
    , dt(0)
    , tn(0)
    , tstart(0)
    , N(0)
    , nSteps(0)
    , nAccept(0)
    , nReject(0)
    , nEval(0)
    , ode_(nullptr)
{
}

void RkIntegratorBase::setOde(QssOde* ode)
{
    ode_ = ode;
}

void RkIntegratorBase::initialize(size_t N_)
{
    N = N_;
    y.resize(N);
    scratch.resize(N);
    k1.resize(N);
    k2.resize(N);
    k3.resize(N);
    k4.resize(N);
    yerr.resize(N);
    ytmp.resize(N);
}

void RkIntegratorBase::setState(const dvec& yIn, double tstart_)
{
    assert(yIn.size() == (int)N);
    y = yIn;
    tstart = tstart_;
    tn = 0.0;
    dt = dtmax;  // Start with maximum step size
    nSteps = 0;
    nAccept = 0;
    nReject = 0;
    nEval = 0;
}

double RkIntegratorBase::errorNorm(const dvec& err, const dvec& y,
                                 double abstol, double reltol)
{
    double norm = 0.0;
    for (size_t i = 0; i < N; i++) {
        double sc = abstol + reltol * std::abs(y[i]);
        norm = std::max(norm, std::abs(err[i]) / sc);
    }
    return norm;
}

double RkIntegratorBase::adjustStepSize(double dt, double error, double order)
{
    const double safety = 0.9;
    const double maxFactor = 5.0;
    const double minFactor = 0.2;

    double factor = safety * std::pow(error, -1.0 / order);
    factor = std::min(maxFactor, std::max(minFactor, factor));
    
    return std::min(dtmax, std::max(dtmin, dt * factor));
}

int RkIntegratorBase::integrateToTime(double tf)
{
    //std::cout << "RK integrateToTime" << std::endl;
    while (tn < tf) {
        int ret = integrateOneStep(tf);
        if (ret != 0) {
            //std::cout << "RK integrateToTime failed" << std::endl;
            return ret;
        }
    }
    return 0;
}

int RK23Integrator::integrateOneStep(double tf)
{
    dvec d(N); // Dummy vector for QssOde interface

    // Don't step past tf
    double dtTemp = std::min(dt, tf - tn);
    
    while (true) {
        // First stage
        ode_->odefun(tn + tstart, y, k1, d);
        nEval++;

        // Second stage
        ytmp = y + dtTemp * (a21 * k1);
        ode_->odefun(tn + tstart + a21*dtTemp, ytmp, k2, d);
        nEval++;

        // Third stage
        ytmp = y + dtTemp * (a31*k1 + a32*k2);
        ode_->odefun(tn + tstart + 0.75*dtTemp, ytmp, k3, d);
        nEval++;

        // Fourth stage
        ytmp = y + dtTemp * (a41*k1 + a42*k2 + a43*k3);
        ode_->odefun(tn + tstart + dtTemp, ytmp, k4, d);
        nEval++;

        // Error estimate
        yerr = dtTemp * (e1*k1 + e2*k2 + e3*k3 + e4*k4);
        
        // Error control
        double error = errorNorm(yerr, y, abstol, reltol);
        
        if (error <= 1.0) {
            // Step accepted
            y += dtTemp * (b1*k1 + b2*k2 + b3*k3 + b4*k4);
            tn += dtTemp;
            dt = adjustStepSize(dtTemp, error, 2.0);
            nSteps++;
            nAccept++;
            return 0;
        } else {
            // Step rejected
            dt = adjustStepSize(dtTemp, error, 2.0);
            if (dt < dtmin) {
                return -1; // Step size too small
            }
            dtTemp = dt;
            nReject++;
        }
    }
}