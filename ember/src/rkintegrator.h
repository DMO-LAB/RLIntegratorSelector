#pragma once

#include "mathUtils.h"

// Forward declaration of ODE class (same as used by QssIntegrator)
class QssOde;

//! Base class for Runge-Kutta integrators
class RkIntegratorBase
{
public:
    RkIntegratorBase();
    virtual ~RkIntegratorBase() {}

    //! Set the ODE object to be integrated
    void setOde(QssOde* ode);

    //! Initialize integrator arrays (problem size of N)
    virtual void initialize(size_t N);

    //! Set state at the start of a global timestep
    void setState(const dvec& yIn, double tStart);

    //! Take as many steps as needed to reach `tf`
    //! Note that `tf` is relative to tstart, not absolute
    int integrateToTime(double tf);

    //! Take one step toward `tf` without stepping past it
    virtual int integrateOneStep(double tf) = 0;

    //! Error tolerances
    double abstol; //!< Absolute error tolerance
    double reltol; //!< Relative error tolerance

    //! Step size controls
    double dtmin; //!< Minimum timestep allowed
    double dtmax; //!< Maximum timestep allowed
    double dt;    //!< Current timestep
    
    //! State variables
    double tn;    //!< Internal integrator time (relative to tstart)
    double tstart;//!< Start time for the current integration
    dvec y;       //!< current state vector
    size_t N;     //!< Number of state variables

    //! Statistics
    int nSteps;     //!< Number of steps taken
    int nAccept;    //!< Number of accepted steps
    int nReject;    //!< Number of rejected steps
    int nEval;      //!< Number of function evaluations

protected:
    QssOde* ode_;

    //! Compute error norm for step size control
    double errorNorm(const dvec& err, const dvec& y, double abstol, double reltol);

    //! Adjust step size based on error estimate
    double adjustStepSize(double dt, double error, double order);

    dvec scratch;   //!< Temporary work array
    dvec k1, k2, k3, k4; //!< Stage vectors for RK23
    dvec yerr;      //!< Error estimate
    dvec ytmp;      //!< Temporary state vector
};

//! Runge-Kutta-Fehlberg 2(3) method (Bogacki-Shampine)
class RK23Integrator : public RkIntegratorBase
{
public:
    RK23Integrator() {}
    virtual ~RK23Integrator() {}

    //! Take one step toward tf without stepping past it
    virtual int integrateOneStep(double tf);

private:
    static const double a21, a31, a32;
    static const double a41, a42, a43;
    static const double b1, b2, b3, b4;
    static const double e1, e2, e3, e4;
};