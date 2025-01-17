#pragma once

#include "mathUtils.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <cfloat>

// Forward declaration of ODE class
class QssOde;

//! BDF (Backward Differentiation Formula) Integrator
class BDFIntegrator
{
public:
    BDFIntegrator();
    virtual ~BDFIntegrator() {}

    //! Set the ODE object to be integrated
    void setOde(QssOde* ode);

    //! Initialize integrator arrays (problem size of N)
    virtual void initialize(size_t N);

    //! Set state at the start of a global timestep
    void setState(const dvec& yIn, double tStart);

    //! Take as many steps as needed to reach tf
    int integrateToTime(double tf);

    //! Take one step toward tf without stepping past it
    int integrateOneStep(double tf);

    //! State variables
    double tn;    //!< Current time relative to tstart
    double tstart;//!< Start time
    dvec y;       //!< Current state vector
    size_t N;     //!< Number of state variables

    //! Error tolerances
    double abstol; //!< Absolute error tolerance
    double reltol; //!< Relative error tolerance

    //! Step size controls
    double dtmin; //!< Minimum timestep allowed
    double dtmax; //!< Maximum timestep allowed
    double dt;    //!< Current timestep

    //! BDF method order (1 to 5)
    int order;

    //! Maximum number of Newton iterations
    int maxNewtonIter;

    //! Statistics
    int nSteps;     //!< Number of steps taken
    int nAccept;    //!< Number of accepted steps
    int nReject;    //!< Number of rejected steps
    int nEval;      //!< Number of function evaluations
    int nJac;       //!< Number of Jacobian evaluations

protected:
    QssOde* ode_;

    //! History of solution values needed for multistep method
    std::vector<dvec> yHistory;
    std::vector<double> tHistory;

    //! Compute error norm for step size control
    double errorNorm(const dvec& err, const dvec& y) const;

    //! Adjust step size based on error estimate
    double adjustStepSize(double dt, double error, double order);

    //! Solve nonlinear system using Newton iteration
    int solveNewton(double t, const dvec& ypred, dvec& y);

    //! Compute numerical Jacobian
    void computeJacobian(double t, const dvec& y, Eigen::MatrixXd& J);

    //! Predict solution using extrapolation from history
    void predict(dvec& ypred);

    //! Update solution history
    void updateHistory(double t, const dvec& y);

    //! Get coefficients for current BDF order
    void getBDFCoefficients(std::vector<double>& alpha) const;

private:
    Eigen::MatrixXd J;  //!< Jacobian matrix
    Eigen::MatrixXd I;  //!< Identity matrix
    Eigen::VectorXd f;  //!< Function evaluation
    Eigen::VectorXd d;  //!< Dummy array for QssOde interface
    Eigen::VectorXd dx; //!< Newton update
    Eigen::VectorXd residual; //!< Residual for Newton solver
    dvec scratch;       //!< Temporary work array
};