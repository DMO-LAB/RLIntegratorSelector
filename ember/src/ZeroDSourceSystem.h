// ZeroDSourceSystem.h
#pragma once

#include "sourceSystem.h"
#include "mathUtils.h"
#include "sundialsUtils.h"
#include "qssintegrator.h"
#include "cantera/base/Solution.h"
#include "readConfig.h"
#include "chemistry0d.h"
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <memory>

// Base class interface for zero-D integrators
class ZeroDSourceSystem {
public:
    virtual ~ZeroDSourceSystem() = default;
    virtual void initialize(size_t nSpec) = 0;
    virtual void setGas(CanteraGas* gas_) = 0;
    virtual void setOptions(ConfigOptions& options_) = 0;
    virtual void setState(double tInitial, double tt, const dvec& yy) = 0;
    virtual int integrateToTime(double tf) = 0;
    virtual int integrateOneStep(double tf) = 0;
    virtual double time() const = 0;

    double T;  // Temperature
    dvec Y;    // Mass fractions
    size_t nSpec; // Number of species
};

// CVODE implementation for zero-D combustion
class ZeroDSourceCVODE : public ZeroDSourceSystem, public sdODE 
{
public:
    ZeroDSourceCVODE();
    
    // ZeroDSourceSystem interface implementation
    void initialize(size_t nSpec) override;
    void setGas(CanteraGas* gas_) override;
    void setOptions(ConfigOptions& options_) override;
    void setState(double tInitial, double tt, const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;

    // sdODE interface implementation
    int f(const realtype t, const sdVector& y, sdVector& ydot) override;
    int g(realtype t, sdVector& y, realtype *gout) override { return 0; }
    int denseJacobian(const realtype t, const sdVector& y, const sdVector& ydot, sdMatrix& J) override;

private:
    // Helper functions
    int fdJacobian(const realtype t, const sdVector& y, const sdVector& ydot, sdMatrix& J);
    void unroll_y(const sdVector& y);
    void roll_y(sdVector& y) const;
    void roll_ydot(sdVector& ydot) const;
    void updateThermo();

    // Internal state
    CanteraGas* gas;
    ConfigOptions* options;
    std::unique_ptr<SundialsCvode> integrator;
    std::shared_ptr<SundialsContext> sunContext;  // SUNDIALS context for vector creation
    
    // Work arrays
    dvec dYdt;
    dvec wDot;
    dvec hk;
    double rho;
    double cp;
    double dTdt;
    double qDot;
};

class ZeroDSourceQSS : public ZeroDSourceSystem, public QssOde
{
public:
    ZeroDSourceQSS();
    void initialize(size_t nSpec) override;
    void setGas(CanteraGas* gas_) override;
    void setOptions(ConfigOptions& options_) override;
    void setState(double tInitial, double tt, const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;

    // Implementation of QssOde interface
    void odefun(double t, const dvec& y, dvec& q, dvec& d, bool corrector=false) override;

private:
    void unroll_y(const dvec& y, bool corrector=false);
    void roll_y(dvec& y) const;
    void roll_ydot(dvec& q, dvec& d) const;
    void updateThermo();

    CanteraGas* gas;
    ConfigOptions* options;
    QssIntegrator integrator;

    dvec dYdtQ;
    dvec dYdtD;
    dvec wDotQ;
    dvec wDotD;
    dvec hk;
    double rho;
    double cp;
    double dTdtQ;
    double dTdtD;
    double qDot;
};

class ZeroDSourceBoostRK : public ZeroDSourceSystem {
public:
    ZeroDSourceBoostRK();
    void initialize(size_t nSpec) override;
    void setGas(CanteraGas* gas_) override;
    void setOptions(ConfigOptions& options_) override;
    void setState(double tInitial, double tt, const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;

public:
    using state_type = std::vector<double>;
    // Public operator() needed for boost::odeint
    void operator()(const state_type& y, state_type& dydt, double t);
    
private:
    void unroll_y(const state_type& y);
    void roll_y(state_type& y) const;
    void roll_ydot(state_type& dydt) const;
    void updateThermo();

    CanteraGas* gas;
    ConfigOptions* options;
    
    // Boost.Odeint stepper type
    using base_stepper_type = boost::numeric::odeint::runge_kutta_cash_karp54<state_type>;
    using controlled_stepper_type = boost::numeric::odeint::controlled_runge_kutta<base_stepper_type>;
    controlled_stepper_type stepper;
    
    state_type state;
    double tStart;
    double tNow;
    
    dvec dYdt;
    dvec wDot;
    dvec hk;
    double rho;
    double cp;
    double dTdt;
    double qDot;
    size_t nSteps;
};

class ZeroDSourceRK45 : public ZeroDSourceSystem {
public:
    ZeroDSourceRK45();
    
    // ZeroDSourceSystem interface implementation
    void initialize(size_t nSpec) override;
    void setGas(CanteraGas* gas_) override;
    void setOptions(ConfigOptions& options_) override;
    void setState(double tInitial, double tt, const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;

private:
    void computeDerivatives(const std::vector<double>& y, std::vector<double>& dy);
    void updateThermo();

    // Butcher tableau coefficients for RK45
    static constexpr double RK45A21 = 1.0/4.0;
    static constexpr double RK45A31 = 3.0/32.0;
    static constexpr double RK45A32 = 9.0/32.0;
    static constexpr double RK45A41 = 1932.0/2197.0;
    static constexpr double RK45A42 = -7200.0/2197.0;
    static constexpr double RK45A43 = 7296.0/2197.0;
    static constexpr double RK45A51 = 439.0/216.0;
    static constexpr double RK45A52 = -8.0;
    static constexpr double RK45A53 = 3680.0/513.0;
    static constexpr double RK45A54 = -845.0/4104.0;
    static constexpr double RK45A61 = -8.0/27.0;
    static constexpr double RK45A62 = 2.0;
    static constexpr double RK45A63 = -3544.0/2565.0;
    static constexpr double RK45A64 = 1859.0/4104.0;
    static constexpr double RK45A65 = -11.0/40.0;

    // 4th order solution coefficients
    static constexpr double RK45B1 = 25.0/216.0;
    static constexpr double RK45B3 = 1408.0/2565.0;
    static constexpr double RK45B4 = 2197.0/4104.0;
    static constexpr double RK45B5 = -1.0/5.0;
    static constexpr double RK45B6 = 0.0;

    // 5th order solution coefficients
    static constexpr double RK45C1 = 16.0/135.0;
    static constexpr double RK45C3 = 6656.0/12825.0;
    static constexpr double RK45C4 = 28561.0/56430.0;
    static constexpr double RK45C5 = -9.0/50.0;

    CanteraGas* gas;
    ConfigOptions* options;
    
    std::vector<double> state;
    double tStart;
    double tNow;
    
    std::vector<double> dYdt;
    std::vector<double> wDot;
    std::vector<double> hk;
    double rho;
    double cp;
    double dTdt;
    double qDot;
    size_t nSteps;

    // RK45 workspace vectors
    std::vector<double> k1, k2, k3, k4, k5, k6;
    std::vector<double> temp, temp2, yError;

    // Integration parameters
    double atol;
    double rtol;
    double hmin;
    double hmax;
    double safety;
};