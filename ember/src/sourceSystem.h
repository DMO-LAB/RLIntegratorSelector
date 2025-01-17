#pragma once

#include "mathUtils.h"
#include "sundialsUtils.h"
#include "qssintegrator.h"
#include "quasi2d.h"
#include "callback.h"
#include "sourceSystem.h"
#include "rkintegrator.h"
#include "bdfintegrator.h"
#include <arkode/arkode.h>            // For ARKStepEvolve
#include <arkode/arkode_erkstep.h>    // For explicit RK methods
#include <nvector/nvector_serial.h>   // For N_Vector
#include <sunmatrix/sunmatrix_dense.h> // For SUNMatrix
#include <sunlinsol/sunlinsol_dense.h> // For dense linear solver
#include "seulexintegrator.h"
#include <boost/numeric/odeint.hpp>
// #include "odepack.hpp" // ODEPACK C++ binding

class PerfTimer;
class CanteraGas;
class ScalarFunction;
class ConfigOptions;

//! Base class used to integrate the chemical source term at a single point.
class SourceSystem
{
public:
    SourceSystem();
    virtual ~SourceSystem() {}

    //! Set the initial condition for integrator
    //! @param tInitial  integrator start time
    //! @param uu        tangential velocity
    //! @param tt        temperature
    //! @param yy        vector of species mass fractions
    virtual void setState(double tInitial, double uu, double tt,
                          const dvec& yy) = 0;

    //! Take as many steps as needed to reach `tf`.
    //! `tf` is relative to `tInitial`.
    virtual int integrateToTime(double tf) = 0;

    //! Take one step toward `tf` without stepping past it.
    //! `tf` is relative to `tInitial`.
    virtual int integrateOneStep(double tf) = 0;

    //! Current integrator time, relative to `tInitial`.
    virtual double time() const = 0;

    //! Extract current internal integrator state into U, T, and Y
    virtual void unroll_y() = 0;

    virtual void setDebug(bool debug_) { debug = debug_; }

    //! Return a string indicating the number of internal timesteps taken
    virtual std::string getStats() = 0;

    //! Compute thermodynamic properties (density, enthalpies, heat capacities)
    //! from the current state variables.
    void updateThermo();

    //! Calculate the heat release rate associated with a possible external
    //! ignition source.
    double getQdotIgniter(double t);

    //! Set the CanteraGas object to use for thermodynamic and kinetic property
    //! calculations.
    void setGas(CanteraGas* _gas) { gas = _gas; }

    //! Resize internal arrays for a problem of the specified size (`nSpec+2`)
    virtual void initialize(size_t nSpec);

    //! Set integrator tolerances and other parameters
    virtual void setOptions(ConfigOptions& options_);

    //! Set integrator tolerances
    virtual void setTolerances(double reltol, double abstol) = 0;  // Make this pure virtual

    void setTimers(PerfTimer* reactionRates, PerfTimer* thermo,
                   PerfTimer* jacobian);

    //! Set the index j and position x that are represented by this system.
    void setPosition(size_t j, double x);

    //! Set the function used to compute the strain rate as a function of time
    void setStrainFunction(ScalarFunction* f) { strainFunction = f; }

    //! Set the function used to compute the reaction rate multiplier
    void setRateMultiplierFunction(ScalarFunction* f) { rateMultiplierFunction = f; }

    //! Set the function used to compute the heat loss rate to the environment
    void setHeatLossFunction(IntegratorCallback* f) { heatLoss = f; }

    //! Set the density of the unburned mixture.
    //! This value appears in the source term of the momentum equation.
    void setRhou(double _rhou) { rhou = _rhou; }

    //! Set all the balanced splitting constants to zero.
    void resetSplitConstants() { splitConst.setZero(); }

    //! Assign the interpolators used for solving quasi-2D problems
    void setupQuasi2d(std::shared_ptr<BilinearInterpolator> vzInterp,
                      std::shared_ptr<BilinearInterpolator> TInterp);

    //! Write the current values of the state variables, formatted to be read by
    //! Python, to the specified stream. Call with `init=true` when first called
    //! to include initializers for the variables.
    virtual void writeState(std::ostream& out, bool init);

    virtual void writeJacobian(std::ostream& out) {};

    double U; //!< tangential velocity
    double T; //!< temperature
    dvec Y; //!< species mass fraction

    //! Extra constant term introduced by splitting
    dvec splitConst;

protected:
    bool debug;
    ConfigOptions* options;

    //! Cantera data
    CanteraGas* gas;

    //! Timer for time spent evaluating reaction rates
    PerfTimer* reactionRatesTimer;

    //! Timer for time spent evaluating thermodynamic properties
    PerfTimer* thermoTimer;

    //! Timer for time spent evaluating and factorizing the Jacobian
    PerfTimer* jacobianTimer;

    //! A class that provides the strain rate and its time derivative
    ScalarFunction* strainFunction;

    //! Provides a multiplier (optional) for the production terms
    ScalarFunction* rateMultiplierFunction;

    //! Heat loss rate
    IntegratorCallback* heatLoss;

    size_t nSpec; //!< number of species
    int j; //!< grid index for this system
    double x; //!< grid position for this system
    double rhou; //!< density of the unburned gas
    double qDot; //!< heat release rate per unit volume [W/m^3]
    double qLoss; //!< heat loss to the environment [W/m^3]
    double qDot_external; //!< heat release rate from an external source [W/m^3]

    // Physical properties
    double rho; //!< density [kg/m^3]
    double cp; //!< specific heat capacity (average) [J/kg*K]
    dvec cpSpec; //!< species specific heat capacity [J/mol*K]
    double Wmx; //!< mixture molecular weight [kg/mol]
    dvec W; //!< species molecular weights [kg/kmol]
    dvec hk; //!< species enthalpies [J/kmol]

    //! Flag set to 'true' when solving a quasi-2D problem with prescribed
    //! velocity and temperature fields.
    bool quasi2d;

    //! An interpolator for computing the axial (z) velocity when solving a
    //! quasi-2D problem
    std::shared_ptr<BilinearInterpolator> vzInterp;

    //! An interpolator for computing the temperature when solving a
    //! quasi-2D problem
    std::shared_ptr<BilinearInterpolator> TInterp;
};

//! Represents a system of equations used to integrate the (chemical) source
//! term at a single point using the CVODE integrator.
class SourceSystemCVODE : public SourceSystem, sdODE
{
public:
    SourceSystemCVODE() {}

    //! The ODE function: ydot = f(t,y)
    int f(const realtype t, const sdVector& y, sdVector& ydot);

    //! Calculate the Jacobian matrix: J = df/dy
    int denseJacobian(const realtype t, const sdVector& y,
                      const sdVector& ydot, sdMatrix& J);

    //! A simpler finite difference based Jacobian
    int fdJacobian(const realtype t, const sdVector& y,
                   const sdVector& ydot, sdMatrix& J);

    void setState(double tInitial, double uu, double tt, const dvec& yy);

    int integrateToTime(double tf);
    int integrateOneStep(double tf);
    double time() const;

    //! fill in the current state variables from `y`
    void unroll_y(const sdVector& y, double t);

    //! fill in the current state variables from the integrator state
    void unroll_y() {
        unroll_y(integrator->y, integrator->tInt);
    }

    //! fill in `y` with the values of the current state variables
    void roll_y(sdVector& y) const;

    //! fill in `ydot` with the values of the current time derivatives
    void roll_ydot(sdVector& ydot) const;

    std::string getStats();
    void initialize(size_t nSpec);
    void setOptions(ConfigOptions& options);
    void setTolerances(double reltol, double abstol) override;

    virtual void writeState(std::ostream& out, bool init);

    //! Print the current Jacobian matrix ot the specified stream
    void writeJacobian(std::ostream& out);

    double dUdt; //!< time derivative of the tangential velocity
    double dTdt; //!< time derivative of the temperature
    dvec dYdt; //!< time derivative of the species mass fractions
    dvec wDot; //!< species net production rates [kmol/m^3*s]

private:
    std::unique_ptr<SundialsCvode> integrator;

    SundialsContext sunContext;
};

class SourceSystemQSS : public SourceSystem, QssOde
{
public:
    SourceSystemQSS();

    //! The ODE function: ydot = f(t,y)
    void odefun(double t, const dvec& y, dvec& q, dvec& d,
                bool corrector=false);

    double time() const { return integrator.tn; }

    //! Assign the current state variables from `y`.
    //! The current value of the temperature is not updated during corrector
    //! iterations.
    void unroll_y(const dvec& y, bool corrector=false);

    //! Assign the state variables using the current integrator state.
    void unroll_y() { unroll_y(integrator.y); }

    //! fill in `y` with current state variables.
    void roll_y(dvec& y) const;

    //! fill in `q` and `d` with current creation and destruction rates for
    //! each component. The net time derivative is `ydot = q - d`.
    void roll_ydot(dvec& q, dvec& d) const;

    void initialize(size_t nSpec);
    void setOptions(ConfigOptions& options);
    void setTolerances(double reltol, double abstol) override;
    void setState(double tStart, double uu, double tt, const dvec& yy);
    int integrateToTime(double tf) { return integrator.integrateToTime(tf); }
    int integrateOneStep(double tf) { return integrator.integrateOneStep(tf); }

    virtual std::string getStats();

    // creation and destruction rates of state variables
    double dUdtQ; //!< tangential velocity "creation" rate
    double dUdtD; //!< tangential velocity "destruction" rate
    double dTdtQ; //!< temperature "creation" rate
    double dTdtD; //!< temperature "destruction" rate
    dvec dYdtQ; //!< species mass fraction creation rate
    dvec dYdtD; //!< species mass fraction destruction rate

    double tCall; //!< the last time at which odefun was called
    dvec wDotQ, wDotD; //!< species production / destruction rates [kmol/m^3*s]

private:
    QssIntegrator integrator;
};



//! Implementation of SourceSystem using RK23 (Bogacki-Shampine) integrator
class SourceSystemRK23 : public SourceSystem, public QssOde
{
public:
    SourceSystemRK23();
    
    //! Initialize arrays based on number of species
    virtual void initialize(size_t nSpec) override;
    
    //! Set integrator options from configuration
    virtual void setOptions(ConfigOptions& opts) override;

    virtual void setTolerances(double reltol, double abstol) override;
    
    //! Set the initial state
    virtual void setState(double tStart, double u, double T, const dvec& Y) override;
    
    //! Advance solution from current time to tf
    virtual int integrateToTime(double tf) override;
    
    //! Advance solution by one step without going past tf
    virtual int integrateOneStep(double tf) override;
    
    //! Get current time
    virtual double time() const override { return integrator.tn; }
    
    //! Unroll current state
    virtual void unroll_y() override { unroll_y(integrator.y); }
    
    //! Get performance statistics
    virtual std::string getStats() override;
    
    //! Write additional state information (for debugging)
    virtual void writeState(std::ostream& out, bool init=false) override;

    //! Implementation of ODE function required by QssOde
    virtual void odefun(double t, const dvec& y, dvec& q, dvec& d, bool corrector=false) override;

protected:
    RK23Integrator integrator;
    
    //! Convert full state vector to component values
    void unroll_y(const dvec& y);
    
    //! Convert component values to full state vector
    void roll_y(dvec& y) const;
    
    //! Convert component derivatives to full derivative vector
    void roll_ydot(dvec& ydot) const;

    dvec dYdt; //!< Species mass fraction derivatives
    dvec wDot; //!< Species production rates
    double dUdt; //!< Velocity derivative
    double dTdt; //!< Temperature derivative
    double qDot; //!< Heat release rate
    double qDot_external; //!< External heat source term
};

// Forward-declare wrapper function
static int f_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data);

class SourceSystemARK : public SourceSystem
{
public:
    SourceSystemARK();
    ~SourceSystemARK();

    // Override basic methods
    void initialize(size_t new_nSpec) override;
    void setOptions(ConfigOptions& opts) override;
    void setTolerances(double reltol, double abstol) override;
    void setState(double tInitial, double uu, double tt, const dvec& yy) override;
    int  integrateOneStep(double tf) override;
    int  integrateToTime(double tf) override;
    double time() const override { return tNow - tStart; }

    virtual void unroll_y() override { unroll_y(nv_y); }
    
    // SUNDIALS callback
    int f(realtype t, N_Vector y, N_Vector ydot);

    // I/O, debug, etc.
    void writeState(std::ostream& out, bool init) override;
    std::string getStats() override;
    
private:
    // Internal helpers
    void unroll_y(N_Vector y);
    void roll_y(N_Vector y) const;
    void roll_ydot(N_Vector ydot) const;

    // SUNDIALS data
    SUNContext    sunContext;  // main SUNDIALS context
    void*         arkMem;      // ARKode integrator memory
    N_Vector      nv_y;        // state vector
    N_Vector      nv_abstol;   // absolute tolerance vector
    
    double tStart;  // absolute start time
    double tNow;    // current absolute time

    // Local scratch
    dvec dYdt;      // species rates
    dvec wDot;      // reaction rates
    double dUdt;
    double dTdt;
    double qDot;
    double qDot_external;
};

// The wrapper that ARKode calls to evaluate f(t, y, ydot):
static int f_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    auto* sys = static_cast<SourceSystemARK*>(user_data);
    return sys->f(t, y, ydot);
}


class SourceSystemBoostRK;

struct SystemWrapper {
    using state_type = std::vector<double>;
    SourceSystemBoostRK* source;
    
    SystemWrapper(SourceSystemBoostRK* src) : source(src) {}
    void operator()(const state_type& y, state_type& dydt, double t);
};

class SourceSystemBoostRK : public SourceSystem 
{
public:
    using base_stepper_type = boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>>;
    using controlled_stepper_type = boost::numeric::odeint::controlled_runge_kutta<base_stepper_type>;
    using state_type = std::vector<double>;

    friend struct SystemWrapper;
    
    SourceSystemBoostRK();
    
    void initialize(size_t nSpec) override;
    void setOptions(ConfigOptions& opts) override;
    void setTolerances(double reltol, double abstol) override;
    void setState(double tInitial, double uu, double tt, const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;
    void unroll_y() override;
    std::string getStats() override;
    
    void writeState(std::ostream& out, bool init) override;

private:
    void unroll_y(const state_type& y);
    void roll_y(state_type& y) const; 
    void roll_ydot(state_type& dydt) const;

    std::unique_ptr<SystemWrapper> system_wrapper;
    controlled_stepper_type stepper;
    state_type state;
    
    double tStart;
    double tNow; 
    dvec dYdt;
    dvec wDot;
    double dUdt;
    double dTdt;
    double qDot;
    double qDot_external;
    size_t nSteps;
};

// class SourceSystemSEULEX : public SourceSystem {
// public:
//     SourceSystemSEULEX();
    
//     void initialize(size_t nSpec) override;
//     void setOptions(ConfigOptions& opts) override;
//     void setState(double tInitial, double uu, double tt, const dvec& yy) override;
//     int integrateToTime(double tf) override;
//     int integrateOneStep(double tf) override;
//     double time() const override { return tNow - tStart; }
//     void unroll_y() override;
//     std::string getStats() override;
    
// private:
//     odepack::DLSODE solver;
//     std::vector<double> state;
//     std::vector<double> abstol;
//     std::vector<double> reltol;
    
//     static void f_wrapper(int* n, double* t, double* y, double* ydot, void* data);
//     static void jac_wrapper(int* n, double* t, double* y, double* ml, 
//                           double* mu, double* pd, int* nrowpd, void* data);
    
//     void f(double t, const std::vector<double>& y, std::vector<double>& ydot);
//     void jac(double t, const std::vector<double>& y, std::vector<std::vector<double>>& J);
    
//     dvec dYdt;
//     dvec wDot;
//     double dUdt;
//     double dTdt;
//     double qDot;
//     double qDot_external;
//     double tStart;
//     double tNow;
// };
