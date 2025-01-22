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

    //! Compute thermodynamic properties from the current state variables.
    void updateThermo();

    //! Calculate the heat release rate associated with ignition source.
    double getQdotIgniter(double t);

    //! Set the CanteraGas object for thermo and kinetic properties
    void setGas(CanteraGas* _gas) { gas = _gas; }

    //! Resize internal arrays for a problem of the specified size
    virtual void initialize(size_t nSpec);

    //! Set integrator tolerances and other parameters
    virtual void setOptions(ConfigOptions& options_);

    //! Set integrator tolerances
    virtual void setTolerances(double reltol, double abstol) = 0;

    void setTimers(PerfTimer* reactionRates, PerfTimer* thermo,
                   PerfTimer* jacobian);

    //! Set the index j and position x that are represented by this system.
    void setPosition(size_t j, double x);

    //! Set the strain rate function
    void setStrainFunction(ScalarFunction* f) { strainFunction = f; }

    //! Set the reaction rate multiplier function
    void setRateMultiplierFunction(ScalarFunction* f) { rateMultiplierFunction = f; }

    //! Set the heat loss function
    void setHeatLossFunction(IntegratorCallback* f) { heatLoss = f; }

    //! Set the density of the unburned mixture
    void setRhou(double _rhou) { rhou = _rhou; }

    //! Set all balanced splitting constants to zero
    void resetSplitConstants() { splitConst.setZero(); }

    //! Set up quasi-2D interpolators
    void setupQuasi2d(std::shared_ptr<BilinearInterpolator> vzInterp,
                      std::shared_ptr<BilinearInterpolator> TInterp);

    //! Write current state variables
    virtual void writeState(std::ostream& out, bool init);

    //! Write Jacobian matrix
    virtual void writeJacobian(std::ostream& out) {}

    double U; //!< tangential velocity
    double T; //!< temperature
    dvec Y; //!< species mass fraction
    dvec splitConst; //!< splitting constants

protected:
    bool debug;
    ConfigOptions* options;
    CanteraGas* gas;
    PerfTimer* reactionRatesTimer;
    PerfTimer* thermoTimer;
    PerfTimer* jacobianTimer;
    ScalarFunction* strainFunction;
    ScalarFunction* rateMultiplierFunction;
    IntegratorCallback* heatLoss;
    size_t nSpec;
    int j;
    double x;
    double rhou;
    double qDot;
    double qLoss;
    double qDot_external;
    double rho;
    double cp;
    dvec cpSpec;
    double Wmx;
    dvec W;
    dvec hk;
    bool quasi2d;
    std::shared_ptr<BilinearInterpolator> vzInterp;
    std::shared_ptr<BilinearInterpolator> TInterp;
};

//! CVODE integrator implementation
class SourceSystemCVODE : public SourceSystem, sdODE
{
public:
    SourceSystemCVODE() {}

    // Implement sdODE interface
    int f(const realtype t, const sdVector& y, sdVector& ydot) override;
    int denseJacobian(const realtype t, const sdVector& y,
                      const sdVector& ydot, sdMatrix& J) override;
    int fdJacobian(const realtype t, const sdVector& y,
                   const sdVector& ydot, sdMatrix& J);

    // Implement SourceSystem interface
    void setState(double tInitial, double uu, double tt, 
                  const dvec& yy) override;
    int integrateToTime(double tf) override;
    int integrateOneStep(double tf) override;
    double time() const override;
    void unroll_y() override;
    std::string getStats() override;
    void initialize(size_t nSpec) override;
    void setOptions(ConfigOptions& options) override;
    void writeState(std::ostream& out, bool init) override;
    void writeJacobian(std::ostream& out) override;
    void setTolerances(double reltol, double abstol) override;

    // Helper functions
    void unroll_y(const sdVector& y, double t);
    void roll_y(sdVector& y) const;
    void roll_ydot(sdVector& ydot) const;

    double dUdt; //!< time derivative of tangential velocity
    double dTdt; //!< time derivative of temperature
    dvec dYdt; //!< time derivative of species mass fractions
    dvec wDot; //!< species net production rates [kmol/m^3*s]

private:
    std::unique_ptr<SundialsCvode> integrator;
    SundialsContext sunContext;
};

//! QSS integrator implementation
class SourceSystemQSS : public SourceSystem, QssOde
{
public:
    SourceSystemQSS();

    // Implement QssOde interface
    void odefun(double t, const dvec& y, dvec& q, dvec& d,
                bool corrector=false) override;

    // Implement SourceSystem interface
    double time() const override { return integrator.tn; }
    void unroll_y() override { unroll_y(integrator.y); }
    void initialize(size_t nSpec) override;
    void setOptions(ConfigOptions& options) override;
    void setTolerances(double reltol, double abstol) override;
    void setState(double tStart, double uu, double tt,
                  const dvec& yy) override;
    int integrateToTime(double tf) override { 
        return integrator.integrateToTime(tf); 
    }
    int integrateOneStep(double tf) override { 
        return integrator.integrateOneStep(tf); 
    }
    std::string getStats() override;

    // Helper functions
    void unroll_y(const dvec& y, bool corrector=false);
    void roll_y(dvec& y) const;
    void roll_ydot(dvec& q, dvec& d) const;

    // Creation and destruction rates
    double dUdtQ;
    double dUdtD;
    double dTdtQ;
    double dTdtD;
    dvec dYdtQ;
    dvec dYdtD;
    double tCall;
    dvec wDotQ, wDotD;

private:
    QssIntegrator integrator;
};


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
    
    // Implement SourceSystem interface
    void initialize(size_t nSpec) override;
    void setOptions(ConfigOptions& opts) override;
    void setTolerances(double reltol, double abstol) override;
    void setState(double tInitial, double uu, double tt,
                  const dvec& yy) override;
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