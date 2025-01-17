#include "ZeroDSourceSystem.h"
#include "readConfig.h"
#include "debugUtils.h"
#include "mathUtils.h"

// Constructor
ZeroDSourceCVODE::ZeroDSourceCVODE() 
    : gas(nullptr)
    , options(nullptr)
    , dTdt(0.0)
    , qDot(0.0)
{
    sunContext = std::make_shared<SundialsContext>();
    integrator = std::make_unique<SundialsCvode>(0); // Size set in initialize()
}

void ZeroDSourceCVODE::initialize(size_t nSpec_) {
    nSpec = nSpec_;
    Y.resize(nSpec);
    dYdt.resize(nSpec);
    wDot.resize(nSpec);
    hk.resize(nSpec);
    
    integrator.reset(new SundialsCvode(nSpec + 1)); // +1 for temperature
    integrator->setODE(static_cast<sdODE*>(this));
    
    // CVODE settings
    integrator->linearMultistepMethod = CV_BDF;
    integrator->maxNumSteps = 10000;
    integrator->setBandwidth(-1, -1);  // Dense Jacobian

}

void ZeroDSourceCVODE::setGas(CanteraGas* gas_) {
    gas = gas_;
}

void ZeroDSourceCVODE::setOptions(ConfigOptions& options_) {
    options = &options_;
    
    if (!integrator) {
        throw DebugException("CVODE integrator not initialized");
    }
    
    // Integration tolerances
    integrator->abstol[0] = options->cvode_abstol_temperature;  // Temperature tolerance
    for (size_t k = 0; k < nSpec; k++) {
        integrator->abstol[k + 1] = options->cvode_abstol_species;
    }
    integrator->reltol = options->cvode_reltol;
    integrator->minStep = options->cvode_minStep;
}

void ZeroDSourceCVODE::setState(double tInitial, double tt, const dvec& yy) {
    if (!integrator) {
        throw DebugException("CVODE integrator not initialized");
    }
    
    integrator->t0 = tInitial;
    T = tt;
    Y = yy;
    
    sdVector& y = integrator->y;
    y[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        y[k + 1] = Y[k];
    }
    
    integrator->initialize();
}

int ZeroDSourceCVODE::f(const realtype t, const sdVector& y, sdVector& ydot) {
    unroll_y(y);
    
    // Chemistry
    gas->setStateMass(Y, T);
    gas->getReactionRates(wDot);
    updateThermo();
    
    // Energy equation
    qDot = -(wDot * hk).sum();
    dTdt = qDot/(rho*cp);
    
    // Species equations
    dvec W(nSpec);
    gas->getMolecularWeights(W);
    dYdt = wDot.array() * W.array() / rho;
    
    roll_ydot(ydot);
    return 0;
}

int ZeroDSourceCVODE::denseJacobian(const realtype t, const sdVector& y, 
                                   const sdVector& ydot, sdMatrix& J) 
{
    return fdJacobian(t, y, ydot, J);
}

int ZeroDSourceCVODE::fdJacobian(const realtype t, const sdVector& y,
                                const sdVector& ydot, sdMatrix& J)
{
    double eps = sqrt(DBL_EPSILON);
    sdVector yplusdy(y.length(), *sunContext);
    sdVector ydot2(y.length(), *sunContext);
    size_t nVars = nSpec + 1;  // +1 for temperature

    for (size_t i = 0; i < nVars; i++) {
        // Copy current state
        for (size_t k = 0; k < nVars; k++) {
            yplusdy[k] = y[k];
        }

        // Perturb the i-th component
        double dy = std::max(abs(y[i]) * eps, eps);
        yplusdy[i] += dy;

        // Get derivatives at perturbed state
        f(t, yplusdy, ydot2);

        // Calculate column i of the Jacobian
        for (size_t k = 0; k < nVars; k++) {
            J(k,i) = (ydot2[k] - ydot[k])/dy;
        }
    }

    return 0;
}

void ZeroDSourceCVODE::updateThermo() {
    gas->getEnthalpies(hk);
    rho = gas->getDensity();
    cp = gas->getSpecificHeatCapacity();
}

void ZeroDSourceCVODE::unroll_y(const sdVector& y) {
    T = y[0];
    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = y[k + 1];
    }
}

void ZeroDSourceCVODE::roll_y(sdVector& y) const {
    y[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        y[k + 1] = Y[k];
    }
}

void ZeroDSourceCVODE::roll_ydot(sdVector& ydot) const {
    ydot[0] = dTdt;
    for (size_t k = 0; k < nSpec; k++) {
        ydot[k + 1] = dYdt[k];
    }
}

int ZeroDSourceCVODE::integrateToTime(double tf) {
    try {
        return integrator->integrateToTime(integrator->t0 + tf);
    } catch (const std::exception& e) {
        return -1;
    }
}

int ZeroDSourceCVODE::integrateOneStep(double tf) {
    return integrator->integrateOneStep(integrator->t0 + tf);
}

double ZeroDSourceCVODE::time() const {
    return integrator->tInt - integrator->t0;
}

// QSS Implementation
ZeroDSourceQSS::ZeroDSourceQSS()
    : gas(nullptr)
    , options(nullptr)
    , dTdtQ(0.0)
    , dTdtD(0.0)
    , qDot(0.0)
{
    integrator.setOde(static_cast<QssOde*>(this));
}

void ZeroDSourceQSS::initialize(size_t nSpec_) {
    nSpec = nSpec_;
    integrator.initialize(nSpec + 1); // +1 for temperature
    
    Y.resize(nSpec);
    dYdtQ.setConstant(nSpec, 0);
    dYdtD.setConstant(nSpec, 0);
    wDotQ.resize(nSpec);
    wDotD.resize(nSpec);
    hk.resize(nSpec);
}

void ZeroDSourceQSS::setGas(CanteraGas* gas_) {
    gas = gas_;
}

void ZeroDSourceQSS::setOptions(ConfigOptions& options_) {
    options = &options_;
    
    integrator.epsmin = options->qss_epsmin;
    integrator.epsmax = options->qss_epsmax;
    integrator.dtmin = options->qss_dtmin;
    integrator.dtmax = options->qss_dtmax;
    integrator.itermax = options->qss_iterationCount;
    integrator.abstol = options->qss_abstol;
    integrator.stabilityCheck = options->qss_stabilityCheck;
    integrator.ymin.setConstant(nSpec + 1, options->qss_minval);
}

void ZeroDSourceQSS::setState(double tInitial, double tt, const dvec& yy) {
    dvec yIn(nSpec + 1);
    T = tt;
    Y = yy;
    
    yIn[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        yIn[k + 1] = Y[k];
    }
    
    integrator.setState(yIn, tInitial);
}

void ZeroDSourceQSS::odefun(double t, const dvec& y, dvec& q, dvec& d, bool corrector) {
    unroll_y(y, corrector);
    
    gas->setStateMass(Y, T);
    gas->getCreationRates(wDotQ);
    gas->getDestructionRates(wDotD);
    
    if (!corrector) {
        updateThermo();
    }
    
    qDot = -((wDotQ - wDotD) * hk).sum();
    
    dTdtQ = qDot/(rho*cp);
    dTdtD = 0.0;
    
    dvec W(nSpec);
    gas->getMolecularWeights(W);
    dYdtQ = wDotQ.array() * W.array() / rho;
    dYdtD = wDotD.array() * W.array() / rho;
    
    roll_ydot(q, d);
}

void ZeroDSourceQSS::updateThermo() {
    gas->getEnthalpies(hk);
    rho = gas->getDensity();
    cp = gas->getSpecificHeatCapacity();
}

void ZeroDSourceQSS::unroll_y(const dvec& y, bool corrector) {
    if (!corrector) {
        T = y[0];
    }
    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = y[k + 1];
    }
}

void ZeroDSourceQSS::roll_y(dvec& y) const {
    y[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        y[k + 1] = Y[k];
    }
}

void ZeroDSourceQSS::roll_ydot(dvec& q, dvec& d) const {
    q[0] = dTdtQ;
    d[0] = dTdtD;
    
    for (size_t k = 0; k < nSpec; k++) {
        q[k + 1] = dYdtQ[k];
        d[k + 1] = dYdtD[k];
    }
}

int ZeroDSourceQSS::integrateToTime(double tf) {
    return integrator.integrateToTime(tf);
}

int ZeroDSourceQSS::integrateOneStep(double tf) {
    return integrator.integrateOneStep(tf);
}

double ZeroDSourceQSS::time() const {
    return integrator.tn;
}

// BoostRK Implementation
ZeroDSourceBoostRK::ZeroDSourceBoostRK()
    : gas(nullptr)
    , options(nullptr)
    , stepper(boost::numeric::odeint::make_controlled(1.0e-8, 1.0e-6, base_stepper_type()))
    , tStart(0.0)
    , tNow(0.0)
    , dTdt(0.0)
    , qDot(0.0)
    , nSteps(0)
{
}

void ZeroDSourceBoostRK::initialize(size_t nSpec_) {
    nSpec = nSpec_;
    Y.resize(nSpec);
    dYdt.resize(nSpec);
    wDot.resize(nSpec);
    hk.resize(nSpec);
    state.resize(nSpec + 1); // +1 for temperature
}

void ZeroDSourceBoostRK::setGas(CanteraGas* gas_) {
    gas = gas_;
}

void ZeroDSourceBoostRK::setOptions(ConfigOptions& options_) {
    options = &options_;
    stepper = boost::numeric::odeint::make_controlled(
        options->rk23AbsTol,
        options->rk23RelTol,
        base_stepper_type()
    );
}

void ZeroDSourceBoostRK::setState(double tInitial, double tt, const dvec& yy) {
    tStart = tInitial;
    tNow = tInitial;
    T = tt;
    Y = yy;
    
    roll_y(state);
}

void ZeroDSourceBoostRK::operator()(const state_type& y, state_type& dydt, double t) {
    unroll_y(y);
    
    gas->setStateMass(Y, T);
    gas->getReactionRates(wDot);
    updateThermo();
    
    // Calculate heat release
    qDot = -(wDot * hk).sum();
    dTdt = qDot/(rho*cp);
    
    dvec W(nSpec);
    gas->getMolecularWeights(W);
    dYdt = wDot.array() * W.array() / rho;
    
    roll_ydot(dydt);
}

void ZeroDSourceBoostRK::updateThermo() {
    gas->getEnthalpies(hk);
    rho = gas->getDensity();
    cp = gas->getSpecificHeatCapacity();
}

void ZeroDSourceBoostRK::unroll_y(const state_type& y) {
    T = y[0];
    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = y[k + 1];
    }
}

void ZeroDSourceBoostRK::roll_y(state_type& y) const {
    y[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        y[k + 1] = Y[k];
    }
}

void ZeroDSourceBoostRK::roll_ydot(state_type& dydt) const {
    dydt[0] = dTdt;
    for (size_t k = 0; k < nSpec; k++) {
        dydt[k + 1] = dYdt[k];
    }
}

int ZeroDSourceBoostRK::integrateToTime(double tf) {
    try {
        boost::numeric::odeint::integrate_adaptive(
            stepper,
            std::ref(*this),
            state,
            tNow - tStart,
            tf,
            options->rk23MinTimestep,
            [this](const state_type& state, double t) { nSteps++; }
        );
        
        tNow = tStart + tf;
        unroll_y(state);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int ZeroDSourceBoostRK::integrateOneStep(double tf) {
    try {
        boost::numeric::odeint::integrate_const(
            stepper,
            std::ref(*this),
            state,
            tNow - tStart,
            tf,
            options->rk23MinTimestep,
            [this](const state_type& state, double t) { nSteps++; }
        );
        
        tNow = tStart + tf;
        unroll_y(state);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

double ZeroDSourceBoostRK::time() const {
    return tNow - tStart;
}

ZeroDSourceRK45::ZeroDSourceRK45()
    : gas(nullptr)
    , options(nullptr)
    , tStart(0.0)
    , tNow(0.0)
    , dTdt(0.0)
    , qDot(0.0)
    , nSteps(0)
    , atol(1e-8)
    , rtol(1e-6)
    , hmin(1e-10)
    , hmax(1.0)
    , safety(0.9)
{
}

void ZeroDSourceRK45::initialize(size_t nSpec_) {
    nSpec = nSpec_;
    const size_t n = nSpec + 1; // +1 for temperature

    // Main arrays
    Y.resize(nSpec);
    state.resize(n);
    dYdt.resize(nSpec);
    wDot.resize(nSpec);
    hk.resize(nSpec);

    // RK temporary arrays
    k1.resize(n);
    k2.resize(n);
    k3.resize(n);
    k4.resize(n);
    k5.resize(n);
    k6.resize(n);
    temp.resize(n);
    temp2.resize(n);
    yError.resize(n);
}

void ZeroDSourceRK45::setGas(CanteraGas* gas_) {
    gas = gas_;
}

void ZeroDSourceRK45::setOptions(ConfigOptions& options_) {
    options = &options_;
    atol = options->rk45AbsTol;
    rtol = options->rk45RelTol;
    hmin = options->rk45MinStep;
    hmax = options->rk45MaxStep;
}

void ZeroDSourceRK45::setState(double tInitial, double tt, const dvec& yy) {
    tStart = tInitial;
    tNow = tInitial;
    T = tt;
    Y = yy;
    
    state[0] = T;
    for (size_t k = 0; k < nSpec; k++) {
        state[k + 1] = Y[k];
    }
}

int ZeroDSourceRK45::integrateToTime(double tf) {
    const double dt = tf - (tNow - tStart);
    if (dt <= 0) return 0;

    double h = std::min(dt, hmax); // Initial step size
    double t = tNow - tStart;

    while (t < tf) {
        if (t + h > tf) {
            h = tf - t;
        }

        // Take step and estimate error
        computeDerivatives(state, k1);
        
        // RK45 stages
        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45A21 * k1[i]);
        }
        computeDerivatives(temp, k2);

        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45A31 * k1[i] + RK45A32 * k2[i]);
        }
        computeDerivatives(temp, k3);

        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45A41 * k1[i] + RK45A42 * k2[i] + RK45A43 * k3[i]);
        }
        computeDerivatives(temp, k4);

        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45A51 * k1[i] + RK45A52 * k2[i] + RK45A53 * k3[i] + RK45A54 * k4[i]);
        }
        computeDerivatives(temp, k5);

        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45A61 * k1[i] + RK45A62 * k2[i] + RK45A63 * k3[i] + RK45A64 * k4[i] + RK45A65 * k5[i]);
        }
        computeDerivatives(temp, k6);

        // Compute error estimate
        double error = 0.0;
        for (size_t i = 0; i < state.size(); i++) {
            temp[i] = state[i] + h * (RK45B1 * k1[i] + RK45B3 * k3[i] + RK45B4 * k4[i] + RK45B5 * k5[i] + RK45B6 * k6[i]);
            temp2[i] = state[i] + h * (RK45C1 * k1[i] + RK45C3 * k3[i] + RK45C4 * k4[i] + RK45C5 * k5[i]);
            yError[i] = std::abs(temp[i] - temp2[i]);
            
            const double sc = atol + rtol * std::max(std::abs(state[i]), std::abs(temp[i]));
            error = std::max(error, yError[i]/sc);
        }

        // Accept or reject step
        if (error <= 1.0) {
            t += h;
            state = temp;
            nSteps++;
        }

        // Compute new step size
        double h_new = h * safety * std::pow(1.0/error, 0.2);
        h = std::min(std::max(h_new, hmin), hmax);
    }

    // Update temperature and mass fractions
    T = state[0];
    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = state[k + 1];
    }
    tNow = tStart + t;

    return 0;
}

int ZeroDSourceRK45::integrateOneStep(double tf) {
    // For one step, we'll use a fixed step size
    double h = std::min(tf - (tNow - tStart), hmax);
    
    computeDerivatives(state, k1);
    
    // RK45 stages (4th order only for efficiency)
    for (size_t i = 0; i < state.size(); i++) {
        temp[i] = state[i] + h * (RK45A21 * k1[i]);
    }
    computeDerivatives(temp, k2);

    for (size_t i = 0; i < state.size(); i++) {
        temp[i] = state[i] + h * (RK45A31 * k1[i] + RK45A32 * k2[i]);
    }
    computeDerivatives(temp, k3);

    for (size_t i = 0; i < state.size(); i++) {
        temp[i] = state[i] + h * (RK45A41 * k1[i] + RK45A42 * k2[i] + RK45A43 * k3[i]);
    }
    computeDerivatives(temp, k4);

    // Update solution using 4th order method
    for (size_t i = 0; i < state.size(); i++) {
        state[i] += h * (RK45B1 * k1[i] + RK45B3 * k3[i] + RK45B4 * k4[i]);
    }

    // Update temperature and mass fractions
    T = state[0];
    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = state[k + 1];
    }
    tNow += h;
    nSteps++;

    return 0;
}

double ZeroDSourceRK45::time() const {
    return tNow - tStart;
}

void ZeroDSourceRK45::computeDerivatives(const std::vector<double>& y, 
                                       std::vector<double>& dy) 
{
    // Unpack state
    double temp = y[0];
    std::vector<double> mass_fracs(nSpec);
    for (size_t k = 0; k < nSpec; k++) {
        mass_fracs[k] = y[k + 1];
    }

    // Compute derivatives
    gas->setStateMass(mass_fracs.data(), temp);
    gas->getReactionRates(wDot.data());
    updateThermo();

    // Energy equation
    qDot = 0.0;
    for (size_t k = 0; k < nSpec; k++) {
        qDot -= wDot[k] * hk[k];
    }
    dTdt = qDot/(rho*cp);

    // Species equations
    std::vector<double> W(nSpec);
    gas->getMolecularWeights(W.data());
    for (size_t k = 0; k < nSpec; k++) {
        dYdt[k] = wDot[k] * W[k] / rho;
    }

    // Pack derivatives
    dy[0] = dTdt;
    for (size_t k = 0; k < nSpec; k++) {
        dy[k + 1] = dYdt[k];
    }
}

void ZeroDSourceRK45::updateThermo() {
    gas->getEnthalpies(hk.data());
    rho = gas->getDensity();
    cp = gas->getSpecificHeatCapacity();
}