#include "sourceSystem.h"
#include "readConfig.h"
#include "perfTimer.h"
#include "sundialsUtils.h"
#include "chemistry0d.h"
#include "scalarFunction.h"
#include "debugUtils.h"
#include "seulexintegrator.h"
#include <boost/format.hpp>
#include <arkode/arkode_erkstep.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h> 
#include <sundials/sundials_types.h>
#include <limits>       // for DBL_EPSILON
#include <cmath>        // for sqrt, etc.
#include <stdexcept>    // for std::runtime_error
#include <boost/numeric/odeint.hpp>
// #include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>

// Instead of SUNFALSE/SUNTRUE
#define ARK_NORMAL   1
#define ARK_ONE_STEP 2



SourceSystem::SourceSystem()
    : U(NaN)
    , T(NaN)
    , debug(false)
    , options(NULL)
    , gas(NULL)
    , strainFunction(NULL)
    , rateMultiplierFunction(NULL)
    , heatLoss(NULL)
    , qLoss(0.0)
    , quasi2d(false)
{
}

void SourceSystem::updateThermo()
{
    thermoTimer->start();
    gas->getEnthalpies(hk);
    rho = gas->getDensity();
    cp = gas->getSpecificHeatCapacity();
    thermoTimer->stop();
}

double SourceSystem::getQdotIgniter(double t)
{
    ConfigOptions& opt = *options;
    if (t >= opt.ignition_tStart &&
        t < opt.ignition_tStart + opt.ignition_duration) {
        return opt.ignition_energy /
                (opt.ignition_stddev * sqrt(2 * M_PI) * opt.ignition_duration) *
                exp(-pow(x - opt.ignition_center, 2) /
                    (2 * pow(opt.ignition_stddev, 2)));
    } else {
        //std::cout << "No ignition source" << std::endl;
        return 0.0;
    }
}

void SourceSystem::setOptions(ConfigOptions& options_)
{
    options = &options_;
}

void SourceSystem::initialize(size_t new_nSpec)
{
    nSpec = new_nSpec;

    Y.setConstant(nSpec, NaN);
    cpSpec.resize(nSpec);
    splitConst.resize(nSpec + 2);
    hk.resize(nSpec);

    W.resize(gas->nSpec); // move this to initialize
    gas->getMolecularWeights(W);
}

void SourceSystem::setTimers
(PerfTimer* reactionRates, PerfTimer* thermo, PerfTimer* jacobian)
{
    reactionRatesTimer = reactionRates;
    thermoTimer = thermo;
    jacobianTimer = jacobian;
}

void SourceSystem::setPosition(size_t _j, double _x)
{
    j = static_cast<int>(_j);
    x = _x;
}

void SourceSystem::setupQuasi2d(std::shared_ptr<BilinearInterpolator> vz_int,
                                std::shared_ptr<BilinearInterpolator> T_int)
{
    quasi2d = true;
    vzInterp = vz_int;
    TInterp = T_int;
}

void SourceSystem::writeState(std::ostream& out, bool init)
{
    if (init) {
        out << "T = []" << std::endl;
        out << "Y = []" << std::endl;
        out << "t = []" << std::endl;
    }

    Eigen::IOFormat fmt(15, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

    out << "T.append(" << T << ")" << std::endl;
    out << "Y.append(" << Y.format(fmt) << ")" << std::endl;
    out << "t.append(" << time() << ")" << std::endl;
}

// ----------------------------------------------------------------------------

void SourceSystemCVODE::initialize(size_t new_nSpec)
{
    SourceSystem::initialize(new_nSpec);
    dYdt.resize(nSpec);
    wDot.resize(nSpec);

    integrator.reset(new SundialsCvode(static_cast<int>(nSpec+2)));
    integrator->setODE(this);
    integrator->linearMultistepMethod = CV_BDF;
    integrator->maxNumSteps = 1000000;
}

// void SourceSystemCVODE::setOptions(ConfigOptions& opts)
// {
//     SourceSystem::setOptions(opts);
//     integrator->abstol[kMomentum] = options->integratorMomentumAbsTol;
//     integrator->abstol[kEnergy] = options->integratorEnergyAbsTol;
//     for (size_t k=0; k<nSpec; k++) {
//         integrator->abstol[kSpecies+k] = options->integratorSpeciesAbsTol;
//     }
//     integrator->reltol = options->integratorRelTol;
//     integrator->minStep = options->integratorMinTimestep;
// }

// void SourceSystemCVODE::setTolerances(double reltol, double abstol)
// {
//     integrator->reltol = reltol;
//     integrator->abstol[kMomentum] = abstol;
//     integrator->abstol[kEnergy] = abstol;
//     for (size_t k=0; k<nSpec; k++) {
//         integrator->abstol[kSpecies+k] = abstol;
//     }
// }

// Move the tolerance setting logic from setOptions to setTolerances
void SourceSystemCVODE::setTolerances(double reltol, double abstol) {
    integrator->reltol = reltol;
    integrator->abstol[kMomentum] = abstol;
    integrator->abstol[kEnergy] = abstol;
    for (size_t k=0; k<nSpec; k++) {
        integrator->abstol[kSpecies+k] = abstol;
    }
}

// Modify setOptions to use setTolerances
void SourceSystemCVODE::setOptions(ConfigOptions& opts) {
    SourceSystem::setOptions(opts);
    setTolerances(options->integratorRelTol, options->integratorSpeciesAbsTol);
    integrator->abstol[kMomentum] = options->integratorMomentumAbsTol;
    integrator->abstol[kEnergy] = options->integratorEnergyAbsTol;
    integrator->minStep = options->integratorMinTimestep;
}

int SourceSystemCVODE::f(const realtype t, const sdVector& y, sdVector& ydot)
{
    unroll_y(y, t);

    reactionRatesTimer->start();
    if (rateMultiplierFunction) {
        gas->setRateMultiplier(rateMultiplierFunction->a(t));
    }
    gas->thermo->setMassFractions_NoNorm(Y.data());
    gas->thermo->setState_TP(T, gas->pressure);
    gas->getReactionRates(wDot);
    reactionRatesTimer->stop();

    updateThermo();

    // *** Calculate the time derivatives
    double scale;
    if (!quasi2d) {
        scale = 1.0;
        dUdt = splitConst[kMomentum];
        qDot_external = getQdotIgniter(t);
        // std::cout << "qqdot: " << qDot_external << std::endl;
        qDot = - (wDot * hk).sum() + qDot_external;
        if (heatLoss && options->alwaysUpdateHeatFlux) {
            std::cout << "Updating heat loss" << std::endl;
            qDot -= heatLoss->eval(x, t, U, T, Y);
        } else {
            qDot -= qLoss;
        }
        dTdt = qDot/(rho*cp) + splitConst[kEnergy];
    } else {
        scale = 1.0/vzInterp->get(x, t);
        dUdt = splitConst[kMomentum];
        dTdt = splitConst[kEnergy];
    }
    dYdt = scale * wDot * W / rho + splitConst.tail(nSpec);

    roll_ydot(ydot);
    return 0;
}

int SourceSystemCVODE::denseJacobian(const realtype t, const sdVector& y,
                                     const sdVector& ydot, sdMatrix& J)
{
    // TODO: Verify that f has just been called so that we don't need to
    // unroll y and compute all the transport properties.

    fdJacobian(t, y, ydot, J);
    return 0;

    double a = strainFunction->a(t);
    double dadt = strainFunction->dadt(t);
    double A = a*a + dadt;

    // Additional properties not needed for the normal function evaluations:
    thermoTimer->start();
    gas->getSpecificHeatCapacities(cpSpec);
    Wmx = gas->getMixtureMolecularWeight();
    thermoTimer->stop();

    // The constant "800" here has been empirically determined to give
    // good performance for typical test cases. This value can have
    // a substantial impact on the convergence rate of the solver.
    double eps = sqrt(DBL_EPSILON)*800;

    // *** Derivatives with respect to temperature
    double TplusdT = T*(1+eps);

    double dwhdT = 0;
    dvec dwdT(nSpec);
    dvec wDot2(nSpec);

    reactionRatesTimer->start();
    gas->setStateMass(Y, TplusdT);
    gas->getReactionRates(wDot2);
    reactionRatesTimer->stop();

    for (size_t k=0; k<nSpec; k++) {
        dwdT[k] = (wDot2[k]-wDot[k])/(TplusdT-T);
        dwhdT += hk[k]*dwdT[k] + cpSpec[k]*wDot[k];
    }

    double drhodT = -rho/T;

    // *** Derivatives with respect to species concentration
    dmatrix dwdY(nSpec, nSpec);
    dvec hdwdY = dvec::Zero(nSpec);
    dvec YplusdY(nSpec);
    dvec drhodY(nSpec);

    double scale = (quasi2d) ? 1.0/vzInterp->get(x, t) : 1.0;

    for (size_t k=0; k<nSpec; k++) {
        YplusdY[k] = (abs(Y[k]) > eps/2) ? Y[k]*(1+eps) : eps;
        reactionRatesTimer->start();
        gas->thermo->setMassFractions_NoNorm(YplusdY.data());
        gas->thermo->setState_TP(T, gas->pressure);

        gas->getReactionRates(wDot2);
        reactionRatesTimer->stop();

        for (size_t i=0; i<nSpec; i++) {
            dwdY(i,k) = (wDot2[i]-wDot[i])/(YplusdY[k]-Y[k]);
            hdwdY[k] += hk[i]*dwdY(i,k);
        }

        drhodY[k] = rho*(W[k]-Wmx)/(W[k]*(1-Y[k]*(1-eps)));
    }

    for (size_t k=0; k<nSpec; k++) {
        for (size_t i=0; i<nSpec; i++) {
            // dSpecies/dY
            J(kSpecies+k, kSpecies+i) = scale *
                (dwdY(k,i)*W[k]/rho - wDot[k]*W[k]*drhodY[i]/(rho*rho));
        }
        if (!quasi2d) {
            // dSpecies/dT
            J(kSpecies+k, kEnergy) = dwdT[k]*W[k]/rho -
                wDot[k]*W[k]*drhodT/(rho*rho);

            // dEnergy/dY
            J(kEnergy, kSpecies+k) = -hdwdY[k]/(rho*cp) -
                qDot*drhodY[k]/(rho*rho*cp);

            // dMomentum/dY
            J(kMomentum, kSpecies+k) = -A*drhodY[k]/(rho*rho);
        }
    }

    if (!quasi2d) {
        // dEnergy/dT
        J(kEnergy, kEnergy) = -dwhdT/(rho*cp) - qDot*drhodT/(rho*rho*cp);

        // dMomentum/dU
        J(kMomentum, kMomentum) = 0;

        // dMomentum/dT
        J(kMomentum, kEnergy) = -A*drhodT/(rho*rho);
    }

    return 0;
}

int SourceSystemCVODE::fdJacobian(const realtype t, const sdVector& y,
                                  const sdVector& ydot, sdMatrix& J)
{
    jacobianTimer->start();
    sdVector yplusdy(y.length(), sunContext);
    sdVector ydot2(y.length(), sunContext);
    size_t nVars = nSpec+2;
    double eps = sqrt(DBL_EPSILON);
    double atol = DBL_EPSILON;

    for (size_t i=0; i<nVars; i++) {
        for (size_t k=0; k<nVars; k++) {
            yplusdy[k] = y[k];
        }
        double dy = (abs(y[i]) > atol) ? abs(y[i])*(eps) : abs(y[i])*eps + atol;
        yplusdy[i] += dy;
        f(t, yplusdy, ydot2);
        for (size_t k=0; k<nVars; k++) {
            J(k,i) = (ydot2[k]-ydot[k])/dy;
        }
    }

    jacobianTimer->stop();

    return 0;
}

void SourceSystemCVODE::setState
(double tInitial, double uu, double tt, const dvec& yy)
{
    integrator->t0 = tInitial;
    integrator->y[kMomentum] = uu;
    integrator->y[kEnergy] = tt;
    Eigen::Map<dvec>(&integrator->y[kSpecies], nSpec) = yy;
    integrator->initialize();
    if (heatLoss && !options->alwaysUpdateHeatFlux) {
        qLoss = heatLoss->eval(x, tInitial, uu, tt, const_cast<dvec&>(yy));
    }
    //std::cout << "Setting state for cvode system" << std::endl;
}

int SourceSystemCVODE::integrateToTime(double tf)
{
    try {
        return integrator->integrateToTime(integrator->t0 + tf);
    } catch (Cantera::CanteraError& err) {
        logFile.write(err.what());
        return -1;
    }
}

int SourceSystemCVODE::integrateOneStep(double tf)
{
    return integrator->integrateOneStep(integrator->t0 + tf);
}

double SourceSystemCVODE::time() const
{
    return integrator->tInt - integrator->t0;
}

void SourceSystemCVODE::unroll_y()
{
    unroll_y(integrator->y, integrator->tInt);
}

void SourceSystemCVODE::unroll_y(const sdVector& y, double t)
{
    if (!quasi2d) {
        T = y[kEnergy];
        U = y[kMomentum];
    } else {
        T = TInterp->get(x, t);
        U = 0;
    }
    Y = Eigen::Map<dvec>(&y[kSpecies], nSpec);
}

void SourceSystemCVODE::roll_y(sdVector& y) const
{
    y[kEnergy] = T;
    y[kMomentum] = U;
    Eigen::Map<dvec>(&y[kSpecies], nSpec) = Y;
}

void SourceSystemCVODE::roll_ydot(sdVector& ydot) const
{
    ydot[kEnergy] = dTdt;
    ydot[kMomentum] = dUdt;
    Eigen::Map<dvec>(&ydot[kSpecies], nSpec) = dYdt;
}

std::string SourceSystemCVODE::getStats()
{
    return (format("%i") % integrator->getNumSteps()).str();
}

void SourceSystemCVODE::writeState(std::ostream& out, bool init)
{
    SourceSystem::writeState(out, init);
    if (init) {
        out << "dTdt = []" << std::endl;
        out << "dYdt = []" << std::endl;
        out << "splitConstT = []" << std::endl;
        out << "splitConstY = []" << std::endl;
    }

    Eigen::IOFormat fmt(15, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

    out << "dTdt.append(" << dTdt << ")" << std::endl;
    out << "dYdt.append(" << dYdt.format(fmt) << ")" << std::endl;
    out << "splitConstT.append(" << splitConst[kEnergy] << ")" << std::endl;
    out << "splitConstY.append(" << splitConst.tail(nSpec).format(fmt) << ")" << std::endl;
}

void SourceSystemCVODE::writeJacobian(std::ostream& out)
{
    int N = integrator->y.length();
    double t = integrator->tInt;
    sdMatrix J(N,N, sunContext);
    sdVector ydot(N, sunContext);
    f(t, integrator->y, ydot);
    denseJacobian(t, integrator->y, ydot, J);

    out << "J = []" << std::endl;
    for (int i=0; i<N; i++) {
        out << "J.append([";
        for (int k=0; k<N; k++) {
            out << boost::format("%.5e, ") % J(i,k);
        }
        out << "])" << std::endl;
    }
}

// ----------------------------------------------------------------------------

SourceSystemQSS::SourceSystemQSS()
{
    integrator.setOde(this);
    dUdtQ = 0;
    dUdtD = 0;
    dTdtQ = 0;
    dTdtD = 0;
}

void SourceSystemQSS::initialize(size_t new_nSpec)
{
    SourceSystem::initialize(new_nSpec);
    integrator.initialize(new_nSpec + 2);

    dYdtQ.setConstant(nSpec, 0);
    dYdtD.setConstant(nSpec, 0);
    wDotD.resize(nSpec);
    wDotQ.resize(nSpec);

    integrator.enforce_ymin[kMomentum] = false;
}

void SourceSystemQSS::setOptions(ConfigOptions& opts)
{
    SourceSystem::setOptions(opts);
    integrator.epsmin = options->qss_epsmin;
    integrator.epsmax = options->qss_epsmax;
    integrator.dtmin = options->qss_dtmin;
    integrator.dtmax = options->qss_dtmax;
    integrator.itermax = options->qss_iterationCount;
    integrator.abstol = options->qss_abstol;
    integrator.stabilityCheck = options->qss_stabilityCheck;
    integrator.ymin.setConstant(nSpec + 2, options->qss_minval);
    integrator.ymin[kMomentum] = -1e4;
}

void SourceSystemQSS::setTolerances(double reltol, double abstol)
{
    integrator.abstol = abstol;
}

void SourceSystemQSS::setState
(double tStart, double uu, double tt, const dvec& yy)
{
    dvec yIn(nSpec + 2);
    yIn << uu, tt, yy;
    integrator.setState(yIn, tStart);
    if (heatLoss && !options->alwaysUpdateHeatFlux) {
        qLoss = heatLoss->eval(x, tStart, uu, tt, const_cast<dvec&>(yy));
    }
    // std::cout << "Setting state for qss system" << std::endl;
}

void SourceSystemQSS::odefun(double t, const dvec& y, dvec& q, dvec& d,
                             bool corrector)
{
    tCall = t;
    unroll_y(y, corrector);

    // *** Update auxiliary data ***
    reactionRatesTimer->start();
    if (rateMultiplierFunction) {
        gas->setRateMultiplier(rateMultiplierFunction->a(t));
    }
    gas->setStateMass(Y, T);
    gas->getCreationRates(wDotQ);
    gas->getDestructionRates(wDotD);
    reactionRatesTimer->stop();

    if (!corrector) {
        updateThermo();
    }

    qDot = - ((wDotQ - wDotD) * hk).sum() + getQdotIgniter(t);

    // *** Calculate the time derivatives
    double scale;
    if (!quasi2d) {
        scale = 1.0;
        dUdtQ = splitConst[kMomentum];
        dUdtD = 0;
        dTdtQ = qDot/(rho*cp) + splitConst[kEnergy];
    } else {
        scale = 1.0/vzInterp->get(x, t);
        dUdtQ = 0;
        dUdtD = 0;
        dTdtQ = 0;
    }

    if (heatLoss && options->alwaysUpdateHeatFlux) {
        dTdtD = heatLoss->eval(x, t, U, T, const_cast<dvec&>(Y)) / (rho*cp);
    } else {
        dTdtD = qLoss / (rho*cp);
    }

    dYdtQ = scale * wDotQ * W / rho + splitConst.tail(nSpec);
    dYdtD = scale * wDotD * W / rho;

    assert(rhou > 0);
    assert(rho > 0);
    assert(U > -1e100 && U < 1e100);
    assert(splitConst[kMomentum] > -1e100 && splitConst[kMomentum] < 1e100);

    assert(dUdtQ > -1e100 && dUdtQ < 1e100);
    assert(dUdtD > -1e100 && dUdtD < 1e100);

    roll_ydot(q, d);
}

void SourceSystemQSS::unroll_y(const dvec& y, bool corrector)
{
    if (!quasi2d) {
        if (!corrector) {
            T = y[kEnergy];
        }
        U = y[kMomentum];
    } else {
        if (!corrector) {
            T = TInterp->get(x, tCall);
        }
        U = 0;
    }
    Y = y.tail(nSpec);
}

void SourceSystemQSS::roll_y(dvec& y) const
{
    y << U, T, Y;
}

void SourceSystemQSS::roll_ydot(dvec& q, dvec& d) const
{
    q << dUdtQ, dTdtQ, dYdtQ;
    d << dUdtD, dTdtD, dYdtD;
}

std::string SourceSystemQSS::getStats()
{
    return (format("%i/%i") % integrator.gcount % integrator.rcount).str();
}

// First define the SystemWrapper implementation
void SystemWrapper::operator()(const state_type& y, state_type& dydt, double t)
{
    source->unroll_y(y);

    source->reactionRatesTimer->start();
    if (source->rateMultiplierFunction) {
        source->gas->setRateMultiplier(source->rateMultiplierFunction->a(t));
    }
    source->gas->thermo->setMassFractions_NoNorm(source->Y.data());
    source->gas->thermo->setState_TP(source->T, source->gas->pressure);
    source->gas->getReactionRates(source->wDot);
    source->reactionRatesTimer->stop();

    source->updateThermo();

    double scale;
    if (!source->quasi2d) {
        scale = 1.0;
        source->dUdt = source->splitConst[kMomentum];
        source->qDot_external = source->getQdotIgniter(t);
        source->qDot = -(source->wDot * source->hk).sum() + source->qDot_external;

        if (source->heatLoss && source->options->alwaysUpdateHeatFlux) {
            source->qDot -= source->heatLoss->eval(source->x, t, source->U, source->T, source->Y);
        } else {
            source->qDot -= source->qLoss;
        }

        source->dTdt = source->qDot/(source->rho*source->cp) + source->splitConst[kEnergy];
    } else {
        scale = 1.0/source->vzInterp->get(source->x, t);
        source->dUdt = source->splitConst[kMomentum];
        source->dTdt = source->splitConst[kEnergy];
    }

    source->dYdt = scale * (source->wDot.array() * source->W.array()) / source->rho +
                  source->splitConst.tail(source->nSpec).array();

    source->roll_ydot(dydt);
}

SourceSystemBoostRK::SourceSystemBoostRK()
    : stepper(boost::numeric::odeint::make_controlled(1.0e-9, 1.0e-6, base_stepper_type()))
    , tStart(0.0)
    , tNow(0.0)
    , dUdt(0.0)
    , dTdt(0.0)
    , qDot(0.0)
    , qDot_external(0.0)
    , nSteps(0)
{
    system_wrapper = std::make_unique<SystemWrapper>(this);
}

void SourceSystemBoostRK::initialize(size_t new_nSpec)
{
    SourceSystem::initialize(new_nSpec);
    dYdt.resize(nSpec);
    wDot.resize(nSpec);
    state.resize(nSpec + 2);
}

void SourceSystemBoostRK::setOptions(ConfigOptions& opts)
{
    SourceSystem::setOptions(opts);
    stepper = boost::numeric::odeint::make_controlled(
        opts.rk23AbsTol,
        opts.rk23RelTol,
        base_stepper_type()
    );
}

void SourceSystemBoostRK::setTolerances(double reltol, double abstol)
{
   stepper = boost::numeric::odeint::make_controlled(reltol, abstol, base_stepper_type());
}

void SourceSystemBoostRK::setState(double tInitial, double uu, double tt, const dvec& yy)
{
    tStart = tInitial;
    tNow = tInitial;
    U = uu;
    T = tt;
    Y = yy;

    roll_y(state);

    if (heatLoss && !options->alwaysUpdateHeatFlux) {
        qLoss = heatLoss->eval(x, tInitial, uu, tt, const_cast<dvec&>(yy));
    }
}

int SourceSystemBoostRK::integrateToTime(double tf)
{
    try {
        boost::numeric::odeint::integrate_adaptive(
            stepper,
            std::ref(*system_wrapper),
            state,
            tNow,
            tf,
            options->rk23MinTimestep
        );

        tNow = tf;
        unroll_y(state);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int SourceSystemBoostRK::integrateOneStep(double tf)
{
    try {
        boost::numeric::odeint::integrate_const(
            stepper,
            std::ref(*system_wrapper),
            state,
            tNow,
            tf,
            options->rk23MinTimestep
        );

        tNow = tf;
        unroll_y(state);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

void SourceSystemBoostRK::unroll_y()
{
    unroll_y(state);
}

void SourceSystemBoostRK::unroll_y(const state_type& y)
{
    if (!quasi2d) {
        U = y[kMomentum];
        T = y[kEnergy];
    } else {
        U = 0.0;
        T = TInterp->get(x, tNow);
    }

    for (size_t k = 0; k < nSpec; k++) {
        Y[k] = y[kSpecies + k];
    }
}

void SourceSystemBoostRK::roll_y(state_type& y) const
{
    y[kMomentum] = U;
    y[kEnergy] = T;
    for (size_t k = 0; k < nSpec; k++) {
        y[kSpecies + k] = Y[k];
    }
}

void SourceSystemBoostRK::roll_ydot(state_type& dydt) const
{
    dydt[kMomentum] = dUdt;
    dydt[kEnergy] = dTdt;
    for (size_t k = 0; k < nSpec; k++) {
        dydt[kSpecies + k] = dYdt[k];
    }
}

double SourceSystemBoostRK::time() const
{
    return (tNow - tStart);
}

std::string SourceSystemBoostRK::getStats()
{
    return "Boost RK solver steps: " + std::to_string(nSteps);
}

void SourceSystemBoostRK::writeState(std::ostream& out, bool init)
{
    SourceSystem::writeState(out, init);

    if (init) {
        out << "dTdt = []" << std::endl;
        out << "dYdt = []" << std::endl;
    }

    Eigen::IOFormat fmt(15, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");
    out << "dTdt.append(" << dTdt << ")" << std::endl;
    out << "dYdt.append(" << dYdt.format(fmt) << ")" << std::endl;
}