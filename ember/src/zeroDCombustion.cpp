
#include "ZeroDCombustion.h"
#include "readConfig.h"
#include "debugUtils.h"
#include "mathUtils.h"

ZeroDCombustion::ZeroDCombustion() 
    : nSpec(0)
    , T(300.0)
    , tStart(0.0)
    , tNow(0.0)
    , hasOptions(false)
    , isInitialized(false)
{
    try {
        initializeTimers();
        gas = std::make_unique<CanteraGas>();
    } catch (const std::exception& e) {
        logFile.write("Error in ZeroDCombustion constructor: " + std::string(e.what()));
        throw;
    }
}

void ZeroDCombustion::initializeTimers() {
    reactionRatesTimer = std::make_unique<PerfTimer>();
    thermoTimer = std::make_unique<PerfTimer>();
    jacobianTimer = std::make_unique<PerfTimer>();
}

ZeroDCombustion::~ZeroDCombustion() {
    integrator.reset();
    gas.reset();
}

void ZeroDCombustion::setOptions(const ConfigOptions& opts) {
    options = opts;
    hasOptions = true;
    if (gas) {
        gas->setOptions(opts);
    }
}

void ZeroDCombustion::initialize() {
    if (!hasOptions) {
        throw DebugException("Cannot initialize - options not set");
    }
    
    try {
        gas->initialize();
        nSpec = gas->nSpec;
        Y.resize(nSpec);

        // Set initial composition to pure N2
        gas->thermo->setState_TPX(T, options.pressure, "N2:1.0");
        gas->getMassFractions(Y.data());
        
        std::string integratorType = options.chemistryIntegrator;
        if (integratorType.empty()) {
            integratorType = "qss"; // Default to QSS if not specified
        }
        createIntegrator(integratorType);
        
        isInitialized = true;
    } catch (const std::exception& e) {
        logFile.write("Error during initialization: " + std::string(e.what()));
        throw;
    }
}

void ZeroDCombustion::createIntegrator(const std::string& type) {
    try {
        if (type == "cvode") {
            integrator = std::make_unique<ZeroDSourceCVODE>();
        } else if (type == "qss") {
            integrator = std::make_unique<ZeroDSourceQSS>();
        } else if (type == "boostRK") {
            integrator = std::make_unique<ZeroDSourceBoostRK>();
        } else if (type == "rk45") {
            integrator = std::make_unique<ZeroDSourceRK45>();
        } else {
            throw DebugException("Unknown integrator type: " + type);
        }

        integrator->setGas(gas.get());
        integrator->initialize(nSpec);
        integrator->setOptions(options);
        
        // Initialize integrator state
        updateIntegratorState();
        
    } catch (const std::exception& e) {
        logFile.write("Error creating integrator: " + std::string(e.what()));
        throw;
    }
}

void ZeroDCombustion::checkState() const {
    if (!hasOptions) throw DebugException("Options not set");
    if (!isInitialized) throw DebugException("System not initialized");
    if (!gas) throw DebugException("No valid CanteraGas instance");
    if (!integrator) throw DebugException("No valid integrator instance");
}

void ZeroDCombustion::setState(const double temperature, const std::vector<double>& massFractions) {
    checkState();
    
    if (massFractions.size() != nSpec) {
        throw DebugException("Invalid number of species in mass fraction array");
    }
    
    T = temperature;
    Y = massFractions;
    updateIntegratorState();
}

void ZeroDCombustion::setIntegratorType(const std::string& type) {
    checkState();
    createIntegrator(type);
}

void ZeroDCombustion::updateIntegratorState() {
    if (!integrator) throw DebugException("No valid integrator");
    
    // Update gas state
    gas->setStateMass(Y.data(), T);
    
    // Set integrator state using the new ZeroDSourceSystem interface
    integrator->setState(tNow, T, Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size()));
}

int ZeroDCombustion::integrateToTime(const double tf) {
    checkState();
    
    if (tf <= time()) return 0;
    
    try {
        int status = integrator->integrateToTime(tf);
        
        if (status >= 0) {
            T = integrator->T;
            Y.resize(nSpec);
            Eigen::Map<Eigen::VectorXd>(Y.data(), nSpec) = integrator->Y;
            tNow = tStart + tf;
        }
        
        return status;
    } catch (const std::exception& e) {
        logFile.write("Integration error: " + std::string(e.what()));
        throw;
    }
}

int ZeroDCombustion::integrateOneStep(const double tf) {
    checkState();
    if (tf <= time()) return 0;
    
    try {
        int status = integrator->integrateOneStep(tf);
        
        if (status >= 0) {
            T = integrator->T;
            Y.resize(nSpec);
            Eigen::Map<Eigen::VectorXd>(Y.data(), nSpec) = integrator->Y;
            tNow = tStart + tf;
        }
        
        return status;
    } catch (const std::exception& e) {
        logFile.write("Integration step error: " + std::string(e.what()));
        throw;
    }
}

std::vector<double> ZeroDCombustion::getReactionRates() const {
    checkState();
    
    std::vector<double> wdot(nSpec);
    gas->setStateMass(Y.data(), T);
    gas->getReactionRates(wdot.data());
    return wdot;
}

double ZeroDCombustion::getHeatReleaseRate() const {
    checkState();
    
    std::vector<double> wdot = getReactionRates();
    std::vector<double> hk(nSpec);
    gas->getEnthalpies(hk.data());
    
    double qdot = 0.0;
    for (size_t k = 0; k < nSpec; k++) {
        qdot -= wdot[k] * hk[k];
    }
    return qdot;
}