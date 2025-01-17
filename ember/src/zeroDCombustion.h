#ifndef ZEROD_COMBUSTION_H
#define ZEROD_COMBUSTION_H

#include "ZeroDSourceSystem.h"
#include "chemistry0d.h"
#include "debugUtils.h"
#include "readConfig.h"
#include "perfTimer.h"
#include <memory>
#include <string>
#include <vector>

class ZeroDCombustion {
public:
    ZeroDCombustion();
    ~ZeroDCombustion();

    void setOptions(const ConfigOptions& options);
    void initialize();
    
    void setState(const double T, const std::vector<double>& Y);
    int integrateToTime(const double tf);
    int integrateOneStep(const double tf);
    
    double getTemperature() const { return T; }
    std::vector<double> getMassFractions() const { return Y; }
    
    void setIntegratorType(const std::string& integratorType);
    
    std::vector<double> getReactionRates() const;
    double getHeatReleaseRate() const;
    
    double time() const { return tNow - tStart; }

private:
    // Core components
    std::unique_ptr<CanteraGas> gas;
    std::unique_ptr<ZeroDSourceSystem> integrator;
    
    // Performance timers
    std::unique_ptr<PerfTimer> reactionRatesTimer;
    std::unique_ptr<PerfTimer> thermoTimer;
    std::unique_ptr<PerfTimer> jacobianTimer;
    
    // State variables
    double T;
    std::vector<double> Y;
    size_t nSpec;
    
    // Time tracking
    double tStart;
    double tNow;
    
    // Configuration
    ConfigOptions options;
    bool hasOptions;
    bool isInitialized;

    // Helper functions
    void createIntegrator(const std::string& type);
    void updateIntegratorState();
    void checkState() const;
    void initializeTimers();
};

#endif // ZEROD_COMBUSTION_H