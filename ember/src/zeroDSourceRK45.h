// #pragma once

// #include "sourceSystem.h"
// #include "mathUtils.h"
// #include "cantera/base/Solution.h"
// #include "chemistry0d.h"
// #include <vector>

// class ZeroDSourceRK45 : public ZeroDSourceSystem {
// public:
//     ZeroDSourceRK45();
    
//     // ZeroDSourceSystem interface implementation
//     void initialize(size_t nSpec) override;
//     void setGas(CanteraGas* gas_) override;
//     void setOptions(ConfigOptions& options_) override;
//     void setState(double tInitial, double tt, const dvec& yy) override;
//     int integrateToTime(double tf) override;
//     int integrateOneStep(double tf) override;
//     double time() const override;

// private:
//     void computeDerivatives(const std::vector<double>& y, std::vector<double>& dy);
//     void updateThermo();

//     // Butcher tableau coefficients for RK45
//     static constexpr double RK45A21 = 1.0/4.0;
//     static constexpr double RK45A31 = 3.0/32.0;
//     static constexpr double RK45A32 = 9.0/32.0;
//     static constexpr double RK45A41 = 1932.0/2197.0;
//     static constexpr double RK45A42 = -7200.0/2197.0;
//     static constexpr double RK45A43 = 7296.0/2197.0;
//     static constexpr double RK45A51 = 439.0/216.0;
//     static constexpr double RK45A52 = -8.0;
//     static constexpr double RK45A53 = 3680.0/513.0;
//     static constexpr double RK45A54 = -845.0/4104.0;
//     static constexpr double RK45A61 = -8.0/27.0;
//     static constexpr double RK45A62 = 2.0;
//     static constexpr double RK45A63 = -3544.0/2565.0;
//     static constexpr double RK45A64 = 1859.0/4104.0;
//     static constexpr double RK45A65 = -11.0/40.0;

//     // 4th order solution coefficients
//     static constexpr double RK45B1 = 25.0/216.0;
//     static constexpr double RK45B3 = 1408.0/2565.0;
//     static constexpr double RK45B4 = 2197.0/4104.0;
//     static constexpr double RK45B5 = -1.0/5.0;
//     static constexpr double RK45B6 = 0.0;

//     // 5th order solution coefficients
//     static constexpr double RK45C1 = 16.0/135.0;
//     static constexpr double RK45C3 = 6656.0/12825.0;
//     static constexpr double RK45C4 = 28561.0/56430.0;
//     static constexpr double RK45C5 = -9.0/50.0;

//     CanteraGas* gas;
//     ConfigOptions* options;
    
//     std::vector<double> state;
//     double tStart;
//     double tNow;
    
//     std::vector<double> dYdt;
//     std::vector<double> wDot;
//     std::vector<double> hk;
//     double rho;
//     double cp;
//     double dTdt;
//     double qDot;
//     size_t nSteps;

//     // RK45 workspace vectors
//     std::vector<double> k1, k2, k3, k4, k5, k6;
//     std::vector<double> temp, temp2, yError;

//     // Integration parameters
//     double atol;
//     double rtol;
//     double hmin;
//     double hmax;
//     double safety;
// };