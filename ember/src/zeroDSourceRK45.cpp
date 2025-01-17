// #include "ZeroDSourceRK45.h"
// #include "debugUtils.h"
// #include <algorithm>
// #include <cmath>

// ZeroDSourceRK45::ZeroDSourceRK45()
//     : gas(nullptr)
//     , options(nullptr)
//     , tStart(0.0)
//     , tNow(0.0)
//     , dTdt(0.0)
//     , qDot(0.0)
//     , nSteps(0)
//     , atol(1e-8)
//     , rtol(1e-6)
//     , hmin(1e-10)
//     , hmax(1.0)
//     , safety(0.9)
// {
// }

// void ZeroDSourceRK45::initialize(size_t nSpec_) {
//     nSpec = nSpec_;
//     const size_t n = nSpec + 1; // +1 for temperature

//     // Main arrays
//     Y.resize(nSpec);
//     state.resize(n);
//     dYdt.resize(nSpec);
//     wDot.resize(nSpec);
//     hk.resize(nSpec);

//     // RK temporary arrays
//     k1.resize(n);
//     k2.resize(n);
//     k3.resize(n);
//     k4.resize(n);
//     k5.resize(n);
//     k6.resize(n);
//     temp.resize(n);
//     temp2.resize(n);
//     yError.resize(n);
// }

// void ZeroDSourceRK45::setGas(CanteraGas* gas_) {
//     gas = gas_;
// }

// void ZeroDSourceRK45::setOptions(ConfigOptions& options_) {
//     options = &options_;
//     atol = options->rk45AbsTol;
//     rtol = options->rk45RelTol;
//     hmin = options->rk45MinStep;
//     hmax = options->rk45MaxStep;
// }

// void ZeroDSourceRK45::setState(double tInitial, double tt, const dvec& yy) {
//     tStart = tInitial;
//     tNow = tInitial;
//     T = tt;
//     Y = yy;
    
//     state[0] = T;
//     for (size_t k = 0; k < nSpec; k++) {
//         state[k + 1] = Y[k];
//     }
// }

// int ZeroDSourceRK45::integrateToTime(double tf) {
//     const double dt = tf - (tNow - tStart);
//     if (dt <= 0) return 0;

//     double h = std::min(dt, hmax); // Initial step size
//     double t = tNow - tStart;

//     while (t < tf) {
//         if (t + h > tf) {
//             h = tf - t;
//         }

//         // Take step and estimate error
//         computeDerivatives(state, k1);
        
//         // RK45 stages
//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45A21 * k1[i]);
//         }
//         computeDerivatives(temp, k2);

//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45A31 * k1[i] + RK45A32 * k2[i]);
//         }
//         computeDerivatives(temp, k3);

//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45A41 * k1[i] + RK45A42 * k2[i] + RK45A43 * k3[i]);
//         }
//         computeDerivatives(temp, k4);

//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45A51 * k1[i] + RK45A52 * k2[i] + RK45A53 * k3[i] + RK45A54 * k4[i]);
//         }
//         computeDerivatives(temp, k5);

//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45A61 * k1[i] + RK45A62 * k2[i] + RK45A63 * k3[i] + RK45A64 * k4[i] + RK45A65 * k5[i]);
//         }
//         computeDerivatives(temp, k6);

//         // Compute error estimate
//         double error = 0.0;
//         for (size_t i = 0; i < state.size(); i++) {
//             temp[i] = state[i] + h * (RK45B1 * k1[i] + RK45B3 * k3[i] + RK45B4 * k4[i] + RK45B5 * k5[i] + RK45B6 * k6[i]);
//             temp2[i] = state[i] + h * (RK45C1 * k1[i] + RK45C3 * k3[i] + RK45C4 * k4[i] + RK45C5 * k5[i]);
//             yError[i] = std::abs(temp[i] - temp2[i]);
            
//             const double sc = atol + rtol * std::max(std::abs(state[i]), std::abs(temp[i]));
//             error = std::max(error, yError[i]/sc);
//         }

//         // Accept or reject step
//         if (error <= 1.0) {
//             t += h;
//             state = temp;
//             nSteps++;
//         }

//         // Compute new step size
//         double h_new = h * safety * std::pow(1.0/error, 0.2);
//         h = std::min(std::max(h_new, hmin), hmax);
//     }

//     // Update temperature and mass fractions
//     T = state[0];
//     for (size_t k = 0; k < nSpec; k++) {
//         Y[k] = state[k + 1];
//     }
//     tNow = tStart + t;

//     return 0;
// }

// int ZeroDSourceRK45::integrateOneStep(double tf) {
//     // For one step, we'll use a fixed step size
//     double h = std::min(tf - (tNow - tStart), hmax);
    
//     computeDerivatives(state, k1);
    
//     // RK45 stages (4th order only for efficiency)
//     for (size_t i = 0; i < state.size(); i++) {
//         temp[i] = state[i] + h * (RK45A21 * k1[i]);
//     }
//     computeDerivatives(temp, k2);

//     for (size_t i = 0; i < state.size(); i++) {
//         temp[i] = state[i] + h * (RK45A31 * k1[i] + RK45A32 * k2[i]);
//     }
//     computeDerivatives(temp, k3);

//     for (size_t i = 0; i < state.size(); i++) {
//         temp[i] = state[i] + h * (RK45A41 * k1[i] + RK45A42 * k2[i] + RK45A43 * k3[i]);
//     }
//     computeDerivatives(temp, k4);

//     // Update solution using 4th order method
//     for (size_t i = 0; i < state.size(); i++) {
//         state[i] += h * (RK45B1 * k1[i] + RK45B3 * k3[i] + RK45B4 * k4[i]);
//     }

//     // Update temperature and mass fractions
//     T = state[0];
//     for (size_t k = 0; k < nSpec; k++) {
//         Y[k] = state[k + 1];
//     }
//     tNow += h;
//     nSteps++;

//     return 0;
// }

// double ZeroDSourceRK45::time() const {
//     return tNow - tStart;
// }

// void ZeroDSourceRK45::computeDerivatives(const std::vector<double>& y, 
//                                        std::vector<double>& dy) 
// {
//     // Unpack state
//     double temp = y[0];
//     std::vector<double> mass_fracs(nSpec);
//     for (size_t k = 0; k < nSpec; k++) {
//         mass_fracs[k] = y[k + 1];
//     }

//     // Compute derivatives
//     gas->setStateMass(mass_fracs.data(), temp);
//     gas->getReactionRates(wDot.data());
//     updateThermo();

//     // Energy equation
//     qDot = -(wDot * hk).sum();
//     dTdt = qDot/(rho*cp);

//     // Species equations
//     std::vector<double> W(nSpec);
//     gas->getMolecularWeights(W.data());
//     for (size_t k = 0; k < nSpec; k++) {
//         dYdt[k] = wDot[k] * W[k] / rho;
//     }

//     // Pack derivatives
//     dy[0] = dTdt;
//     for (size_t k = 0; k < nSpec; k++) {
//         dy[k + 1] = dYdt[k];
//     }
// }

// void ZeroDSourceRK45::updateThermo() {
//     gas->getEnthalpies(hk.data());
//     rho = gas->getDensity();
//     cp = gas->getSpecificHeatCapacity();
// }