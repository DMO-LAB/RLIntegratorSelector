// // #include "seulexIntegrator.h"
// // #include <algorithm>
// // #include <cmath>
// // #include <stdexcept>

// // SEULEXIntegrator::SEULEXIntegrator()
// // {
// //     initializeSequence();
// // }

// // void SEULEXIntegrator::initialize(size_t size) 
// // {
// //     n = size;
// //     y.resize(n);
// //     ydot.resize(n);
// //     jac.resize(n, n);
// //     iterMatrix = MatrixXd::Zero(n, n);  // Initialize as MatrixXd
// //     work.resize(n);
    
// //     // Initialize extrapolation table
// //     extrapolationTable.resize(maxOrder);
// //     for (auto& vec : extrapolationTable) {
// //         vec.resize(n);
// //     }
// // }

// // void SEULEXIntegrator::setState(const dvec& yIn, double t0)
// // {
// //     y = yIn;
// //     tn = t0;
// //     tstart = t0;
// //     nSteps = 0;
// //     nReject = 0;
// //     nJacEvals = 0;
// //     nFuncEvals = 0;
// // }

// // void SEULEXIntegrator::initializeSequence()
// // {
// //     // Initialize the step sequence (Harmonious sequence)
// //     sequence = {2, 4, 6, 8, 10, 12, 14, 16};
// //     // Ensure we don't exceed maxOrder
// //     if (sequence.size() > static_cast<size_t>(maxOrder)) {
// //         sequence.resize(maxOrder);
// //     }
// // }

// // void SEULEXIntegrator::computeJacobian(double t, const dvec& y)
// // {
// //     jacobian(t, y, jac);
// //     nJacEvals++;
// // }

// // void SEULEXIntegrator::linearImplicitEulerStep(const dvec& yin, double dt, dvec& yout)
// // {
// //     // Evaluate RHS
// //     rhs(tn, yin, ydot);
// //     nFuncEvals++;

// //     // Form iteration matrix: (I - dt*J)
// //     MatrixXd I = MatrixXd::Identity(n, n);
// //     MatrixXd J = jac.matrix();  // Convert to matrix format
// //     iterMatrix = I - dt * J;

// //     // Solve linear system
// //     work = dt * ydot;
// //     if (!solveLinearSystem(iterMatrix, work, yout)) {
// //         throw std::runtime_error("Linear solver failed in SEULEX step");
// //     }
// //     yout += yin;
// // }

// // bool SEULEXIntegrator::solveLinearSystem(const MatrixXd& A, const dvec& b, dvec& x)
// // {
// //     // Convert A to Matrix format if it isn't already
// //     MatrixXd A_matrix = A;
    
// //     // Convert b to Matrix format (column vector)
// //     Eigen::VectorXd b_vector = b.matrix();
    
// //     // Compute LU decomposition
// //     lu.compute(A_matrix);
    
// //     // Solve the system
// //     Eigen::VectorXd x_vector = lu.solve(b_vector);
    
// //     // Convert solution back to Array format
// //     x = x_vector.array();
    
// //     // Check if solve was successful by computing residual
// //     double relative_error = (A_matrix * x_vector - b_vector).norm() / b_vector.norm();
// //     return relative_error < 1e-10;  // Use appropriate tolerance
// // }

// // void SEULEXIntegrator::extrapolate(int k, const std::vector<dvec>& table, dvec& result)
// // {
// //     // Perform polynomial extrapolation using the Neville algorithm
// //     std::vector<dvec> x(k + 1);
// //     for (int j = 0; j <= k; j++) {
// //         x[j] = table[j];
// //     }

// //     for (int m = 1; m <= k; m++) {
// //         for (int i = 0; i <= k - m; i++) {
// //             double factor = static_cast<double>(sequence[i + m]) / 
// //                           static_cast<double>(sequence[i]);
// //             x[i] = (factor * x[i + 1] - x[i]) / (factor - 1.0);
// //         }
// //     }
// //     result = x[0];
// // }

// // double SEULEXIntegrator::estimateError(const std::vector<dvec>& table, int k)
// // {
// //     // Compute relative error estimate from extrapolation table
// //     dvec errorVec = (table[k] - table[k-1]).abs();
// //     dvec scaleVec = (abstol + reltol * table[k].abs());
// //     return (errorVec / scaleVec).maxCoeff();
// // }

// // double SEULEXIntegrator::predictNewStepSize(double error, double dt, int k)
// // {
// //     // Predict new step size based on error estimate
// //     double scale = safety * std::pow(error, -1.0 / (2*k + 1));
// //     scale = std::min(MAX_SCALE, std::max(MIN_SCALE, scale));
// //     double newDt = dt * scale;
    
// //     // Enforce bounds
// //     newDt = std::min(dtmax, std::max(dtmin, newDt));
// //     return newDt;
// // }

// // int SEULEXIntegrator::integrateToTime(double tf)
// // {
// //     if (tf == tn) return 0;
    
// //     // Initial step size
// //     double dt = std::min(dtmax, tf - tn);
    
// //     while (tn < tf) {
// //         // Don't step past tf
// //         dt = std::min(dt, tf - tn);
        
// //         // Compute Jacobian for this step
// //         computeJacobian(tn, y);
        
// //         bool stepAccepted = false;
// //         int attempts = 0;
// //         const int maxAttempts = 3;

// //         while (!stepAccepted && attempts < maxAttempts) {
// //             try {
// //                 // Build extrapolation table
// //                 for (int k = 0; k < maxOrder; k++) {
// //                     int numSubsteps = sequence[k];
// //                     double subDt = dt / numSubsteps;
                    
// //                     dvec yTemp = y;
// //                     for (int i = 0; i < numSubsteps; i++) {
// //                         linearImplicitEulerStep(yTemp, subDt, work);
// //                         yTemp = work;
// //                     }
// //                     extrapolationTable[k] = yTemp;

// //                     // Check convergence starting with k=1
// //                     if (k > 0) {
// //                         double error = estimateError(extrapolationTable, k);
// //                         if (error < 1.0) {
// //                             // Step accepted - extrapolate to get final solution
// //                             extrapolate(k, extrapolationTable, y);
// //                             dt = predictNewStepSize(error, dt, k);
// //                             stepAccepted = true;
// //                             break;
// //                         }
// //                     }
// //                 }

// //                 if (!stepAccepted) {
// //                     // Reduce step size and try again
// //                     dt *= 0.5;
// //                     nReject++;
// //                 }
// //             }
// //             catch (const std::exception&) {
// //                 dt *= 0.5;
// //                 nReject++;
// //             }
// //             attempts++;
// //         }

// //         if (!stepAccepted) {
// //             return -1;  // Step failed
// //         }

// //         tn += dt;
// //         nSteps++;
        
// //         if (nSteps >= maxSteps) {
// //             return 1;  // Max steps exceeded
// //         }
// //     }
    
// //     return 0;  // Success
// // }

// // int SEULEXIntegrator::integrateOneStep(double tf)
// // {
// //     if (tf == tn) return 0;
    
// //     // Choose step size for this step
// //     double dt = std::min(dtmax, tf - tn);
    
// //     // Compute Jacobian
// //     computeJacobian(tn, y);
    
// //     bool stepAccepted = false;
// //     int attempts = 0;
// //     const int maxAttempts = 3;
    
// //     while (!stepAccepted && attempts < maxAttempts) {
// //         try {
// //             // Build extrapolation table
// //             for (int k = 0; k < maxOrder; k++) {
// //                 int numSubsteps = sequence[k];
// //                 double subDt = dt / numSubsteps;
                
// //                 dvec yTemp = y;
// //                 for (int i = 0; i < numSubsteps; i++) {
// //                     linearImplicitEulerStep(yTemp, subDt, work);
// //                     yTemp = work;
// //                 }
// //                 extrapolationTable[k] = yTemp;

// //                 // Check convergence starting with k=1
// //                 if (k > 0) {
// //                     double error = estimateError(extrapolationTable, k);
// //                     if (error < 1.0) {
// //                         // Step accepted - extrapolate to get final solution
// //                         extrapolate(k, extrapolationTable, y);
// //                         dt = predictNewStepSize(error, dt, k);
// //                         stepAccepted = true;
// //                         break;
// //                     }
// //                 }
// //             }

// //             if (!stepAccepted) {
// //                 // Reduce step size and try again
// //                 dt *= 0.5;
// //                 nReject++;
// //             }
// //         }
// //         catch (const std::exception&) {
// //             dt *= 0.5;
// //             nReject++;
// //         }
// //         attempts++;
// //     }

// //     if (!stepAccepted) {
// //         return -1;  // Step failed
// //     }

// //     tn += dt;
// //     nSteps++;
    
// //     return (nSteps >= maxSteps) ? 1 : 0;
// // }


// #include "SEULEXIntegrator.h"
// #include <algorithm>
// #include <cmath>
// #include <stdexcept>
// #include <iostream> // (optional) for debug prints

// SEULEXIntegrator::SEULEXIntegrator()
//     : n(0)
// {
//     // We only initialize the sequence array here, but internal vectors
//     // get allocated when initialize(nvar) is called.
//     initializeSequence();
// }

// void SEULEXIntegrator::initialize(size_t nvar)
// {
//     n = nvar;
//     y.resize(n);
//     ydot.resize(n);
//     jac_.resize(n, n);
//     iterMatrix = Eigen::MatrixXd::Zero(n, n);
//     work.resize(n);

//     // Prepare extrapolation table for up to maxOrder
//     extrapolationTable.resize(maxOrder);
//     for (auto &vec : extrapolationTable) {
//         vec.resize(n);
//     }
// }

// void SEULEXIntegrator::setState(const dvec &yIn, double t0)
// {
//     y = yIn;
//     tn = t0;
//     tstart = t0;
//     nSteps = 0;
//     nReject = 0;
//     nJacEvals = 0;
//     nFuncEvals = 0;
// }

// void SEULEXIntegrator::initializeSequence()
// {
//     // Example “harmonious” or “evenly spaced” sequence:
//     // 2,4,6,8,... up to maxOrder entries
//     sequence.clear();
//     int val = 2;
//     for (int i = 0; i < maxOrder; ++i) {
//         sequence.push_back(val);
//         val += 2; // increment by 2
//     }
// }

// /**
//  * @brief Perform a single sub-step of Implicit Euler from tIn to tIn+dt
//  * using Newton iteration. 
//  */
// void SEULEXIntegrator::implicitEulerNewtonSubstep(double tIn,
//                                                   const dvec &yIn,
//                                                   double dt,
//                                                   dvec &yOut)
// {
//     // We'll do a standard Newton iteration:
//     //    y_{m+1} = y_{m} - [I - dt J(t_{n+1}, y_m)]^{-1} * [ y_m - y_in - dt f(t_{n+1}, y_m) ]
//     // but for clarity we'll keep the usual simpler approach:
//     //
//     // We want yOut such that:
//     //   yOut = yIn + dt * f(tIn+dt, yOut)
//     //
//     // Let's define G(yOut) = yOut - yIn - dt * f(tIn+dt, yOut) = 0
//     //
//     // Then Newton iteration: yOut_{m+1} = yOut_m - [dG/dyOut]^(-1) * G(yOut_m)
//     // where dG/dyOut = I - dt * J(tIn+dt, yOut_m).

//     // Start with an initial guess = explicit Euler or simply yIn
//     dvec yGuess = yIn; 
//     double tEval = tIn + dt;

//     for (int m = 0; m < maxNewtonIters; ++m)
//     {
//         // Evaluate f(t, yGuess)
//         rhs(tEval, yGuess, ydot);
//         nFuncEvals++;

//         // Evaluate J(t, yGuess)
//         jacobian(tEval, yGuess, jac_);
//         nJacEvals++;

//         // Construct iteration matrix: A = I - dt * J
//         Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) - dt * jac_.matrix();

//         // Right-hand side of the Newton correction:
//         // R = yGuess - yIn - dt*f(tEval,yGuess)
//         // We want to solve A * delta = -R   for delta
//         dvec R = yGuess - yIn - dt * ydot;
//         R = -R;  // move to the RHS of A*delta = -R

//         dvec delta(n);
//         if (!solveLinearSystem(A, R, delta)) {
//             // If the solver fails, you might want to do something smarter
//             // but let's throw for now
//             throw std::runtime_error("Newton sub-step: linear solver failed");
//         }

//         // Update the guess
//         yGuess += delta;

//         // Check for convergence
//         double relNorm = delta.matrix().norm() / std::max(1e-14, yGuess.matrix().norm());
//         if (relNorm < newtonTol) {
//             // Converged
//             break;
//         }
//     }

//     // yOut is the final guess
//     yOut = yGuess;
// }

// bool SEULEXIntegrator::solveLinearSystem(const Eigen::MatrixXd &A,
//                                          const dvec &b,
//                                          dvec &x)
// {
//     // LU decomposition
//     lu.compute(A);

//     Eigen::VectorXd bx = b.matrix();
//     Eigen::VectorXd sol = lu.solve(bx);

//     // Convert solution back to Array
//     x = sol.array();

//     // Check residual
//     double residual = (A * sol - bx).norm() / std::max(1e-14, bx.norm());
//     if (residual > 1e-11) {
//         // The threshold here could be adapted to problem scale
//         return false;
//     }
//     return true;
// }

// void SEULEXIntegrator::extrapolate(int k,
//                                    const std::vector<dvec> &table,
//                                    dvec &result)
// {
//     // The "Neville" scheme for polynomial extrapolation 
//     // table[0..k] holds successively refined sub-step solutions
//     // We'll do this in-place on a temporary vector of size (k+1).
//     std::vector<dvec> temp(k+1);
//     for (int j = 0; j <= k; j++) {
//         temp[j] = table[j];
//     }

//     for (int m = 1; m <= k; m++) {
//         for (int i = 0; i <= (k - m); i++) {
//             // ratio between sub-step counts: e.g. factor = sequence[i+m]/sequence[i]
//             double factor = static_cast<double>(sequence[i+m]) /
//                             static_cast<double>(sequence[i]);
//             temp[i] = (factor * temp[i+1] - temp[i]) / (factor - 1.0);
//         }
//     }
//     result = temp[0];
// }

// double SEULEXIntegrator::estimateError(const std::vector<dvec> &table, int k)
// {
//     // We'll compare table[k] and table[k-1]
//     // errorVec = abs( table[k] - table[k-1] )
//     // scaleVec = abstol + reltol * abs(table[k])
//     dvec diff = (table[k] - table[k-1]).abs();
//     dvec scale = (table[k].abs() * reltol) + abstol;
//     // maxCoeff() on the ratio
//     double err = (diff / scale).maxCoeff();
//     return err;
// }

// double SEULEXIntegrator::predictNewStepSize(double error, double dt, int k)
// {
//     // Suppose the local order for the k-th extrapolation is (k+1).
//     // So scale factor ~ error^(-1/(k+1)) -- or sometimes -1/(2k+1) for Euler-based
//     // Below we use a typical exponent for an Euler-based method:  -1/(k+1)
//     // or you might see  -1/(k + (some fraction)).
//     // 
//     // The user code had  -1/(2*k + 1). 
//     // We'll keep a typical approach for "extrapolation from order 1,2,..."
//     // but let's stay consistent with the user code if desired:
//     double exponent = -1.0 / (2 * (k+1) - 1); // i.e. -1/(2k+1) if local order = k+1
//     double scale = safety * std::pow(error, exponent);

//     // clamp scale to [MIN_SCALE, MAX_SCALE]
//     scale = std::min(MAX_SCALE, std::max(MIN_SCALE, scale));

//     double newDt = dt * scale;
//     newDt = std::max(dtmin, std::min(dtmax, newDt));
//     return newDt;
// }

// int SEULEXIntegrator::integrateToTime(double tf)
// {
//     if (tf <= tn) {
//         return 0; // nothing to do
//     }

//     // initial step guess
//     double dt = std::min(dtmax, tf - tn);

//     while (tn < tf) {
//         // ensure we do not overshoot
//         dt = std::min(dt, tf - tn);

//         bool stepAccepted = false;
//         int attempts = 0;
//         const int maxAttempts = 10;

//         while (!stepAccepted && attempts < maxAttempts)
//         {
//             try {
//                 // Build the extrapolation table up to maxOrder
//                 for (int k = 0; k < maxOrder; k++)
//                 {
//                     int numSub = sequence[k];
//                     double subDt = dt / double(numSub);

//                     dvec yTemp = y;
//                     // Perform the substeps
//                     for (int i = 0; i < numSub; i++) {
//                         implicitEulerNewtonSubstep(tn + i*subDt, yTemp, subDt, work);
//                         yTemp = work;
//                     }
//                     extrapolationTable[k] = yTemp;

//                     // If k>0, we can check the error estimate
//                     if (k > 0) {
//                         double err = estimateError(extrapolationTable, k);
//                         if (err < 1.0) {
//                             // Accept the step: extrapolate
//                             extrapolate(k, extrapolationTable, y);
//                             double newDt = predictNewStepSize(err, dt, k);

//                             // We finalize the step
//                             tn += dt;
//                             dt = newDt; // update dt for the next step
//                             nSteps++;
//                             stepAccepted = true;
//                             break;
//                         }
//                     }
//                 }

//                 // If we never accepted the step in the for-loop
//                 if (!stepAccepted) {
//                     // reduce dt, increment nReject
//                     dt *= 0.5;
//                     dt = std::max(dt, dtmin);
//                     nReject++;
//                 }
//             }
//             catch (const std::exception &ex) {
//                 // e.g. if the linear solver failed or Newton diverged
//                 // reduce dt and try again
//                 dt *= 0.5;
//                 dt = std::max(dt, dtmin);
//                 nReject++;
//             }
//             attempts++;

//             if (nSteps >= maxSteps) {
//                 return 1; // Max steps exceeded
//             }
//         }

//         if (!stepAccepted) {
//             // we failed to take a step after multiple attempts
//             return -1;
//         }

//         if (std::abs(tf - tn) < 1e-14) {
//             // We reached tf
//             break;
//         }
//     }

//     return 0; // success
// }

// int SEULEXIntegrator::integrateOneStep(double tf)
// {
//     // Similar approach, but only do one step from tn toward tf.
//     if (tf <= tn) {
//         return 0;
//     }
//     double dt = std::min(dtmax, tf - tn);

//     bool stepAccepted = false;
//     int attempts = 0;
//     const int maxAttempts = 10;

//     while (!stepAccepted && attempts < maxAttempts)
//     {
//         try {
//             for (int k = 0; k < maxOrder; k++)
//             {
//                 int numSub = sequence[k];
//                 double subDt = dt / double(numSub);

//                 dvec yTemp = y;
//                 for (int i = 0; i < numSub; i++) {
//                     implicitEulerNewtonSubstep(tn + i*subDt, yTemp, subDt, work);
//                     yTemp = work;
//                 }
//                 extrapolationTable[k] = yTemp;

//                 if (k > 0) {
//                     double err = estimateError(extrapolationTable, k);
//                     if (err < 1.0) {
//                         // Accept step
//                         extrapolate(k, extrapolationTable, y);
//                         tn += dt;
//                         dt = predictNewStepSize(err, dt, k);
//                         nSteps++;
//                         stepAccepted = true;
//                         break;
//                     }
//                 }
//             }

//             if (!stepAccepted) {
//                 // reject, reduce dt
//                 dt *= 0.5;
//                 dt = std::max(dt, dtmin);
//                 nReject++;
//             }
//         }
//         catch(const std::exception &ex) {
//             dt *= 0.5;
//             dt = std::max(dt, dtmin);
//             nReject++;
//         }
//         attempts++;

//         if (nSteps >= maxSteps) {
//             return 1; // max steps exceeded
//         }
//     }

//     if (!stepAccepted) {
//         return -1; // step failed
//     }

//     // We did exactly one step (or reached tf if dt was exactly tf-tn)
//     return 0;
// }



// seulexintegrator.cpp
#include "seulexintegrator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

SEULEXIntegrator::SEULEXIntegrator() : n(0) {
    initializeSequence();
}

void SEULEXIntegrator::initialize(size_t nvar) {
    n = nvar;
    y.resize(n);
    ydot.resize(n);
    jac_.resize(n, n);
    work.resize(n);
    
    extrapolationTable.resize(maxOrder);
    for (auto& vec : extrapolationTable) {
        vec.resize(n);
    }
}

void SEULEXIntegrator::setState(const dvec& yIn, double t0) {
    y = yIn;
    tn = 0.0;
    tstart = t0;
    nSteps = 0;
    nReject = 0;
    nJacEvals = 0;
    nFuncEvals = 0;
}

void SEULEXIntegrator::initializeSequence() {
    // Use a geometric sequence better suited for stiff problems
    sequence.clear();
    int val = 2;
    for (int i = 0; i < maxOrder; ++i) {
        sequence.push_back(val);
        val *= 2; // Geometric growth: 2,4,8,16,...
    }
}

void SEULEXIntegrator::implicitEulerNewtonSubstep(double tIn, const dvec& yIn, 
                                                 double dt, dvec& yOut)
{
    dvec yGuess = yIn;
    double tEval = tIn + dt;
    double lastNorm = 1e10;  // Add convergence monitoring

    for (int m = 0; m < maxNewtonIters; ++m) {
        rhs(tEval, yGuess, ydot);
        nFuncEvals++;

        jacobian(tEval, yGuess, jac_);
        nJacEvals++;

        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n, n) - dt * jac_.matrix();
        dvec R = yGuess - yIn - dt * ydot;
        R = -R;

        dvec delta(n);
        if (!solveLinearSystem(A, R, delta)) {
            throw std::runtime_error("Newton iteration failed: linear solver error");
        }

        yGuess += delta;

        double relNorm = delta.matrix().norm() / 
                        std::max(1e-14, yGuess.matrix().norm());
                        
        // Add divergence check
        if (relNorm > lastNorm * 2) {
            throw std::runtime_error("Newton iteration diverging");
        }
        lastNorm = relNorm;
        
        if (relNorm < newtonTol) {
            yOut = yGuess;
            return;
        }
    }
    
    throw std::runtime_error("Newton iteration failed to converge");
}

bool SEULEXIntegrator::solveLinearSystem(const Eigen::MatrixXd& A,
                                        const dvec& b, dvec& x)
{
    lu.compute(A);
    
    Eigen::VectorXd bx = b.matrix();
    Eigen::VectorXd sol = lu.solve(bx);
    x = sol.array();

    double residual = (A * sol - bx).norm() / std::max(1e-14, bx.norm());
    return residual <= 1e-10;
}

void SEULEXIntegrator::extrapolate(int k, const std::vector<dvec>& table,
                                  dvec& result)
{
    std::vector<dvec> temp(k+1);
    for (int j = 0; j <= k; j++) {
        temp[j] = table[j];
    }

    for (int m = 1; m <= k; m++) {
        for (int i = 0; i <= (k - m); i++) {
            double factor = static_cast<double>(sequence[i+m]) /
                          static_cast<double>(sequence[i]);
            temp[i] = (factor * temp[i+1] - temp[i]) / (factor - 1.0);
        }
    }
    result = temp[0];
}

double SEULEXIntegrator::estimateError(const std::vector<dvec>& table, int k) {
    dvec diff = (table[k] - table[k-1]).abs();
    dvec scale = (table[k].abs() * reltol + abstol).max(1e-30);  // Avoid division by zero
    dvec err = diff / scale;
    
    // Add protection against NaN/Inf
    if (!err.allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    
    return err.maxCoeff();
}

double SEULEXIntegrator::predictNewStepSize(double error, double dt, int k) {
    double order = k + 1;
    double exponent = -1.0 / order;
    
    // Remove error clamping or make it less aggressive
    error = std::max(error, 1e-8);  // Less aggressive minimum
    
    double scale = safety * std::pow(error, exponent);
    // More conservative scaling limits
    scale = std::min(2.0, std::max(0.1, scale));
    
    double newDt = dt * scale;
    return std::max(dtmin, std::min(dtmax, newDt));
}

int SEULEXIntegrator::integrateToTime(double tf) {
    // if (tf <= tn) {
    //     std::cout << tf << " <= " << tn << std::endl;
    //     return 0;
    // }

    double dt = std::min(dtmax, (tf - tn));  // More conservative initial step
    // std::cout << "Initial dt: " << dt << std::endl;

    while (tn < tf) {
        dt = std::min(dt, tf - tn);
        bool stepAccepted = false;
        int attempts = 0;
        const int maxAttempts = 10;

        while (!stepAccepted && attempts < maxAttempts) {
            try {
                for (int k = 0; k < maxOrder; k++) {
                    int numSub = sequence[k];
                    double subDt = dt / double(numSub);

                    dvec yTemp = y;
                    for (int i = 0; i < numSub; i++) {
                        implicitEulerNewtonSubstep(tn + i*subDt, yTemp, subDt, work);
                        yTemp = work;
                    }
                    extrapolationTable[k] = yTemp;

                    if (k > 0) {
                        double err = estimateError(extrapolationTable, k);
                        if (err < 1.0) {
                            extrapolate(k, extrapolationTable, y);
                            dt = predictNewStepSize(err, dt, k);
                            tn += dt;
                            nSteps++;
                            stepAccepted = true;
                            break;
                        }
                    }
                }

                if (!stepAccepted) {
                    dt *= 0.5;
                    dt = std::max(dt, dtmin);
                    nReject++;
                    std::cout << "Rejected step, new dt: " << dt << std::endl;
                }
            }
            catch (const std::exception& ex) {
                dt *= 0.5;
                dt = std::max(dt, dtmin);
                nReject++;
                std::cout << "Rejected step, new dt: " << dt << std::endl;
            }
            attempts++;
        }

        if (!stepAccepted) {
            std::cout << "Step failed" << std::endl;
            return -1;
        }
        if (nSteps >= maxSteps) {
            std::cout << "Max steps exceeded" << std::endl;
            return 1;
        }
        if (std::abs(tf - tn) < 1e-14) {
            // std::cout << "Reached tf" << std::endl;
            break;
        }
    }

    return 0;
}

int SEULEXIntegrator::integrateOneStep(double tf) {
    if (tf <= tn) {
        std::cout << "Nothing to do" << std::endl;
        return 0;
    }
    
    double dt = std::min(dtmax, tf - tn);
    bool stepAccepted = false;
    int attempts = 0;
    const int maxAttempts = 10;

    while (!stepAccepted && attempts < maxAttempts) {
        try {
            for (int k = 0; k < maxOrder; k++) {
                int numSub = sequence[k];
                double subDt = dt / double(numSub);

                dvec yTemp = y;
                for (int i = 0; i < numSub; i++) {
                    implicitEulerNewtonSubstep(tn + i*subDt, yTemp, subDt, work);
                    yTemp = work;
                }
                extrapolationTable[k] = yTemp;

                if (k > 0) {
                    double err = estimateError(extrapolationTable, k);
                    if (err < 1.0) {
                        extrapolate(k, extrapolationTable, y);
                        dt = predictNewStepSize(err, dt, k);
                        tn += dt;
                        nSteps++;
                        stepAccepted = true;
                        break;
                    }
                }
            }

            if (!stepAccepted) {
                dt *= 0.5;
                dt = std::max(dt, dtmin);
                nReject++;
            }
        }
        catch (const std::exception& ex) {
            dt *= 0.5;
            dt = std::max(dt, dtmin);
            nReject++;
        }
        attempts++;
    }

    if (!stepAccepted) return -1;
    if (nSteps >= maxSteps) return 1;

    return 0;
}