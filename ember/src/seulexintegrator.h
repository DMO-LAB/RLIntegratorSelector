// // #pragma once

// // #include <vector>
// // #include <eigen3/Eigen/Dense>

// // using dvec = Eigen::ArrayXd;
// // using dmatrix = Eigen::ArrayXXd;
// // using Eigen::MatrixXd;

// // class SEULEXIntegrator 
// // {
// // public:
// //     SEULEXIntegrator();
// //     ~SEULEXIntegrator() = default;

// //     // Core functions
// //     void initialize(size_t n);
// //     void setState(const dvec& yIn, double t0);
// //     int integrateToTime(double tf);
// //     int integrateOneStep(double tf);

// //     // Configuration parameters
// //     double abstol{1e-10};      // Absolute tolerance
// //     double reltol{1e-6};       // Relative tolerance
// //     double dtmin{1e-15};       // Minimum timestep
// //     double dtmax{1e-4};        // Maximum timestep
// //     double safety{0.9};        // Safety factor for step size control
// //     int maxSteps{10000};       // Maximum number of steps
// //     int maxOrder{8};           // Maximum extrapolation order
    
// //     // Current state
// //     dvec y;                    // Current solution vector
// //     double tn{0.0};           // Current time
// //     double tstart{0.0};       // Initial time
// //     int nSteps{0};            // Number of steps taken
// //     int nReject{0};           // Number of rejected steps
// //     int nJacEvals{0};         // Number of Jacobian evaluations
// //     int nFuncEvals{0};        // Number of function evaluations

// //     // Function pointers for the ODE system
// //     std::function<void(double t, const dvec& y, dvec& ydot)> rhs;
// //     std::function<void(double t, const dvec& y, dmatrix& jac)> jacobian;

// // protected:
// //     // Core implementation methods
// //     void linearImplicitEulerStep(const dvec& yin, double dt, dvec& yout);
// //     double estimateError(const std::vector<dvec>& table, int k);
// //     void computeJacobian(double t, const dvec& y);
// //     void extrapolate(int k, const std::vector<dvec>& table, dvec& result);
    
// //     // Helper functions
// //     void initializeSequence();
// //     double predictNewStepSize(double error, double dt, int k);
// //     bool solveLinearSystem(const MatrixXd& A, const dvec& b, dvec& x);

// //     // Internal work arrays and matrices
// //     size_t n;                          // System size
// //     std::vector<int> sequence;         // Step sequence for extrapolation
// //     std::vector<dvec> extrapolationTable;  // Table for extrapolation
// //     dmatrix jac;                       // Jacobian matrix
// //     MatrixXd iterMatrix;               // Iteration matrix
// //     dvec ydot;                         // RHS evaluation
// //     dvec work;                         // Work array

// //     // LU decomposition workspace
// //     Eigen::PartialPivLU<MatrixXd> lu;

// //     // Constants
// //     static constexpr double MIN_SCALE = 0.2;    // Minimum stepsize scale factor
// //     static constexpr double MAX_SCALE = 6.0;    // Maximum stepsize scale factor
// //     static constexpr double MIN_ACCEPT = 0.1;   // Minimum acceptable error ratio
// // };


// #pragma once

// #include <vector>
// #include <functional>
// #include <eigen3/Eigen/Dense>

// // For convenience in passing around solution vectors and Jacobians
// using dvec    = Eigen::ArrayXd;   // 1D array, dynamic size
// using dmatrix = Eigen::ArrayXXd;  // 2D array, dynamic size

// /**
//  * @class SEULEXIntegrator
//  *
//  * A simple example of a SEULEX (implicit Euler + extrapolation) integrator
//  * for stiff or mildly stiff ODEs. Uses Newton iteration for each sub-step
//  * of the implicit Euler method. Then uses Richardson extrapolation to improve
//  * accuracy.
//  */
// class SEULEXIntegrator
// {
// public:
//     SEULEXIntegrator();
//     ~SEULEXIntegrator() = default;

//     // ------------------------------------------------------------------------
//     // Configuration parameters
//     // ------------------------------------------------------------------------
//     double abstol  {1e-10};      ///< Absolute tolerance
//     double reltol  {1e-6};       ///< Relative tolerance
//     double dtmin   {1e-15};      ///< Minimum timestep
//     double dtmax   {1e-2};       ///< Maximum timestep
//     double safety  {0.9};        ///< Safety factor in step-size control
//     int    maxSteps{10000};      ///< Maximum number of steps
//     int    maxOrder{8};          ///< Maximum extrapolation order
//     int    maxNewtonIters{10};   ///< Maximum Newton iterations per sub-step
//     double newtonTol{1e-12};     ///< Newton iteration (relative) tolerance

//     // Step size scale factors
//     static constexpr double MIN_SCALE = 0.2;  ///< Min scale factor for dt
//     static constexpr double MAX_SCALE = 6.0;  ///< Max scale factor for dt

//     // ------------------------------------------------------------------------
//     // State variables
//     // ------------------------------------------------------------------------
//     dvec   y;       ///< Current solution vector
//     double tn{0.0}; ///< Current time
//     double tstart{0.0};  ///< Initial time
//     int    nSteps{0};    ///< Number of steps taken
//     int    nReject{0};   ///< Number of rejected steps
//     int    nJacEvals{0}; ///< Number of Jacobian evaluations
//     int    nFuncEvals{0};///< Number of function evaluations

//     // User-provided function handles for ODE system
//     std::function<void(double t, const dvec &y, dvec &ydot)> rhs; 
//     std::function<void(double t, const dvec &y, dmatrix &jac)> jacobian;

//     // ------------------------------------------------------------------------
//     // Public interface
//     // ------------------------------------------------------------------------

//     /**
//      * @brief Allocate/resize internal buffers to handle a system of size `nvar`.
//      */
//     void initialize(size_t nvar);

//     /**
//      * @brief Set the initial state (y(t0), t0).
//      */
//     void setState(const dvec &yIn, double t0);

//     /**
//      * @brief Integrate from the current time tn up to time tf.
//      *
//      * @return  0 on success, >0 or <0 if a failure or maximum steps exceeded.
//      */
//     int integrateToTime(double tf);

//     /**
//      * @brief Take one step toward tf (but do not necessarily reach it).
//      *
//      * @return 0 on success, -1 on step failure, +1 if max steps are exceeded.
//      */
//     int integrateOneStep(double tf);

// protected:
//     // ------------------------------------------------------------------------
//     // Internal methods
//     // ------------------------------------------------------------------------

//     /**
//      * @brief Perform an Implicit Euler sub-step with Newton iteration,
//      * from time tIn to tIn+dt. 
//      * 
//      * yIn is the solution at the start of the sub-step,
//      * yOut is the updated solution at tIn+dt.
//      * The iteration is done in-place, yOut becomes the final result.
//      */
//     void implicitEulerNewtonSubstep(double tIn, const dvec &yIn, double dt, dvec &yOut);

//     /**
//      * @brief Polynomial extrapolation of order k using the Neville scheme.
//      *
//      * @param k        index of the highest entry to extrapolate from
//      * @param table    the array of sub-step solutions
//      * @param result   the extrapolated solution
//      */
//     void extrapolate(int k, const std::vector<dvec> &table, dvec &result);

//     /**
//      * @brief Compute a local error estimate from the extrapolation table.
//      *
//      * This uses the difference between table[k] and table[k-1].
//      */
//     double estimateError(const std::vector<dvec> &table, int k);

//     /**
//      * @brief Predict a new step size based on the current error estimate.
//      */
//     double predictNewStepSize(double error, double dt, int k);

//     /**
//      * @brief Solve a linear system A x = b using an LU factorization in Eigen.
//      *
//      * @return true if the solution residual is below a certain tolerance.
//      */
//     bool solveLinearSystem(const Eigen::MatrixXd &A, const dvec &b, dvec &x);

//     /**
//      * @brief Initialize the substep sequence (the "harmonious" sequence).
//      */
//     void initializeSequence();

//     // ------------------------------------------------------------------------
//     // Internal data
//     // ------------------------------------------------------------------------
//     size_t n;                          ///< Dimension of the ODE system
//     std::vector<int> sequence;         ///< Substep counts for extrapolation
//     std::vector<dvec> extrapolationTable; ///< Table for Richardson extrapolation
//     dmatrix jac_;                      ///< Current Jacobian in array form
//     Eigen::MatrixXd iterMatrix;        ///< Re-used matrix for iteration
//     dvec    ydot;                      ///< Temporary for function evaluations
//     dvec    work;                      ///< Temporary vector, e.g. for solves

//     Eigen::PartialPivLU<Eigen::MatrixXd> lu;  ///< LU decomposition object
// };




// seulexintegrator.h
#pragma once

#include <vector>
#include <functional>
#include <eigen3/Eigen/Dense>

using dvec = Eigen::ArrayXd;
using dmatrix = Eigen::ArrayXXd;

class SEULEXIntegrator 
{
public:
    SEULEXIntegrator();
    ~SEULEXIntegrator() = default;

    // Configuration parameters
    double abstol{1e-10};      // Absolute tolerance
    double reltol{1e-6};       // Relative tolerance
    double dtmin{1e-15};       // Minimum timestep
    double dtmax{1e-4};        // Maximum timestep
    double safety{0.9};        // Safety factor for step size control
    int maxSteps{10000};       // Maximum number of steps
    int maxOrder{8};           // Maximum extrapolation order
    int maxNewtonIters{10};    // Maximum Newton iterations per sub-step
    double newtonTol{1e-12};   // Newton iteration tolerance

    // Step size scale factors
    static constexpr double MIN_SCALE = 0.2;
    static constexpr double MAX_SCALE = 6.0;

    // Current state
    dvec y;                    // Current solution
    double tn{0.0};           // Current time
    double tstart{0.0};       // Initial time
    int nSteps{0};           // Number of steps taken
    int nReject{0};          // Number of rejected steps
    int nJacEvals{0};        // Number of Jacobian evaluations
    int nFuncEvals{0};       // Number of function evaluations

    // User-provided functions for ODE system
    std::function<void(double t, const dvec&, dvec&)> rhs;
    std::function<void(double t, const dvec&, dmatrix&)> jacobian;

    // Interface methods
    void initialize(size_t nvar);
    void setState(const dvec& yIn, double t0);
    int integrateToTime(double tf);
    int integrateOneStep(double tf);

protected:
    // Internal methods
    void implicitEulerNewtonSubstep(double tIn, const dvec& yIn, double dt, dvec& yOut);
    void extrapolate(int k, const std::vector<dvec>& table, dvec& result);
    double estimateError(const std::vector<dvec>& table, int k);
    double predictNewStepSize(double error, double dt, int k);
    bool solveLinearSystem(const Eigen::MatrixXd& A, const dvec& b, dvec& x);
    void initializeSequence();

    // Internal data
    size_t n;                          // System dimension
    std::vector<int> sequence;         // Substep counts
    std::vector<dvec> extrapolationTable; // Extrapolation table
    dmatrix jac_;                      // Current Jacobian
    dvec ydot;                        // Temporary for function evaluations
    dvec work;                        // Work vector
    Eigen::PartialPivLU<Eigen::MatrixXd> lu; // LU decomposition
};