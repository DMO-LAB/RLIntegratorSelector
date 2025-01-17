#include "splitSolver.h"
#include "readConfig.h"

void SplitSolver::setOptions(const ConfigOptions& options_)
{
    options = options_;
    useBalancedSplitting = (options.splittingMethod == "balanced");
    dt = options.globalTimestep;
}

void SplitSolver::resize(index_t nRows, index_t nCols)
{
    state.resize(nRows, nCols);
    startState.resize(nRows, nCols);
    deltaConv.resize(nRows, nCols);
    deltaDiff.resize(nRows, nCols);
    deltaProd.resize(nRows, nCols);
    ddtConv.resize(nRows, nCols);
    ddtDiff.resize(nRows, nCols);
    ddtProd.resize(nRows, nCols);
    ddtCross.resize(nRows, nCols);
    splitConstConv.setZero(nRows, nCols);
    splitConstDiff.setZero(nRows, nCols);
    splitConstProd.setZero(nRows, nCols);
}

void SplitSolver::calculateTimeDerivatives(double dt)
{
    ddtConv = deltaConv / dt - splitConstConv;
    ddtDiff = deltaDiff / dt - splitConstDiff;
    ddtProd = deltaProd / dt - splitConstProd;
}

int SplitSolver::step()
{
    setupStep();

    if (useBalancedSplitting) {
        splitConstDiff = 0.25 * (ddtProd + ddtConv + ddtCross - 3 * ddtDiff);
        splitConstConv = 0.25 * (ddtProd + ddtDiff + ddtCross - 3 * ddtConv);
        splitConstProd = 0.5 * (ddtConv + ddtDiff + ddtCross - ddtProd);

        std::string splitConstFile = (format("splitConst_t%.6f") % t).str();
    } else {
        splitConstDiff = ddtCross;
    }

    if (t == tStart && options.outputProfiles) {
        writeStateFile("", false, false);
    }

    if (options.debugIntegratorStages(tNow)) {
        writeStateFile((format("start_t%.6f") % t).str(), true, false);
    }

    prepareIntegrators();
    integrateSplitTerms(t, dt);
    calculateTimeDerivatives(dt);
    return finishStep();
}

void SplitSolver::integrateSplitTerms(double t, double dt)
{
    splitTimer.start();
    deltaConv.setZero();
    deltaProd.setZero();
    deltaDiff.setZero();
    splitTimer.stop();

    // std::cout << "Integrating first diffusion terms" << std::endl;
    _integrateDiffusionTerms(t, t + 0.25*dt, 1); // step 1/4
    // std::cout << "Integrating first convection terms" << std::endl;
    _integrateConvectionTerms(t, t + 0.5*dt, 1); // step 1/2
    // std::cout << "Integrating second diffusion terms" << std::endl;
    _integrateDiffusionTerms(t + 0.25*dt, t + 0.5*dt, 2); // step 2/4
    // std::cout << "Integrating production terms" << std::endl;
    _integrateProductionTerms(t, t + dt, 1); // full step
    // std::cout << "Integrating third diffusion terms" << std::endl;
    _integrateDiffusionTerms(t + 0.5*dt, t + 0.75*dt, 3); // step 3/4
    // std::cout << "Integrating second convection terms" << std::endl;
    _integrateConvectionTerms(t + 0.5*dt, t + dt, 2); // step 2/2
    // std::cout << "Integrating fourth diffusion terms" << std::endl;
    _integrateDiffusionTerms(t + 0.75*dt, t + dt, 4); // step 4/4
    
}

void SplitSolver::_integrateDiffusionTerms(double tStart, double tEnd, int stage)
{
    tStageStart = tStart;
    tStageEnd = tEnd;
    assert(mathUtils::notnan(state));
    logFile.verboseWrite(format("diffusion terms %i/4...") % stage, false);
    startState = state;
    integrateDiffusionTerms();
    assert(mathUtils::notnan(state));
    //print the temperature state
    // std::cout << "Diffusion Temp State: ";
    // for (int i = 20; i < 50; ++i) {
    //     std::cout << state.row(1)[i] << " ";
    // }
    // std::cout << std::endl;
    deltaDiff += state - startState;
    if (stage && options.debugIntegratorStages(tNow)) {
        writeStateFile((format("diff%i_t%.6f") % stage % tNow).str(), true, false);
    }
}

void SplitSolver::_integrateProductionTerms(double tStart, double tEnd, int stage)
{
    tStageStart = tStart;
    tStageEnd = tEnd;
    logFile.verboseWrite("Source term...", false);
    assert(mathUtils::notnan(state));
    startState = state;
    integrateProductionTerms();
    assert(mathUtils::notnan(state));
    deltaProd += state - startState;
    if (options.debugIntegratorStages(tNow)) {
        writeStateFile((format("prod_t%.6f") % tNow).str(), true, false);
    }
    // std::cout << "Production State Prod: " << deltaProd << std::endl;
}

void SplitSolver::_integrateConvectionTerms(double tStart, double tEnd, int stage)
{
    tStageStart = tStart;
    tStageEnd = tEnd;
    assert(stage == 1 || stage == 2);
    assert(mathUtils::notnan(state));
    logFile.verboseWrite(format("convection term %i/2...") % stage, false);
    startState = state;
    integrateConvectionTerms();
    assert(mathUtils::notnan(state));
    // std::cout << "Convection Temp State: ";
    // for (int i = 20; i < 50; ++i) {
    //     std::cout << state.row(1)[i] << " ";
    // }
    // std::cout << std::endl;
    deltaConv += state - startState;
    if (options.debugIntegratorStages(tNow)) {
        writeStateFile((format("conv%i_t%.6f") % stage % tNow).str(), true, false);
    }
    // std::cout << "Convection State Conv: " << deltaConv << std::endl;
}
