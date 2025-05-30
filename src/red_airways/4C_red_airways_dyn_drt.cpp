// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_dyn_drt.hpp"

#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_red_airways_implicitintegration.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_red_airways_resulttest.hpp"
#include "4C_utils_result_test.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * Main control routine for reduced dimensional airway network including|
 * various solvers                                                      |
 *                                                                      ||
 *----------------------------------------------------------------------*/
void dyn_red_airways_drt() { dyn_red_airways_drt(false); }

std::shared_ptr<Airway::RedAirwayImplicitTimeInt> dyn_red_airways_drt(bool CoupledTo3D)
{
  if (Global::Problem::instance()->does_exist_dis("red_airway") == false)
  {
    return nullptr;
  }

  // 1. Access the discretization
  std::shared_ptr<Core::FE::Discretization> actdis = nullptr;
  actdis = Global::Problem::instance()->get_dis("red_airway");

  // Set degrees of freedom in the discretization
  if (!actdis->filled())
  {
    actdis->fill_complete();
  }

  // If discretization is empty, then return empty time integration
  if (actdis->num_global_elements() < 1)
  {
    return nullptr;
  }

  // 2. Context for output and restart
  std::shared_ptr<Core::IO::DiscretizationWriter> output = actdis->writer();
  output->write_mesh(0, 0.0);

  // 3. Set pointers and variables for ParameterList rawdyn
  const Teuchos::ParameterList& rawdyn =
      Global::Problem::instance()->reduced_d_airway_dynamic_params();

  // 4. Create a linear solver
  // Get the solver number for the LINEAR_SOLVER
  const int linsolvernumber = rawdyn.get<int>("LINEAR_SOLVER");
  // Check if the present solver has a valid solver number
  if (linsolvernumber == (-1))
  {
    FOUR_C_THROW(
        "no linear solver defined. Please set LINEAR_SOLVER in REDUCED DIMENSIONAL AIRWAYS DYNAMIC "
        "to a valid number!");
  }
  // Create the solver
  std::unique_ptr<Core::LinAlg::Solver> solver = std::make_unique<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(linsolvernumber), actdis->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  actdis->compute_null_space_if_necessary(solver->params());

  // 5. Set parameters in list required for all schemes
  Teuchos::ParameterList airwaystimeparams;

  // Number of degrees of freedom
  const int ndim = Global::Problem::instance()->n_dim();
  airwaystimeparams.set<int>("number of degrees of freedom", 1 * ndim);

  // Time step size
  airwaystimeparams.set<double>("time step size", rawdyn.get<double>("TIMESTEP"));
  // Maximum number of timesteps
  airwaystimeparams.set<int>("max number timesteps", rawdyn.get<int>("NUMSTEP"));

  // Restart
  airwaystimeparams.set("write restart every", rawdyn.get<int>("RESTARTEVERY"));
  // Solution output
  airwaystimeparams.set("write solution every", rawdyn.get<int>("RESULTSEVERY"));

  // Solver type
  airwaystimeparams.set("solver type", rawdyn.get<Airway::RedAirwaysSolvertype>("SOLVERTYPE"));
  // Tolerance
  airwaystimeparams.set("tolerance", rawdyn.get<double>("TOLERANCE"));
  // Maximum number of iterations
  airwaystimeparams.set("maximum iteration steps", rawdyn.get<int>("MAXITERATIONS"));

  if (rawdyn.get<bool>("COMPAWACINTER"))
    airwaystimeparams.set("CompAwAcInter", true);
  else
    airwaystimeparams.set("CompAwAcInter", false);

  // Adjust acini volume with pre-stress condition
  if (rawdyn.get<bool>("CALCV0PRESTRESS"))
  {
    airwaystimeparams.set("CalcV0PreStress", true);
    airwaystimeparams.set("transpulmpress", rawdyn.get<double>("TRANSPULMPRESS"));
  }
  else
    airwaystimeparams.set("CalcV0PreStress", false);


  // 6. Create all vectors and variables associated with the time
  // integration (call the constructor);
  // the only parameter from the list required here is the number of
  // velocity degrees of freedom
  std::shared_ptr<Airway::RedAirwayImplicitTimeInt> airwayimplicit =
      std::make_shared<Airway::RedAirwayImplicitTimeInt>(
          actdis, std::move(solver), airwaystimeparams, *output);

  // Initialize state save vectors
  if (CoupledTo3D)
  {
    airwayimplicit->init_save_state();
  }

  // Initial field from restart or calculated by given function
  const int restart = Global::Problem::instance()->restart();
  if (restart && !CoupledTo3D)
  {
    // Read the restart information, set vectors and variables
    airwayimplicit->read_restart(restart);
  }

  if (!CoupledTo3D)
  {
    // Call time-integration scheme for 0D problem
    std::shared_ptr<Teuchos::ParameterList> param_temp;
    airwayimplicit->integrate();

    // Create resulttest
    std::shared_ptr<Core::Utils::ResultTest> resulttest =
        std::make_shared<Airway::RedAirwayResultTest>(*airwayimplicit);

    // Resulttest for 0D problem and testing
    Global::Problem::instance()->add_field_test(resulttest);
    Global::Problem::instance()->test_all(actdis->get_comm());

    return airwayimplicit;
  }
  else
  {
    return airwayimplicit;
  }
}

FOUR_C_NAMESPACE_CLOSE
