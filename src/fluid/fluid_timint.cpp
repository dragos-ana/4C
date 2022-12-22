/*-----------------------------------------------------------*/
/*! \file

\brief Base class for all fluid time integrations


\level 1

*/
/*-----------------------------------------------------------*/

#include "fluid_timint.H"

#include "discret.H"
#include "fluid_utils_mapextractor.H"
#include "globalproblem.H"
#include "inpar_parameterlist_utils.H"
#include "inpar_fluid.H"

#include "fluid_discretization_runtime_vtu_output_params.H"
#include "discretization_runtime_vtu_writer.H"
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

FLD::TimInt::TimInt(const Teuchos::RCP<DRT::Discretization>& discret,
    const Teuchos::RCP<LINALG::Solver>& solver, const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<IO::DiscretizationWriter>& output)
    : discret_(discret),
      solver_(solver),
      params_(params),
      output_(output),
      runtime_output_writer_(Teuchos::null),
      runtime_output_params_(),
      time_(0.0),
      step_(0),
      dta_(params_->get<double>("time step size")),
      stepmax_(params_->get<int>("max number timesteps")),
      maxtime_(params_->get<double>("total time")),
      itemax_(params_->get<int>("max nonlin iter steps")),
      uprestart_(params_->get("write restart every", -1)),
      upres_(params_->get("write solution every", -1)),
      timealgo_(DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(*params_, "time int algo")),
      physicaltype_(DRT::INPUT::get<INPAR::FLUID::PhysicalType>(*params_, "Physical Type")),
      myrank_(discret_->Comm().MyPID()),
      updateprojection_(false),
      projector_(Teuchos::null),
      kspsplitter_(Teuchos::null)
{
  // check for special fluid output which is to be handled by an own writer object
  Teuchos::RCP<const Teuchos::ParameterList> fluid_runtime_output_list =
      Teuchos::rcp(new Teuchos::ParameterList(
          DRT::Problem::Instance()->IOParams().sublist("RUNTIME VTK OUTPUT").sublist("FLUID")));

  bool output_fluid =
      (bool)DRT::INPUT::IntegralValue<int>(*fluid_runtime_output_list, "OUTPUT_FLUID");

  // create and initialize parameter container object for fluid specific runtime vtk output
  if (output_fluid)
  {
    runtime_output_params_.Init(*fluid_runtime_output_list);
    runtime_output_params_.Setup();
    runtime_output_writer_ = Teuchos::rcp(new DiscretizationRuntimeVtuWriter());
  }
}

FLD::TimInt::~TimInt() {}

Teuchos::RCP<const Epetra_Map> FLD::TimInt::DofRowMap(unsigned nds)
{
  return Teuchos::rcp(Discretization()->DofRowMap(nds), false);
}


void FLD::TimInt::IncrementTimeAndStep()
{
  step_ += 1;
  time_ += dta_;

  return;
}