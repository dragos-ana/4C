/*----------------------------------------------------------------------*/
/*! \file
\brief Creation of structural time integrators in accordance with user's wishes


\level 1
*/
/*----------------------------------------------------------------------*/
/* headers */
#include <ctime>
#include <cstdlib>
#include <iostream>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "strtimint_create.H"
#include "strtimint_statics.H"
#include "strtimint_prestress.H"
#include "strtimint_genalpha.H"
#include "strtimint_ost.H"
#include "strtimint_gemm.H"
#include "strtimint_expleuler.H"
#include "strtimint_centrdiff.H"
#include "strtimint_ab2.H"

#include "io.H"
#include "discret.H"
#include "globalproblem.H"
#include "validparameters.H"
#include "linalg_utils_sparse_algebra_math.H"

#include "prestress_service.H"

/*======================================================================*/
/* create marching time integrator */
Teuchos::RCP<STR::TimInt> STR::TimIntCreate(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioflags, const Teuchos::ParameterList& sdyn,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<DRT::Discretization>& actdis,
    Teuchos::RCP<LINALG::Solver>& solver, Teuchos::RCP<LINALG::Solver>& contactsolver,
    Teuchos::RCP<IO::DiscretizationWriter>& output)
{
  // set default output
  Teuchos::RCP<STR::TimInt> sti = Teuchos::null;
  // try implicit integrators
  sti = TimIntImplCreate(timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output);
  // if nothing found try explicit integrators
  if (sti == Teuchos::null)
  {
    sti =
        TimIntExplCreate(timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output);
  }

  // deliver
  return sti;
}

/*======================================================================*/
/* create implicit marching time integrator */
Teuchos::RCP<STR::TimIntImpl> STR::TimIntImplCreate(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioflags, const Teuchos::ParameterList& sdyn,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<DRT::Discretization>& actdis,
    Teuchos::RCP<LINALG::Solver>& solver, Teuchos::RCP<LINALG::Solver>& contactsolver,
    Teuchos::RCP<IO::DiscretizationWriter>& output)
{
  Teuchos::RCP<STR::TimIntImpl> sti = Teuchos::null;

  // TODO: add contact solver...

  // check if we have a problem that needs to be prestressed
  if (::UTILS::PRESTRESS::IsAny())
  {
    sti = Teuchos::rcp(new STR::TimIntPrestress(
        timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
    return sti;
  }

  // create specific time integrator
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    // Static analysis
    case INPAR::STR::dyna_statics:
    {
      sti = Teuchos::rcp(new STR::TimIntStatics(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }

    // Generalised-alpha time integration
    case INPAR::STR::dyna_genalpha:
    {
      sti = Teuchos::rcp(new STR::TimIntGenAlpha(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }

    // One-step-theta (OST) time integration
    case INPAR::STR::dyna_onesteptheta:
    {
      sti = Teuchos::rcp(new STR::TimIntOneStepTheta(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }

    // Generalised energy-momentum method (GEMM)
    case INPAR::STR::dyna_gemm:
    {
      sti = Teuchos::rcp(new STR::TimIntGEMM(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }

    // Everything else
    default:
    {
      // do nothing
      break;
    }
  }  // end of switch(sdyn->Typ)

  // return the integrator
  return sti;
}

/*======================================================================*/
/* create explicit marching time integrator */
Teuchos::RCP<STR::TimIntExpl> STR::TimIntExplCreate(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioflags, const Teuchos::ParameterList& sdyn,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<DRT::Discretization>& actdis,
    Teuchos::RCP<LINALG::Solver>& solver, Teuchos::RCP<LINALG::Solver>& contactsolver,
    Teuchos::RCP<IO::DiscretizationWriter>& output)
{
  Teuchos::RCP<STR::TimIntExpl> sti = Teuchos::null;

  // what's the current problem type?
  ProblemType probtype = DRT::Problem::Instance()->GetProblemType();

  if (probtype == ProblemType::fsi or probtype == ProblemType::fsi_redmodels or
      probtype == ProblemType::fsi_lung or probtype == ProblemType::gas_fsi or
      probtype == ProblemType::ac_fsi or probtype == ProblemType::biofilm_fsi or
      probtype == ProblemType::thermo_fsi)
  {
    dserror("no explicit time integration with fsi");
  }

  // create specific time integrator
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    // forward Euler time integration
    case INPAR::STR::dyna_expleuler:
    {
      sti = Teuchos::rcp(new STR::TimIntExplEuler(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }
    // central differences time integration
    case INPAR::STR::dyna_centrdiff:
    {
      sti = Teuchos::rcp(new STR::TimIntCentrDiff(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }
    // Adams-Bashforth 2nd order (AB2) time integration
    case INPAR::STR::dyna_ab2:
    {
      sti = Teuchos::rcp(new STR::TimIntAB2(
          timeparams, ioflags, sdyn, xparams, actdis, solver, contactsolver, output));
      break;
    }

    // Everything else
    default:
    {
      // do nothing
      break;
    }
  }  // end of switch(sdyn->Typ)

  // return the integrator
  return sti;
}

/*----------------------------------------------------------------------*/