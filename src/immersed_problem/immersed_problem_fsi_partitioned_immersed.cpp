/*----------------------------------------------------------------------*/
/*! \file

\brief partitioned immersed fsi subclass

\level 1


*----------------------------------------------------------------------*/
#include "immersed_problem_fsi_partitioned_immersed.H"

#include "fsi_debugwriter.H"
#include "lib_globalproblem.H"
#include "adapter_str_fsiwrapper.H"


FSI::PartitionedImmersed::PartitionedImmersed(const Epetra_Comm& comm) : Partitioned(comm)
{
  // empty constructor
}


void FSI::PartitionedImmersed::Setup()
{
  // call setup of base class
  FSI::Partitioned::Setup();
}


void FSI::PartitionedImmersed::SetupCoupling(
    const Teuchos::ParameterList& fsidyn, const Epetra_Comm& comm)
{
  if (Comm().MyPID() == 0)
    std::cout << "\n SetupCoupling in FSI::PartitionedImmersed ..." << std::endl;

  // for immersed fsi
  coupsfm_ = Teuchos::null;
  matchingnodes_ = false;

  // enable debugging
  if (DRT::INPUT::IntegralValue<int>(fsidyn, "DEBUGOUTPUT"))
    debugwriter_ = Teuchos::rcp(new UTILS::DebugWriter(StructureField()->Discretization()));
}


void FSI::PartitionedImmersed::ExtractPreviousInterfaceSolution()
{
  // not necessary in immersed fsi.
  // overrides version in fsi_paritioned with "do nothing".
}