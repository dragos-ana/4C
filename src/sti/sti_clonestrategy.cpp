/*----------------------------------------------------------------------*/
/*! \file

\brief strategy for cloning thermo discretization from scatra discretization

\level 2

*/
/*----------------------------------------------------------------------*/
#include "sti_clonestrategy.H"

#include "globalproblem.H"

#include "matpar_bundle.H"

#include "scatra_ele.H"

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::ScatraThermoCloneStrategy::CheckMaterialType(const int matid)
{
  // check whether material with specified ID is compatible with cloned element or not
  switch (DRT::Problem::Instance()->Materials()->ById(matid)->Type())
  {
    case INPAR::MAT::m_soret:
    case INPAR::MAT::m_th_fourier_iso:
      // do nothing in case of compatible material
      break;

    default:
    {
      // throw error in case of incompatible material
      dserror("Material with ID %d is not compatible with cloned transport element!", matid);
      break;
    }
  }
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
std::map<std::string, std::string> STI::ScatraThermoCloneStrategy::ConditionsToCopy()
{
  // initialize map
  std::map<std::string, std::string> conditions;

  // insert thermo conditions
  conditions.insert(std::pair<std::string, std::string>("PointThermoCoupling", "PointCoupling"));
  conditions.insert(std::pair<std::string, std::string>("S2IKinetics", "S2IKinetics"));
  conditions.insert(std::pair<std::string, std::string>("S2IMeshtying", "S2IMeshtying"));
  conditions.insert(std::pair<std::string, std::string>("ScaTraFluxCalc", "ScaTraFluxCalc"));
  conditions.insert(std::pair<std::string, std::string>("ThermoDirichlet", "Dirichlet"));
  conditions.insert(std::pair<std::string, std::string>("ThermoPointNeumann", "PointNeumann"));
  conditions.insert(std::pair<std::string, std::string>("ThermoLineNeumann", "LineNeumann"));
  conditions.insert(std::pair<std::string, std::string>("ThermoSurfaceNeumann", "SurfaceNeumann"));
  conditions.insert(std::pair<std::string, std::string>("ThermoVolumeNeumann", "VolumeNeumann"));
  conditions.insert(std::pair<std::string, std::string>("ThermoInitfield", "Initfield"));
  conditions.insert(std::pair<std::string, std::string>("ThermoRobin", "TransportRobin"));
  conditions.insert(
      std::pair<std::string, std::string>("ScatraPartitioning", "ScatraPartitioning"));

  // return map
  return conditions;
}  // STI::ScatraThermoCloneStrategy::ConditionsToCopy()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
bool STI::ScatraThermoCloneStrategy::DetermineEleType(
    DRT::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // set type of cloned element to transport type
  eletype.emplace_back("TRANSP");

  // element should always be cloned
  return true;
}  // STI::ScatraThermoCloneStrategy::DetermineEleType

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::ScatraThermoCloneStrategy::SetElementData(
    Teuchos::RCP<DRT::Element> newele, DRT::Element* oldele, const int matid, const bool isnurbs)
{
  // cast pointers to current element on source discretization and to current cloned element on
  // target discretization
  auto* oldele_transport = dynamic_cast<DRT::ELEMENTS::Transport*>(oldele);
  Teuchos::RCP<DRT::ELEMENTS::Transport> newele_transport =
      Teuchos::rcp_dynamic_cast<DRT::ELEMENTS::Transport>(newele);

  // safety check
  if (oldele_transport == nullptr or newele_transport == Teuchos::null)
    dserror(
        "Expected transport element, but received element of type '%s'!", typeid(*newele).name());

  // provide cloned element with material
  newele_transport->SetMaterial(matid, oldele);

  // provide cloned element with discretization type
  newele_transport->SetDisType(oldele->Shape());

  // provide cloned element with physical implementation type
  switch (oldele_transport->ImplType())
  {
    case INPAR::SCATRA::impltype_elch_diffcond_thermo:
    case INPAR::SCATRA::impltype_elch_diffcond:
    {
      newele_transport->SetImplType(INPAR::SCATRA::impltype_thermo_elch_diffcond);
      break;
    }
    case INPAR::SCATRA::impltype_elch_electrode_thermo:
    case INPAR::SCATRA::impltype_elch_electrode:
    {
      newele_transport->SetImplType(INPAR::SCATRA::impltype_thermo_elch_electrode);
      break;
    }
    default:
    {
      dserror(
          "Scatra-thermo interaction not yet implemented for given element implementation type!");
      break;
    }
  }
}