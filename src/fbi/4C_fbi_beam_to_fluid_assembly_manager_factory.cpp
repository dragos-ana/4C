/*----------------------------------------------------------------------*/
/*! \file

\brief Factory to create appropriate beam to fluid meshtying assembly managers for the desired
constraint discretization approach

\level 2
*/

#include "4C_fbi_beam_to_fluid_assembly_manager_factory.hpp"

#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_fluid_assembly_strategy.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_direct.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_indirect.hpp"
#include "4C_inpar_fbi.hpp"
#include "4C_lib_discret.hpp"

FOUR_C_NAMESPACE_OPEN
/**
 *
 */
Teuchos::RCP<BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManager>
BEAMINTERACTION::BeamToFluidAssemblyManagerFactory::CreateAssemblyManager(
    Teuchos::RCP<const DRT::Discretization> discretization1,
    Teuchos::RCP<const DRT::Discretization> discretization2,
    std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>> interaction_pairs,
    const Teuchos::RCP<FBI::BeamToFluidMeshtyingParams> params_ptr,
    Teuchos::RCP<FBI::UTILS::FBIAssemblyStrategy> assemblystrategy)
{
  // Get the meshtying discretization method.
  INPAR::FBI::BeamToFluidDiscretization meshtying_discretization =
      params_ptr->GetContactDiscretization();

  switch (meshtying_discretization)
  {
    case INPAR::FBI::BeamToFluidDiscretization::mortar:
      return Teuchos::rcp(
          new BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerIndirect(
              interaction_pairs, discretization1, discretization2, params_ptr));
      break;
    case INPAR::FBI::BeamToFluidDiscretization::gauss_point_to_segment:
      return Teuchos::rcp(
          new BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerDirect(
              interaction_pairs, assemblystrategy));
      break;
    default:
      FOUR_C_THROW("Beam To Fluid Meshtying Discretization Type not supported!");
      return Teuchos::null;
  }
}

FOUR_C_NAMESPACE_CLOSE