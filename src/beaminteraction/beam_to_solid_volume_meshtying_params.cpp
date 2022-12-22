/*----------------------------------------------------------------------*/
/*! \file

\brief Data container holding all beam to solid volume meshtying input parameters.

\level 3
*/


#include "beam_to_solid_volume_meshtying_params.H"

#include "beam_to_solid_volume_meshtying_vtk_output_params.H"

#include "inpar_geometry_pair.H"
#include "globalproblem.H"


/**
 *
 */
BEAMINTERACTION::BeamToSolidVolumeMeshtyingParams::BeamToSolidVolumeMeshtyingParams()
    : BeamToSolidParamsBase(),
      integration_points_circumference_(0),
      rotation_coupling_(INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::none),
      rotational_coupling_penalty_parameter_(0.0),
      output_params_ptr_(Teuchos::null),
      couple_restart_state_(false)
{
  // Empty Constructor.
}


/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingParams::Init()
{
  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_solid_contact_params_list =
      DRT::Problem::Instance()->BeamInteractionParams().sublist("BEAM TO SOLID VOLUME MESHTYING");

  // Set the common beam-to-solid parameters.
  SetBaseParams(beam_to_solid_contact_params_list);

  // Get parameters form input file.
  {
    // Number of integrations points along the circumference of the cross section.
    integration_points_circumference_ =
        beam_to_solid_contact_params_list.get<int>("INTEGRATION_POINTS_CIRCUMFERENCE");

    // Type of rotational coupling.
    rotation_coupling_ = Teuchos::getIntegralValue<INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling>(
        beam_to_solid_contact_params_list, "ROTATION_COUPLING");

    // Mortar contact discretization to be used.
    mortar_shape_function_rotation_ =
        Teuchos::getIntegralValue<INPAR::BEAMTOSOLID::BeamToSolidMortarShapefunctions>(
            beam_to_solid_contact_params_list, "ROTATION_COUPLING_MORTAR_SHAPE_FUNCTION");
    if (GetContactDiscretization() ==
            INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization::mortar and
        rotation_coupling_ != INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::none and
        mortar_shape_function_rotation_ ==
            INPAR::BEAMTOSOLID::BeamToSolidMortarShapefunctions::none)
      dserror(
          "If mortar coupling and rotational coupling are activated, the shape function type for "
          "rotational coupling has to be explicitly given.");

    // Penalty parameter for rotational coupling.
    rotational_coupling_penalty_parameter_ =
        beam_to_solid_contact_params_list.get<double>("ROTATION_COUPLING_PENALTY_PARAMETER");

    // If the restart configuration should be coupled.
    couple_restart_state_ = (bool)DRT::INPUT::IntegralValue<int>(
        beam_to_solid_contact_params_list, "COUPLE_RESTART_STATE");
  }

  // Setup the output parameter object.
  {
    output_params_ptr_ = Teuchos::rcp<BeamToSolidVolumeMeshtyingVtkOutputParams>(
        new BeamToSolidVolumeMeshtyingVtkOutputParams());
    output_params_ptr_->Init();
    output_params_ptr_->Setup();
  }

  // Sanity checks.
  if (rotation_coupling_ != INPAR::BEAMTOSOLID::BeamToSolidRotationCoupling::none and
      couple_restart_state_)
    dserror("Coupling restart state combined with rotational coupling is not yet implemented!");

  isinit_ = true;
}

/**
 *
 */
Teuchos::RCP<BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputParams>
BEAMINTERACTION::BeamToSolidVolumeMeshtyingParams::GetVtkOuputParamsPtr()
{
  return output_params_ptr_;
};