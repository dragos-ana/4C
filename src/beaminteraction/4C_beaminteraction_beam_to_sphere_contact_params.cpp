// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_sphere_contact_params.hpp"

#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamInteraction::BeamToSphereContactParams::BeamToSphereContactParams()
    : isinit_(false), issetup_(false), penalty_parameter_(-1.0)
{
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

void BeamInteraction::BeamToSphereContactParams::init()
{
  issetup_ = false;

  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_sphere_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO SPHERE CONTACT");

  penalty_parameter_ = beam_to_sphere_contact_params_list.get<double>("PENALTY_PARAMETER");

  if (penalty_parameter_ < 0.0)
    FOUR_C_THROW("beam-to-sphere penalty parameter must not be negative!");


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamToSphereContactParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
