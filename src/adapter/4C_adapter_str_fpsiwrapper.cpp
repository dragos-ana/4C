// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_fpsiwrapper.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  bool prestress_is_active(const double currentTime)
  {
    Inpar::Solid::PreStress pstype = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
        Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
    const double pstime =
        Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");
    return pstype != Inpar::Solid::PreStress::none && currentTime <= pstime + 1.0e-15;
  }
}  // namespace


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::FPSIStructureWrapper::FPSIStructureWrapper(std::shared_ptr<Structure> structure)
    : FSIStructureWrapper(structure)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FPSIStructureWrapper::extract_interface_dispn(bool FPSI)
{
  if (!FPSI)
  {
    return Adapter::FSIStructureWrapper::extract_interface_dispn();
  }
  else
  {
    // prestressing business
    if (prestress_is_active(time_old()))
    {
      return std::make_shared<Core::LinAlg::Vector<double>>(*interface_->fpsi_cond_map(), true);
    }
    else
    {
      return interface_->extract_fpsi_cond_vector(*dispn());
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>>
Adapter::FPSIStructureWrapper::extract_interface_dispnp(bool FPSI)
{
  if (!FPSI)
  {
    return Adapter::FSIStructureWrapper::extract_interface_dispnp();
  }
  else
  {
    // prestressing business
    if (prestress_is_active(time()))
    {
      return std::make_shared<Core::LinAlg::Vector<double>>(*interface_->fpsi_cond_map(), true);
    }
    else
    {
      return interface_->extract_fpsi_cond_vector(*dispnp());
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
