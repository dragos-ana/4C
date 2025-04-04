// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_poro_wrapper.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_global_data.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/// constructor
Adapter::StructurePoroWrapper::StructurePoroWrapper(
    std::shared_ptr<Field> field, FieldWrapper::Fieldtype type, bool NOXCorrection)
    : FieldWrapper(field, type, NOXCorrection)
{
  switch (type_)
  {
    case FieldWrapper::type_StructureField:
      structure_ = std::dynamic_pointer_cast<FSIStructureWrapper>(field_);
      if (structure_ == nullptr)
        FOUR_C_THROW("StructurePoroWrapper: Cast from Field to FSIStructureWrapper failed!");
      poro_ = nullptr;
      break;
    case FieldWrapper::type_PoroField:
      poro_ = std::dynamic_pointer_cast<PoroElast::Monolithic>(field_);
      if (poro_ == nullptr)
        FOUR_C_THROW("StructurePoroWrapper: Cast from Field to PoroBase failed!");
      structure_ = poro_->structure_field();
      break;
    default:
      FOUR_C_THROW(
          "StructurePoroWrapper - FieldWrapper::Fieldtype not available for this wrapper!");
      break;
  }
}

/// setup
void Adapter::StructurePoroWrapper::setup()
{
  structure_->setup();
  if (type_ == FieldWrapper::type_PoroField)
  {
    poro_->setup_system();
    poro_->setup_newton();  // just to avoid modifications in poro (this sets iterinc_ there)
  }
}

//! unique map of all dofs that should be constrained with DBC
std::shared_ptr<const Core::LinAlg::Map> Adapter::StructurePoroWrapper::combined_dbc_map()
{
  switch (type_)
  {
    case FieldWrapper::type_StructureField:
      return structure_->get_dbc_map_extractor()->cond_map();
      break;
    case FieldWrapper::type_PoroField:
      return poro_->combined_dbc_map();
      break;
    default:
      FOUR_C_THROW("StructurePoroWrapper: type for this wrapper not considered!");
      return nullptr;
      break;
  }
}

//   //! perform result test
void Adapter::StructurePoroWrapper::test_results(Global::Problem* problem)
{
  problem->add_field_test(structure_->create_field_test());

  if (type_ == FieldWrapper::type_PoroField)
    problem->add_field_test(poro_->fluid_field()->create_field_test());
}

const std::shared_ptr<PoroElast::Monolithic>& Adapter::StructurePoroWrapper::poro_field()
{
  if (type_ == Adapter::FieldWrapper::type_PoroField)
    return poro_;
  else
    FOUR_C_THROW("StructurePoroWrapper - Field not a poro_field!");
  return poro_;  // do not remove FOUR_C_THROW!!! - return just to make compiler happy :-)
}

const std::shared_ptr<Adapter::FSIStructureWrapper>&
Adapter::StructurePoroWrapper::structure_field()
{
  if (type_ == FieldWrapper::type_PoroField || type_ == FieldWrapper::type_StructureField)
    return structure_;
  else
    FOUR_C_THROW("StructurePoroWrapper - Field not Structural- or Poro-Field!");
  return structure_;  // do not remove FOUR_C_THROW!!! - return just to make compiler happy :-)
}

//! return poro fluid_field
const std::shared_ptr<Adapter::FluidPoro>& Adapter::StructurePoroWrapper::fluid_field()
{
  if (type_ == FieldWrapper::type_PoroField)
    return poro_->fluid_field();
  else
    FOUR_C_THROW("StructurePoroWrapper - Field not poro_field (no poro fluid field!");
  return poro_
      ->fluid_field();  // do not remove FOUR_C_THROW!!! - return just to make compiler happy :-)
}

//! Insert FSI Condition Vector
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::StructurePoroWrapper::insert_fsi_cond_vector(
    const Core::LinAlg::Vector<double>& cond)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpcond;
  switch (type_)
  {
    case FieldWrapper::type_StructureField:
      return interface()->insert_fsi_cond_vector(cond);
      break;
    case FieldWrapper::type_PoroField:
      tmpcond = interface()->insert_fsi_cond_vector(cond);
      return poro_->extractor()->insert_vector(*tmpcond, 0);  // into structural part = 0
      break;
    default:
      FOUR_C_THROW("StructurePoroWrapper: type for this wrapper not considered!");
      return nullptr;
      break;
  }
}

//! Recover Lagrange Multiplier during iteration (does nothing for structure)
void Adapter::StructurePoroWrapper::recover_lagrange_multiplier_after_newton_step(
    std::shared_ptr<Core::LinAlg::Vector<double>> iterinc)
{
  if (type_ == FieldWrapper::type_PoroField)
    poro_->recover_lagrange_multiplier_after_newton_step(iterinc);
}

FOUR_C_NAMESPACE_CLOSE
