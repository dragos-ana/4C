/*! \file

\brief Implementation of routines for calculation of solid element with templated element
formulation

\level 1
*/

#include "4C_solid_3D_ele_calc.hpp"

#include "4C_discretization_fem_general_cell_type.hpp"
#include "4C_discretization_fem_general_cell_type_traits.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele_calc_displacement_based.hpp"
#include "4C_solid_3D_ele_calc_displacement_based_linear_kinematics.hpp"
#include "4C_solid_3D_ele_calc_fbar.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"
#include "4C_solid_3D_ele_calc_mulf.hpp"
#include "4C_solid_3D_ele_calc_mulf_fbar.hpp"
#include "4C_solid_3D_ele_formulation.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{

  template <CORE::FE::CellType celltype, typename SolidFormulation>
  class ElementFormulationDerivativeEvaluator
  {
   public:
    ElementFormulationDerivativeEvaluator(const DRT::Element& ele,
        const DRT::ELEMENTS::ElementNodes<celltype>& element_nodes,
        const CORE::LINALG::Matrix<3, 1>& xi,
        const DRT::ELEMENTS::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
        const DRT::ELEMENTS::JacobianMapping<celltype>& jacobian_mapping,
        const CORE::LINALG::Matrix<3, 3>& deformation_gradient,
        const DRT::ELEMENTS::PreparationData<SolidFormulation>& preparation_data,
        DRT::ELEMENTS::SolidFormulationHistory<SolidFormulation>& history_data)
        : ele_(ele),
          element_nodes_(element_nodes),
          xi_(xi),
          shape_functions_(shape_functions),
          jacobian_mapping_(jacobian_mapping),
          deformation_gradient_(deformation_gradient),
          preparation_data_(preparation_data),
          history_data_(history_data)
    {
    }

    [[nodiscard]] CORE::LINALG::Matrix<9, CORE::FE::num_nodes<celltype> * CORE::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_displacements() const
    {
      return EvaluateDDeformationGradientDDisplacements(ele_, element_nodes_, xi_, shape_functions_,
          jacobian_mapping_, deformation_gradient_, preparation_data_, history_data_);
    }

    [[nodiscard]] CORE::LINALG::Matrix<9, CORE::FE::dim<celltype>>
    evaluate_d_deformation_gradient_d_xi() const
    {
      return EvaluateDDeformationGradientDXi(ele_, element_nodes_, xi_, shape_functions_,
          jacobian_mapping_, deformation_gradient_, preparation_data_, history_data_);
    }

    [[nodiscard]] CORE::LINALG::Matrix<9,
        CORE::FE::num_nodes<celltype> * CORE::FE::dim<celltype> * CORE::FE::dim<celltype>>
    evaluate_d2_deformation_gradient_d_displacements_d_xi() const
    {
      return EvaluateDDeformationGradientDDisplacementsDXi(ele_, element_nodes_, xi_,
          shape_functions_, jacobian_mapping_, deformation_gradient_, preparation_data_,
          history_data_);
    }

   private:
    const DRT::Element& ele_;
    const DRT::ELEMENTS::ElementNodes<celltype>& element_nodes_;
    const CORE::LINALG::Matrix<3, 1>& xi_;
    const DRT::ELEMENTS::ShapeFunctionsAndDerivatives<celltype>& shape_functions_;
    const DRT::ELEMENTS::JacobianMapping<celltype>& jacobian_mapping_;
    const CORE::LINALG::Matrix<3, 3>& deformation_gradient_;
    const DRT::ELEMENTS::PreparationData<SolidFormulation>& preparation_data_;
    DRT::ELEMENTS::SolidFormulationHistory<SolidFormulation>& history_data_;
  };

  template <typename T>
  T* get_ptr(std::optional<T>& opt)
  {
    return opt.has_value() ? &opt.value() : nullptr;
  }

  template <typename T, typename... Args>
  std::optional<T> make_optional_if(bool condition, Args&&... params)
  {
    if (condition) return std::optional<T>(std::forward<Args>(params)...);
    return std::nullopt;
  }

  template <CORE::FE::CellType celltype, typename SolidFormulation>
  double EvaluateCauchyNDirAtXi(MAT::So3Material& mat,
      const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
          deformation_gradient,
      const CORE::LINALG::Matrix<3, 1>& n, const CORE::LINALG::Matrix<3, 1>& dir, int eleGID,
      const ElementFormulationDerivativeEvaluator<celltype, SolidFormulation>& evaluator,
      DRT::ELEMENTS::CauchyNDirLinearizations<3>& linearizations)
  {
    auto d_cauchyndir_dF = make_optional_if<CORE::LINALG::Matrix<9, 1>>(
        linearizations.d_cauchyndir_dd || linearizations.d_cauchyndir_dxi ||
            linearizations.d2_cauchyndir_dd_dxi,
        true);

    auto d2_cauchyndir_dF2 =
        make_optional_if<CORE::LINALG::Matrix<9, 9>>(linearizations.d2_cauchyndir_dd2, true);

    auto d2_cauchyndir_dF_dn =
        make_optional_if<CORE::LINALG::Matrix<9, 3>>(linearizations.d2_cauchyndir_dd_dn, true);

    auto d2_cauchyndir_dF_ddir =
        make_optional_if<CORE::LINALG::Matrix<9, 3>>(linearizations.d2_cauchyndir_dd_ddir, true);

    std::optional<CORE::LINALG::Matrix<9, CORE::FE::num_nodes<celltype> * CORE::FE::dim<celltype>>>
        d_F_dd{};
    if (linearizations.d_cauchyndir_dd != nullptr ||
        linearizations.d2_cauchyndir_dd_dn != nullptr ||
        linearizations.d2_cauchyndir_dd_ddir != nullptr ||
        linearizations.d2_cauchyndir_dd2 != nullptr)
    {
      d_F_dd.emplace(evaluator.evaluate_d_deformation_gradient_d_displacements());
    }

    std::optional<CORE::LINALG::Matrix<9, CORE::FE::dim<celltype>>> d_F_dxi{};
    if (linearizations.d_cauchyndir_dxi != nullptr)
    {
      d_F_dxi.emplace(evaluator.evaluate_d_deformation_gradient_d_xi());
    }

    std::optional<CORE::LINALG::Matrix<9,
        CORE::FE::num_nodes<celltype> * CORE::FE::dim<celltype> * CORE::FE::dim<celltype>>>
        d2_F_dxi_dd{};
    if (linearizations.d2_cauchyndir_dd_dxi != nullptr)
    {
      d2_F_dxi_dd.emplace(evaluator.evaluate_d2_deformation_gradient_d_displacements_d_xi());
    }

    double cauchy_n_dir = 0;
    mat.EvaluateCauchyNDirAndDerivatives(deformation_gradient, n, dir, cauchy_n_dir,
        linearizations.d_cauchyndir_dn, linearizations.d_cauchyndir_ddir, get_ptr(d_cauchyndir_dF),
        get_ptr(d2_cauchyndir_dF2), get_ptr(d2_cauchyndir_dF_dn), get_ptr(d2_cauchyndir_dF_ddir),
        -1, eleGID, nullptr, nullptr, nullptr, nullptr);

    // evaluate first derivative w.r.t. displacements
    if (linearizations.d_cauchyndir_dd != nullptr)
    {
      FOUR_C_ASSERT(d_F_dd && d_cauchyndir_dF, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateDCauchyNDirDDisplacements<celltype>(
          *d_F_dd, *d_cauchyndir_dF, *linearizations.d_cauchyndir_dd);
    }


    // evaluate second derivative w.r.t. displacements, normal
    if (linearizations.d2_cauchyndir_dd_dn != nullptr)
    {
      FOUR_C_ASSERT(d_F_dd && d2_cauchyndir_dF_dn, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateD2CauchyNDirDDisplacementsDNormal<celltype>(
          *d_F_dd, *d2_cauchyndir_dF_dn, *linearizations.d2_cauchyndir_dd_dn);
    }

    // evaluate second derivative w.r.t. displacements, direction
    if (linearizations.d2_cauchyndir_dd_ddir != nullptr)
    {
      FOUR_C_ASSERT(d_F_dd && d2_cauchyndir_dF_ddir, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateD2CauchyNDirDDisplacementsDDir<celltype>(
          *d_F_dd, *d2_cauchyndir_dF_ddir, *linearizations.d2_cauchyndir_dd_ddir);
    }

    // evaluate second derivative w.r.t. displacements, displacements
    if (linearizations.d2_cauchyndir_dd2 != nullptr)
    {
      FOUR_C_ASSERT(d_F_dd && d2_cauchyndir_dF2, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateD2CauchyNDirDDisplacements2<celltype>(
          *d_F_dd, *d2_cauchyndir_dF2, *linearizations.d2_cauchyndir_dd2);
    }

    // evaluate first derivative w.r.t. xi
    if (linearizations.d_cauchyndir_dxi != nullptr)
    {
      FOUR_C_ASSERT(d_F_dxi && d_cauchyndir_dF, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateDCauchyNDirDXi<celltype>(
          *d_F_dxi, *d_cauchyndir_dF, *linearizations.d_cauchyndir_dxi);
    }

    // evaluate second derivative w.r.t. displacements, xi
    if (linearizations.d2_cauchyndir_dd_dxi != nullptr)
    {
      FOUR_C_ASSERT(d2_F_dxi_dd && d_cauchyndir_dF, "Not all tensors are computed!");
      DRT::ELEMENTS::EvaluateD2CauchyNDirDDisplacementsDXi<celltype>(
          *d2_F_dxi_dd, *d_cauchyndir_dF, *linearizations.d2_cauchyndir_dd_dxi);
    }

    return cauchy_n_dir;
  }
}  // namespace

template <CORE::FE::CellType celltype, typename ElementFormulation>
DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::SolidEleCalc()
    : stiffness_matrix_integration_(
          CreateGaussIntegration<celltype>(GetGaussRuleStiffnessMatrix<celltype>())),
      mass_matrix_integration_(CreateGaussIntegration<celltype>(GetGaussRuleMassMatrix<celltype>()))
{
  DRT::ELEMENTS::ResizeGPHistory(history_data_, stiffness_matrix_integration_.NumPoints());
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::Pack(
    CORE::COMM::PackBuffer& data) const
{
  DRT::ELEMENTS::Pack<ElementFormulation>(data, history_data_);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::Unpack(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  DRT::ELEMENTS::Unpack<ElementFormulation>(position, data, history_data_);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::EvaluateNonlinearForceStiffnessMass(
    const DRT::Element& ele, MAT::So3Material& solid_material,
    const DRT::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, CORE::LINALG::SerialDenseVector* force_vector,
    CORE::LINALG::SerialDenseMatrix* stiffness_matrix, CORE::LINALG::SerialDenseMatrix* mass_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<CORE::LINALG::Matrix<num_dof_per_ele_, num_dof_per_ele_>> stiff{};
  std::optional<CORE::LINALG::Matrix<num_dof_per_ele_, num_dof_per_ele_>> mass{};
  std::optional<CORE::LINALG::Matrix<num_dof_per_ele_, 1>> force{};
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);
  if (mass_matrix != nullptr) mass.emplace(*mass_matrix, true);
  if (force_vector != nullptr) force.emplace(*force_vector, true);

  const ElementNodes<celltype> nodal_coordinates =
      EvaluateElementNodes<celltype>(ele, discretization, lm);

  bool equal_integration_mass_stiffness =
      CompareGaussIntegration(mass_matrix_integration_, stiffness_matrix_integration_);


  EvaluateCentroidCoordinatesAndAddToParameterList(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      Prepare(ele, nodal_coordinates, history_data_);

  double element_mass = 0.0;
  double element_volume = 0.0;
  ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const CORE::LINALG::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        EvaluateGPCoordinatesAndAddToParameterList(nodal_coordinates, shape_functions, params);
        Evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                    deformation_gradient,
                const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
            {
              const Stress<celltype> stress = EvaluateMaterialStress<celltype>(
                  solid_material, deformation_gradient, gl_strain, params, gp, ele.Id());

              if (force.has_value())
              {
                DRT::ELEMENTS::AddInternalForceVector<ElementFormulation, celltype>(linearization,
                    stress, integration_factor, preparation_data, history_data_, gp, *force);
              }

              if (stiff.has_value())
              {
                AddStiffnessMatrix<ElementFormulation, celltype>(linearization, jacobian_mapping,
                    stress, integration_factor, preparation_data, history_data_, gp, *stiff);
              }

              if (mass.has_value())
              {
                if (equal_integration_mass_stiffness)
                {
                  AddMassMatrix(
                      shape_functions, integration_factor, solid_material.Density(gp), *mass);
                }
                else
                {
                  element_mass += solid_material.Density(gp) * integration_factor;
                  element_volume += integration_factor;
                }
              }
            });
      });

  if (mass.has_value() && !equal_integration_mass_stiffness)
  {
    // integrate mass matrix
    FOUR_C_ASSERT(element_mass > 0, "It looks like the element mass is 0.0");
    ForEachGaussPoint<celltype>(nodal_coordinates, mass_matrix_integration_,
        [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, 1>& xi,
            const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp) {
          AddMassMatrix(shape_functions, integration_factor, element_mass / element_volume, *mass);
        });
  }
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::Recover(const DRT::Element& ele,
    const DRT::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::Update(const DRT::Element& ele,
    MAT::So3Material& solid_material, const DRT::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  const ElementNodes<celltype> nodal_coordinates =
      EvaluateElementNodes<celltype>(ele, discretization, lm);

  EvaluateCentroidCoordinatesAndAddToParameterList(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      Prepare(ele, nodal_coordinates, history_data_);

  DRT::ELEMENTS::ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const CORE::LINALG::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        EvaluateGPCoordinatesAndAddToParameterList(nodal_coordinates, shape_functions, params);
        Evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                    deformation_gradient,
                const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
            { solid_material.Update(deformation_gradient, gp, params, ele.Id()); });
      });

  solid_material.Update();
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
double DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::CalculateInternalEnergy(
    const DRT::Element& ele, MAT::So3Material& solid_material,
    const DRT::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  const ElementNodes<celltype> nodal_coordinates =
      EvaluateElementNodes<celltype>(ele, discretization, lm);

  EvaluateCentroidCoordinatesAndAddToParameterList(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      Prepare(ele, nodal_coordinates, history_data_);

  double intenergy = 0;
  DRT::ELEMENTS::ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const CORE::LINALG::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        EvaluateGPCoordinatesAndAddToParameterList(nodal_coordinates, shape_functions, params);
        Evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                    deformation_gradient,
                const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
            {
              double psi = 0.0;
              solid_material.StrainEnergy(gl_strain, psi, gp, ele.Id());
              intenergy += psi * integration_factor;
            });
      });

  return intenergy;
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::CalculateStress(
    const DRT::Element& ele, MAT::So3Material& solid_material, const StressIO& stressIO,
    const StrainIO& strainIO, const DRT::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  std::vector<char>& serialized_stress_data = stressIO.mutable_data;
  std::vector<char>& serialized_strain_data = strainIO.mutable_data;
  CORE::LINALG::SerialDenseMatrix stress_data(stiffness_matrix_integration_.NumPoints(), num_str_);
  CORE::LINALG::SerialDenseMatrix strain_data(stiffness_matrix_integration_.NumPoints(), num_str_);

  const ElementNodes<celltype> nodal_coordinates =
      EvaluateElementNodes<celltype>(ele, discretization, lm);

  EvaluateCentroidCoordinatesAndAddToParameterList(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      Prepare(ele, nodal_coordinates, history_data_);

  DRT::ELEMENTS::ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const CORE::LINALG::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        EvaluateGPCoordinatesAndAddToParameterList(nodal_coordinates, shape_functions, params);
        Evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                    deformation_gradient,
                const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
            {
              const Stress<celltype> stress = EvaluateMaterialStress<celltype>(
                  solid_material, deformation_gradient, gl_strain, params, gp, ele.Id());

              AssembleStrainTypeToMatrixRow<celltype>(
                  gl_strain, deformation_gradient, strainIO.type, strain_data, gp);
              AssembleStressTypeToMatrixRow(
                  deformation_gradient, stress, stressIO.type, stress_data, gp);
            });
      });

  Serialize(stress_data, serialized_stress_data);
  Serialize(strain_data, serialized_strain_data);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::UpdatePrestress(
    const DRT::Element& ele, MAT::So3Material& solid_material,
    const DRT::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  const ElementNodes<celltype> nodal_coordinates =
      EvaluateElementNodes<celltype>(ele, discretization, lm);

  const PreparationData<ElementFormulation> preparation_data =
      Prepare(ele, nodal_coordinates, history_data_);

  DRT::ELEMENTS::UpdatePrestress<ElementFormulation, celltype>(
      ele, nodal_coordinates, preparation_data, history_data_);

  DRT::ELEMENTS::ForEachGaussPoint(nodal_coordinates, stiffness_matrix_integration_,
      [&](const CORE::LINALG::Matrix<DETAIL::num_dim<celltype>, 1>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        EvaluateGPCoordinatesAndAddToParameterList(nodal_coordinates, shape_functions, params);
        Evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                    deformation_gradient,
                const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
            {
              DRT::ELEMENTS::UpdatePrestress<ElementFormulation, celltype>(ele, nodal_coordinates,
                  xi, shape_functions, jacobian_mapping, deformation_gradient, preparation_data,
                  history_data_, gp);
            });
      });
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::Setup(
    MAT::So3Material& solid_material, INPUT::LineDefinition* linedef)
{
  solid_material.Setup(stiffness_matrix_integration_.NumPoints(), linedef);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::MaterialPostSetup(
    const DRT::Element& ele, MAT::So3Material& solid_material)
{
  Teuchos::ParameterList params{};

  // Check if element has fiber nodes, if so interpolate fibers to Gauss Points and add to params
  InterpolateFibersToGaussPointsAndAddToParameterList<celltype>(
      stiffness_matrix_integration_, ele, params);

  // Call PostSetup of material
  solid_material.PostSetup(params, ele.Id());
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::InitializeGaussPointDataOutput(
    const DRT::Element& ele, const MAT::So3Material& solid_material,
    STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.IsParamsInterface(),
      "This action type should only be called from the new time integration framework!");

  AskAndAddQuantitiesToGaussPointDataOutput(
      stiffness_matrix_integration_.NumPoints(), solid_material, gp_data_output_manager);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::EvaluateGaussPointDataOutput(
    const DRT::Element& ele, const MAT::So3Material& solid_material,
    STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.IsParamsInterface(),
      "This action type should only be called from the new time integration framework!");

  CollectAndAssembleGaussPointDataOutput<celltype>(
      stiffness_matrix_integration_, solid_material, ele, gp_data_output_manager);
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
void DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::ResetToLastConverged(
    const DRT::Element& ele, MAT::So3Material& solid_material)
{
  solid_material.ResetStep();
}

template <CORE::FE::CellType celltype, typename ElementFormulation>
double DRT::ELEMENTS::SolidEleCalc<celltype, ElementFormulation>::GetCauchyNDirAtXi(
    const DRT::Element& ele, MAT::So3Material& solid_material, const std::vector<double>& disp,
    const CORE::LINALG::Matrix<3, 1>& xi, const CORE::LINALG::Matrix<3, 1>& n,
    const CORE::LINALG::Matrix<3, 1>& dir, CauchyNDirLinearizations<3>& linearizations)
{
  if constexpr (has_gauss_point_history<ElementFormulation>)
  {
    FOUR_C_THROW(
        "Cannot evaluate the Cauchy stress at xi with an element formulation with Gauss point "
        "history. The element formulation is %s.",
        CORE::UTILS::TryDemangle(typeid(ElementFormulation).name()).c_str());
  }
  else
  {
    ElementNodes<celltype> element_nodes = EvaluateElementNodes<celltype>(ele, disp);

    const ShapeFunctionsAndDerivatives<celltype> shape_functions =
        EvaluateShapeFunctionsAndDerivs<celltype>(xi, element_nodes);

    const JacobianMapping<celltype> jacobian_mapping =
        EvaluateJacobianMapping(shape_functions, element_nodes);

    const PreparationData<ElementFormulation> preparation_data =
        Prepare(ele, element_nodes, history_data_);

    return Evaluate(ele, element_nodes, xi, shape_functions, jacobian_mapping, preparation_data,
        history_data_,
        [&](const CORE::LINALG::Matrix<CORE::FE::dim<celltype>, CORE::FE::dim<celltype>>&
                deformation_gradient,
            const CORE::LINALG::Matrix<num_str_, 1>& gl_strain, const auto& linearization)
        {
          const ElementFormulationDerivativeEvaluator<celltype, ElementFormulation> evaluator(ele,
              element_nodes, xi, shape_functions, jacobian_mapping, deformation_gradient,
              preparation_data, history_data_);

          return EvaluateCauchyNDirAtXi<celltype>(
              solid_material, deformation_gradient, n, dir, ele.Id(), evaluator, linearizations);
        });
  }
}

template <CORE::FE::CellType... celltypes>
struct VerifyPackable
{
  static constexpr bool are_all_packable =
      (DRT::ELEMENTS::IsPackable<DRT::ELEMENTS::SolidEleCalc<celltypes,
              DRT::ELEMENTS::DisplacementBasedFormulation<celltypes>>*> &&
          ...);

  static constexpr bool are_all_unpackable =
      (DRT::ELEMENTS::IsUnpackable<DRT::ELEMENTS::SolidEleCalc<celltypes,
              DRT::ELEMENTS::DisplacementBasedFormulation<celltypes>>*> &&
          ...);

  void StaticAsserts() const
  {
    static_assert(are_all_packable);
    static_assert(are_all_unpackable);
  }
};

template struct VerifyPackable<CORE::FE::CellType::hex8, CORE::FE::CellType::hex18,
    CORE::FE::CellType::hex20, CORE::FE::CellType::hex27, CORE::FE::CellType::nurbs27,
    CORE::FE::CellType::tet4, CORE::FE::CellType::tet10, CORE::FE::CellType::pyramid5,
    CORE::FE::CellType::wedge6, CORE::FE::CellType::hex8, CORE::FE::CellType::hex8,
    CORE::FE::CellType::hex8, CORE::FE::CellType::hex8, CORE::FE::CellType::hex8>;

// explicit instantiations of template classes
// for displacement based formulation
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex8,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::hex8>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex18,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::hex18>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex20,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::hex20>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex27,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::hex27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::nurbs27,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::nurbs27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet4,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::tet4>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet10,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::tet10>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::pyramid5,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::pyramid5>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::wedge6,
    DRT::ELEMENTS::DisplacementBasedFormulation<CORE::FE::CellType::wedge6>>;

// for displacement based formulation with linear kinematics
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex8,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::hex8>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex18,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::hex18>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex20,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::hex20>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex27,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::hex27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::nurbs27,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::nurbs27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet4,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::tet4>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet10,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::tet10>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::pyramid5,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::pyramid5>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::wedge6,
    DRT::ELEMENTS::DisplacementBasedLinearKinematicsFormulation<CORE::FE::CellType::wedge6>>;

// Fbar element technology
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex8,
    DRT::ELEMENTS::FBarFormulation<CORE::FE::CellType::hex8>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::pyramid5,
    DRT::ELEMENTS::FBarFormulation<CORE::FE::CellType::pyramid5>>;

// explicit instantiations for MULF
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex8,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::hex8>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex18,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::hex18>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex20,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::hex20>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex27,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::hex27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::nurbs27,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::nurbs27>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet4,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::tet4>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::tet10,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::tet10>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::pyramid5,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::pyramid5>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::wedge6,
    DRT::ELEMENTS::MulfFormulation<CORE::FE::CellType::wedge6>>;

// explicit instaniations for FBAR+MULF
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::hex8,
    DRT::ELEMENTS::MulfFBarFormulation<CORE::FE::CellType::hex8>>;
template class DRT::ELEMENTS::SolidEleCalc<CORE::FE::CellType::pyramid5,
    DRT::ELEMENTS::MulfFBarFormulation<CORE::FE::CellType::pyramid5>>;

FOUR_C_NAMESPACE_CLOSE