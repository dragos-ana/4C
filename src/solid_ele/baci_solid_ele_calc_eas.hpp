/*! \file

\brief Declaration of routines for calculation of solid element with EAS element technology

\level 1
*/

#ifndef BACI_SOLID_ELE_CALC_EAS_HPP
#define BACI_SOLID_ELE_CALC_EAS_HPP

#include "baci_config.hpp"

#include "baci_discretization_fem_general_utils_gausspoints.hpp"
#include "baci_lib_element.hpp"
#include "baci_solid_ele_calc_interface.hpp"
#include "baci_solid_ele_interface_serializable.hpp"
#include "baci_solid_ele_utils.hpp"

#include <memory>
#include <string>
#include <unordered_map>

BACI_NAMESPACE_OPEN

namespace STR::ELEMENTS
{
  enum class EasType
  {
    soh8_easnone,
    eastype_h8_9,
    eastype_h8_21,
    eastype_sh8_7,
    eastype_sh18_9,
    eastype_undefined
  };

  template <STR::ELEMENTS::EasType eastype>
  struct EasTypeToNumEas
  {
  };
  template <>
  struct EasTypeToNumEas<STR::ELEMENTS::EasType::eastype_h8_9>
  {
    static constexpr int num_eas = 9;
  };
  template <>
  struct EasTypeToNumEas<STR::ELEMENTS::EasType::eastype_h8_21>
  {
    static constexpr int num_eas = 21;
  };
  template <>
  struct EasTypeToNumEas<STR::ELEMENTS::EasType::eastype_sh8_7>
  {
    static constexpr int num_eas = 7;
  };
  template <>
  struct EasTypeToNumEas<STR::ELEMENTS::EasType::eastype_sh18_9>
  {
    static constexpr int num_eas = 9;
  };
  template <>
  struct EasTypeToNumEas<STR::ELEMENTS::EasType::eastype_undefined>
  {
  };

}  // namespace STR::ELEMENTS

namespace MAT
{
  class So3Material;
}

namespace STR::MODELEVALUATOR
{
  class GaussPointDataOutputManager;
}
namespace DRT
{

  namespace ELEMENTS
  {
    /// struct for EAS matrices and vectors to be stored between iterations
    template <CORE::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
    struct EasIterationData
    {
      constexpr static int num_eas = STR::ELEMENTS::EasTypeToNumEas<eastype>::num_eas;

      /// inverse EAS matrix K_{alpha alpha}
      CORE::LINALG::Matrix<num_eas, num_eas> invKaa_{true};

      /// EAS matrix K_{d alpha}
      CORE::LINALG::Matrix<CORE::FE::num_nodes<celltype> * CORE::FE::dim<celltype>, num_eas> Kda_{
          true};

      /// EAS enhacement vector s
      CORE::LINALG::Matrix<num_eas, 1> s_{true};

      /// discrete enhanced strain scalars alpha
      CORE::LINALG::Matrix<num_eas, 1> alpha_{true};
    };

    template <CORE::FE::CellType celltype, STR::ELEMENTS::EasType eastype>
    class SolidEleCalcEas
    {
     public:
      SolidEleCalcEas();

      void Setup(MAT::So3Material& solid_material, INPUT::LineDefinition* linedef);

      void Pack(CORE::COMM::PackBuffer& data) const;

      void Unpack(std::vector<char>::size_type& position, const std::vector<char>& data);

      void MaterialPostSetup(const DRT::Element& ele, MAT::So3Material& solid_material);

      void EvaluateNonlinearForceStiffnessMass(const DRT::Element& ele,
          MAT::So3Material& solid_material, const DRT::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params,
          CORE::LINALG::SerialDenseVector* force_vector,
          CORE::LINALG::SerialDenseMatrix* stiffness_matrix,
          CORE::LINALG::SerialDenseMatrix* mass_matrix);

      void Recover(const DRT::Element& ele, const DRT::Discretization& discretization,
          const std::vector<int>& lm, Teuchos::ParameterList& params);

      void CalculateStress(const DRT::Element& ele, MAT::So3Material& solid_material,
          const StressIO& stressIO, const StrainIO& strainIO,
          const DRT::Discretization& discretization, const std::vector<int>& lm,
          Teuchos::ParameterList& params);

      double CalculateInternalEnergy(const DRT::Element& ele, MAT::So3Material& solid_material,
          const DRT::Discretization& discretization, const std::vector<int>& lm,
          Teuchos::ParameterList& params);

      void Update(const DRT::Element& ele, MAT::So3Material& solid_material,
          const DRT::Discretization& discretization, const std::vector<int>& lm,
          Teuchos::ParameterList& params);

      void InitializeGaussPointDataOutput(const DRT::Element& ele,
          const MAT::So3Material& solid_material,
          STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const;

      void EvaluateGaussPointDataOutput(const DRT::Element& ele,
          const MAT::So3Material& solid_material,
          STR::MODELEVALUATOR::GaussPointDataOutputManager& gp_data_output_manager) const;

      void ResetToLastConverged(const DRT::Element& ele, MAT::So3Material& solid_material);

     private:
      /// EAS matrices and vectors to be stored between iterations
      DRT::ELEMENTS::EasIterationData<celltype, eastype> eas_iteration_data_ = {};

      /// static values for matrix sizes
      static constexpr int num_nodes_ = CORE::FE::num_nodes<celltype>;
      static constexpr int num_dim_ = CORE::FE::dim<celltype>;
      static constexpr int num_dof_per_ele_ = num_nodes_ * num_dim_;
      static constexpr int num_str_ = num_dim_ * (num_dim_ + 1) / 2;

      CORE::FE::GaussIntegration stiffness_matrix_integration_;
      CORE::FE::GaussIntegration mass_matrix_integration_;
    };  // class SolidEleCalcEas
  }     // namespace ELEMENTS
}  // namespace DRT

BACI_NAMESPACE_CLOSE

#endif  // SOLID_ELE_CALC_EAS_H