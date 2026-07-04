// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MONOLITHIC_SOLID_SCALAR_MATERIAL_HPP
#define FOUR_C_MAT_MONOLITHIC_SOLID_SCALAR_MATERIAL_HPP

#include "4C_config.hpp"

#include "4C_art_net_input.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <ostream>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{

  //! utility struct to pass nodal information from the solid-scatra elements into the material
  //! evaluation using (optional) vectors
  struct SolidScalarMaterialNodalInput
  {
    //! nodal scalars (first dimension: node, second dimension: scalars)
    std::optional<std::vector<std::vector<double>>> nodal_scalars = std::nullopt;

    //! simplified growth at element nodes as specified in the scatra framework (first dimension:
    //! node, second dimension: direction)
    std::optional<std::vector<std::vector<double>>> nodal_simplified_growths = std::nullopt;

    //! simplified growth derivative wrt concentration at element nodes as specified in the scatra
    //! framework (first dimension: node, second dimension: direction)
    std::optional<std::vector<std::vector<double>>> nodal_simplified_growth_conc_derivs =
        std::nullopt;

    //! simplified growth derivative wrt potential at element nodes as specified in the scatra
    //! framework (first dimension: node, second dimension: direction)
    std::optional<std::vector<std::vector<double>>> nodal_simplified_growth_pot_derivs =
        std::nullopt;

    //! nodal temperatures
    std::optional<std::vector<double>> nodal_temperatures = std::nullopt;

    //! shape functions \f$ \mathbf{N} \f$
    std::vector<double> shape_func;

    //! shape function derivatives wrt reference coordinates \f$ \frac{\mathrm{d}
    //! \mathbf{N}}{\mathrm{d} \boldsymbol{X}} \f$  (first dimension: node, second dimension:
    //! direction)
    std::vector<std::vector<double>> shape_func_derivs_XYZ;

    // DEBUG: print simplified growths
    void print(std::ostream& os) const
    {
      os << std::format("Simplified growths: \n");
      FOUR_C_ASSERT_ALWAYS(nodal_simplified_growths.has_value(), "Stop");
      for (unsigned int n = 0; n < nodal_simplified_growths->size(); ++n)
      {
        os << "n = " << n << ": " << nodal_simplified_growths->at(n)[0] << ", "
           << nodal_simplified_growths->at(n)[1] << ", " << nodal_simplified_growths->at(n)[1]
           << "\n";
      }
    }
  };

  class MonolithicSolidScalarMaterial
  {
   public:
    virtual ~MonolithicSolidScalarMaterial() = default;

    /*!
     * @brief Evaluate the material law, i.e. the stress tensor and the constitutive tensor
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] glstrain Green-Lagrange strain
     * @param[in] params   Container for additional information
     * @param[in] context   Material evaluation context
     * @param[in] nodal_input   Nodal values for the solid-scatra evaluation
     * @param[out] stress  2nd Piola-Kirchhoff stresses
     * @param[out] cmat    Constitutive matrix
     * @param[in] gp       Current Gauss point
     * @param[in] eleGID   Global element ID
     */
    virtual void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        const SolidScalarMaterialNodalInput& nodal_input,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) {};

    /*!
     * @brief Evaluates the added derivatives of the stress w.r.t. all scalars
     *
     * @param defgrad (in) : Deformation gradient
     * @param glstrain (in) : Green-Lagrange strain
     * @param params (in) : ParameterList for additional parameters
     * @param gp (in) : Gauss points
     * @param eleGID (in) : global element id
     * @param nodal_input (in): nodal input including shape functions and derivatives
     * @return std::vector<std::optional<Core::LinAlg::Matrix<6, 1>>>
     */
    virtual Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_d_stress_d_scalar(
        const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int gp,
        int eleGID, const SolidScalarMaterialNodalInput& nodal_input) = 0;

    /*!
     * @brief Evaluates dS/dc_k for all scalar DOFs per node.
     *
     * Returns a vector of num_scalars tensors. Entry k holds dS/dc_k.
     * The default wraps evaluate_d_stress_d_scalar and places the result in entry 0;
     * all other entries are zero (scalars not affecting the stress).
     * Override for materials where the stress is affected by more than one scalar.
     */
    [[nodiscard]] virtual std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
    evaluate_d_stress_d_scalars(const Core::LinAlg::Tensor<double, 3, 3>& defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int num_scalars,
        int gp, int eleGID, const SolidScalarMaterialNodalInput& nodal_input)
    {
      FOUR_C_ASSERT(num_scalars > 0, "num_scalars must be positive");
      std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> result(num_scalars);
      for (auto& t : result) t.fill(0.0);
      result[0] =
          evaluate_d_stress_d_scalar(defgrad, glstrain, params, context, gp, eleGID, nodal_input);

      return result;
    }
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
