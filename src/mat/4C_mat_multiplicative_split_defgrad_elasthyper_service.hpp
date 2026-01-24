// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_SERVICE_HPP
#define FOUR_C_MAT_MULTIPLICATIVE_SPLIT_DEFGRAD_ELASTHYPER_SERVICE_HPP
#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_matrix_conversion.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_service.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{

  enum class ThermalExpansionMaterialType
  {
    isotropic,
    anisotropic
  };

  inline void evaluate_ce(const Core::LinAlg::Matrix<3, 3>& F,
      const Core::LinAlg::Matrix<3, 3>& iFin, Core::LinAlg::Matrix<3, 3>& Ce)
  {
    static Core::LinAlg::Matrix<3, 3> FiFin(Core::LinAlg::Initialization::uninitialized);
    FiFin.multiply_nn(F, iFin);
    Ce.multiply_tn(FiFin, FiFin);
  }

  inline void evaluatei_cin_ci_cin(const Core::LinAlg::Matrix<3, 3>& C,
      const Core::LinAlg::Matrix<3, 3>& iCin, Core::LinAlg::Matrix<3, 3>& iCinCiCin)
  {
    static Core::LinAlg::Matrix<3, 3> CiCin(Core::LinAlg::Initialization::uninitialized);
    CiCin.multiply_nn(C, iCin);
    iCinCiCin.multiply_nn(iCin, CiCin);
  }

  inline void elast_hyper_evaluate_elastic_part(const Core::LinAlg::Matrix<3, 3>& F,
      const Core::LinAlg::Matrix<3, 3>& iFin, Core::LinAlg::Matrix<6, 1>& S_stress,
      Core::LinAlg::Matrix<6, 6>& cmat,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
      Mat::SummandProperties summandProperties, const int gp, const int eleGID)
  {
    if (summandProperties.anisomod or summandProperties.anisoprinc)
    {
      FOUR_C_THROW(
          "An additional inelastic part is not yet implemented for anisotropic materials.");
    }

    S_stress.clear();
    cmat.clear();

    // Variables needed for the computation of the stress resultants
    static Core::LinAlg::Matrix<3, 3> C(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> Ce(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iC(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iCin(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 3> iCinCiCin(Core::LinAlg::Initialization::zero);

    static Core::LinAlg::Matrix<6, 1> iCinv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<6, 1> iCinCiCinv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<6, 1> iCv(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 1> principleInvariantsCe(Core::LinAlg::Initialization::zero);

    // Compute right Cauchy-Green tensor C=F^TF
    C.multiply_tn(F, F);

    // Compute inverse right Cauchy-Green tensor C^-1
    iC.invert(C);

    // Compute inverse inelastic right Cauchy-Green Tensor
    iCin.multiply_nt(iFin, iFin);

    // Compute iCin * C * iCin
    Mat::evaluatei_cin_ci_cin(C, iCin, iCinCiCin);

    // Compute Ce
    Mat::evaluate_ce(F, iFin, Ce);

    // Compute principal invariants
    Mat::invariants_principal(principleInvariantsCe, Ce);

    Core::LinAlg::Matrix<3, 1> dPIe(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddPIIe(Core::LinAlg::Initialization::zero);

    Mat::elast_hyper_evaluate_invariant_derivatives(
        principleInvariantsCe, dPIe, ddPIIe, potsum, summandProperties, gp, eleGID);

    // 2nd Piola Kirchhoff stress factors (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
    static Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
    // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
    static Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);

    Mat::calculate_gamma_delta(gamma, delta, principleInvariantsCe, dPIe, ddPIIe);

    // Convert necessary tensors to stress-like Voigt-Notation
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCin, iCinv);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCin, iCinCiCinv);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iC, iCv);

    // Contribution to 2nd Piola-Kirchhoff stress tensor
    S_stress.update(gamma(0), iCinv, 1.0);
    S_stress.update(gamma(1), iCinCiCinv, 1.0);
    S_stress.update(gamma(2), iCv, 1.0);

    // Contribution to the linearization
    cmat.multiply_nt(delta(0), iCinv, iCinv, 1.);
    cmat.multiply_nt(delta(1), iCinCiCinv, iCinv, 1.);
    cmat.multiply_nt(delta(1), iCinv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(2), iCinv, iCv, 1.);
    cmat.multiply_nt(delta(2), iCv, iCinv, 1.);
    cmat.multiply_nt(delta(3), iCinCiCinv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(4), iCinCiCinv, iCv, 1.);
    cmat.multiply_nt(delta(4), iCv, iCinCiCinv, 1.);
    cmat.multiply_nt(delta(5), iCv, iCv, 1.);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmat, iCv, delta(6));
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmat, iCinv, delta(7));
  }

  inline void elast_hyper_evaluate_elastic_stress_and_stiffness(
      const Core::LinAlg::Matrix<3, 3>& Ce, const Core::LinAlg::Matrix<3, 1>& gamma,
      const Core::LinAlg::Matrix<8, 1>& delta, Core::LinAlg::Matrix<6, 1>& SeV,
      Core::LinAlg::Matrix<6, 6>& cmateV)
  {
    SeV.clear();
    cmateV.clear();

    // compute terms relevant for the computation
    Core::LinAlg::SymmetricTensor<double, 3, 3> id =
        Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
    Core::LinAlg::Matrix<6, 1> idV = Core::LinAlg::make_stress_like_voigt_view(id);
    Core::LinAlg::Matrix<6, 1> CeV{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(Ce, CeV);
    Core::LinAlg::Matrix<3, 3> iCe{Core::LinAlg::Initialization::zero};
    iCe.invert(Ce);
    Core::LinAlg::Matrix<6, 1> iCeV{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCe, iCeV);

    // contribution to elastic 2nd Piola-Kirchhoff stress tensor
    SeV.update(gamma(0), idV, 1.0);
    SeV.update(gamma(1), CeV, 1.0);
    SeV.update(gamma(2), iCeV, 1.0);

    // Contribution to the linearization
    cmateV.multiply_nt(delta(0), idV, idV, 1.);
    cmateV.multiply_nt(delta(1), idV, CeV, 1.);
    cmateV.multiply_nt(delta(1), CeV, idV, 1.);
    cmateV.multiply_nt(delta(2), idV, iCeV, 1.);
    cmateV.multiply_nt(delta(2), iCeV, idV, 1.);
    cmateV.multiply_nt(delta(3), CeV, CeV, 1.);
    cmateV.multiply_nt(delta(4), CeV, iCeV, 1.);
    cmateV.multiply_nt(delta(4), iCeV, CeV, 1.);
    cmateV.multiply_nt(delta(5), iCeV, iCeV, 1.);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmateV, iCeV, delta(6));
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmateV, idV, delta(7));
  }



}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
