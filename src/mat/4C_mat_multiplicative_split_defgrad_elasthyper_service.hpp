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

  /// struct containing various kinematic quantities for the model evaluation
  struct KinematicQuantities
  {
    // ----- variables of kinematic quantities ----- //

    /// inverse right Cauchy-Green tensor \f$ \mathbf{C}^{-1} \f$ stored as 6x1 vector
    Core::LinAlg::Matrix<6, 1> iCV{Core::LinAlg::Initialization::zero};
    /// inverse inelastic right Cauchy-Green tensor \f$\mathbf{C}_\text{in}^{-1}\f$ stored as 6x1
    /// vector
    Core::LinAlg::Matrix<6, 1> iCinV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{C}_\text{in}^{-1} \cdot \mathbf{C} \cdot \mathbf{C}_\text{in}^{-1} \f$ stored
    /// as 6x1 vector
    Core::LinAlg::Matrix<6, 1> iCinCiCinV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{C}_\text{in}^{-1} \cdot \mathbf{C} \f$
    Core::LinAlg::Matrix<3, 3> iCinCM{Core::LinAlg::Initialization::zero};
    ///\f$ \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el} \f$
    Core::LinAlg::Matrix<3, 3> iFinCeM{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \f$ stored as 9x1 vector
    Core::LinAlg::Matrix<9, 1> CiFin9x1{Core::LinAlg::Initialization::zero};
    ///\f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el} \f$ stored as 9x1
    /// vector
    Core::LinAlg::Matrix<9, 1> CiFinCe9x1{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{C} \cdot \mathbf{F}_\text{in}^{-1} \cdot \mathbf{C}_\text{el}^{-1} \f$ stored
    /// as 9x1 vector
    Core::LinAlg::Matrix<9, 1> CiFiniCe9x1{Core::LinAlg::Initialization::zero};
    /// principal invariants of the elastic right Cauchy-Green tensor
    Core::LinAlg::Matrix<3, 1> prinv{Core::LinAlg::Initialization::zero};
    /// partial derivative of the elastic right Cauchy-Green tensor w.r.t. right Cauchy-Green
    /// tensor \f$ \frac{\partial \boldsymbol{C}_\text{e}}{\partial \boldsymbol{C}} \f$ (Voigt
    /// stress-stress notation)
    Core::LinAlg::Matrix<6, 6> dCedC{Core::LinAlg::Initialization::zero};
    /// partial derivative of the elastic right Cauchy-Green tensor w.r.t. inelastic deformation
    /// gradient \f$ \frac{\partial \boldsymbol{C}_\text{e}}{\partial \boldsymbol{F} _\text{in} ^
    /// { -1 }} \f$(Voigt stress notation)
    Core::LinAlg::Matrix<6, 9> dCediFin{Core::LinAlg::Initialization::zero};
    /// inverse inelastic deformation gradient
    Core::LinAlg::Matrix<3, 3> iFinM{Core::LinAlg::Initialization::zero};
    /// determinant of the inelastic deformation gradient
    double detFin = 1.0;

    // ----- derivatives of principal invariants ----- //

    /// first derivatives of principle invariants
    Core::LinAlg::Matrix<3, 1> dPIe{Core::LinAlg::Initialization::zero};
    /// second derivatives of principle invariants
    Core::LinAlg::Matrix<6, 1> ddPIIe{Core::LinAlg::Initialization::zero};
  };

  /// struct containing various quantities related to temperature for the model evaluation
  struct ThermalQuantities
  {
    // ----- variables of thermal quantities ----- //
    /// thermal right Cauchy-Green deformation tensor \f$ \mathbf{C}_T \f$ stored as 6x1
    /// vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> CTV{Core::LinAlg::Initialization::zero};
    /// inverse thermal right Cauchy-Green deformation tensor \f$ \mathbf{C}_T{-1} \f$ stored as
    /// 6x1 vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> iCTV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{F}_{\text{in}}^{-1} \mathbf{C}_T \mathbf{F}_{\text{in}}^{-T} \f$ stored as 6x1
    /// vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> iFinCTiFinTV{Core::LinAlg::Initialization::zero};
    /// \f$ \mathbf{F}_{\text{in}}^{-1} \mathbf{C}_T^{-1} \mathbf{F}_{\text{in}}^{-T} \f$ stored
    /// as 6x1 vector (stress-form!)
    Core::LinAlg::Matrix<6, 1> iFiniCTiFinTV{Core::LinAlg::Initialization::zero};
    /// derivative of thermal right Cauchy-Green deformation tensor wrt temperature \f$ \mathrm{d}
    /// \mathbf{C}_T / \mathrm{d} T \f$ stored as 6x1 vector (strain-form!)
    Core::LinAlg::Matrix<6, 1> dCTdTV{Core::LinAlg::Initialization::zero};

    /// principal invariants of the thermal right Cauchy-Green tensor
    Core::LinAlg::Matrix<3, 1> prinv{Core::LinAlg::Initialization::zero};

    // ----- derivatives of principal invariants ----- //

    /// first derivatives of principle invariants
    Core::LinAlg::Matrix<3, 1> dPI{Core::LinAlg::Initialization::zero};
    /// second derivatives of principle invariants
    Core::LinAlg::Matrix<6, 1> ddPII{Core::LinAlg::Initialization::zero};
  };



  /// struct containing free-energy related stress factors, as presented in Holzapfel-Nonlinear
  /// Solid Mechanics
  struct StressFactors
  {
    // ----- gamma and delta factors ----- //

    // 2nd Piola Kirchhoff stresses factors (according to Holzapfel-Nonlinear Solid Mechanics p.
    // 216)
    Core::LinAlg::Matrix<3, 1> gamma{Core::LinAlg::Initialization::zero};
    // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
    Core::LinAlg::Matrix<8, 1> delta{Core::LinAlg::Initialization::zero};
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


  /*!
   * @brief calculates the derivatives of the hyper-elastic laws with respect to the invariants
   *
   * @param[in] prinv   Principal invariants of the elastic right Cauchy-Green tensor
   * @param[in] gp      current gauss point
   * @param[in] eleGID  Element ID
   * @param[in] potsumel Vector of hyperelastic summands
   * @param[out] dPI    First derivative w.r.t. principle invariants
   * @param[out] ddPII  Second derivative w.r.t. principle invariants
   * @param[out] ddPII  Second derivative w.r.t. principle invariants
   */

  inline void evaluate_invariant_derivatives(const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
      const int eleGID, const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsumel,
      Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII)
  {
    // clear variables
    dPI.clear();
    ddPII.clear();

    // loop over map of associated potential summands
    // derivatives of strain energy function w.r.t. principal invariants
    for (const auto& p : potsumel)  // only for isotropic components
    {
      p->add_derivatives_principal(dPI, ddPII, prinv, gp, eleGID);
    }
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


  /// S_T
  inline Core::LinAlg::Matrix<6, 1> evaluate_thermal_stress(const ThermalQuantities& thermal_quant,
      const StressFactors& thermal_stress_fact, const Core::LinAlg::Matrix<6, 1>& iCinV,
      const double detFin)
  {
    Core::LinAlg::Matrix<6, 1> thermal_stress{Core::LinAlg::Initialization::zero};

    // extract variables from thermal_quant
    Core::LinAlg::Matrix<3, 3> CT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(thermal_quant.CTV, CT);
    const Core::LinAlg::Matrix<6, 1>& iFinCTiFinTV = thermal_quant.iFinCTiFinTV;
    const Core::LinAlg::Matrix<6, 1>& iFiniCTiFinTV = thermal_quant.iFiniCTiFinTV;
    const Core::LinAlg::Matrix<3, 1>& thermal_gamma = thermal_stress_fact.gamma;

    // clear variables
    thermal_stress.clear();

    // 2nd Piola Kirchhoff stresses
    thermal_stress.update(thermal_gamma(0), iCinV, 1.0);
    thermal_stress.update(thermal_gamma(1), iFinCTiFinTV, 1.0);
    thermal_stress.update(thermal_gamma(2), iFiniCTiFinTV, 1.0);
    thermal_stress.scale(detFin);

    return thermal_stress;
  }



  /// \frac{\partial S_T}{\partial T}
  inline Core::LinAlg::Matrix<6, 1> evaluate_thermal_stress_deriv(
      const Core::LinAlg::Matrix<3, 3>& iFinM, const ThermalQuantities& thermal_quant,
      const StressFactors& thermal_stress_fact)
  {
    Core::LinAlg::Matrix<6, 1> thermal_stress_deriv{Core::LinAlg::Initialization::zero};

    // extract variables from thermal_quant
    Core::LinAlg::Matrix<3, 3> CT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(thermal_quant.CTV, CT);
    const Core::LinAlg::Matrix<6, 1>& dCTdTV = thermal_quant.dCTdTV;
    const Core::LinAlg::Matrix<3, 1>& thermal_gamma = thermal_stress_fact.gamma;
    const Core::LinAlg::Matrix<8, 1>& thermal_delta = thermal_stress_fact.delta;
    const double detFin = 1.0 / iFinM.determinant();

    // clear variables
    thermal_stress_deriv.clear();

    // evaluate purely hyperelastic stiffness with the thermal right CG tensor as input
    Core::LinAlg::Matrix<6, 1> hyperelast_stress{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<6, 6> hyperelast_stiffness{Core::LinAlg::Initialization::zero};
    elast_hyper_evaluate_elastic_stress_and_stiffness(
        CT, thermal_gamma, thermal_delta, hyperelast_stress, hyperelast_stiffness);


    // compute derivative \f$ \frac{\partial \mathbf{S}_{\theta}}{\partial T} \f$
    Core::LinAlg::Matrix<6, 1> pStheta_pT_stress{Core::LinAlg::Initialization::zero};
    pStheta_pT_stress.multiply_nn(1.0, hyperelast_stiffness, dCTdTV, 0.0);
    Core::LinAlg::Matrix<3, 3> pStheta_pT{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(pStheta_pT_stress, pStheta_pT);

    // compute product \f$ \mathbf{F}_{\text{in}}^{-1}  \frac{\partial \mathbf{S}_{\theta}}{\partial
    // T} \mathbf{F}_{\text{in}}^{-T} \f$
    Core::LinAlg::Matrix<3, 3> iFin_pStheta_pT{Core::LinAlg::Initialization::zero};
    iFin_pStheta_pT.multiply_nn(1.0, iFinM, pStheta_pT, 0.0);
    Core::LinAlg::Matrix<3, 3> iFin_pStheta_pT_iFinT{Core::LinAlg::Initialization::zero};
    iFin_pStheta_pT_iFinT.multiply_nt(1.0, iFin_pStheta_pT, iFinM, 0.0);
    Core::LinAlg::Matrix<6, 1> iFin_pStheta_pT_iFinT_V{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFin_pStheta_pT_iFinT, iFin_pStheta_pT_iFinT_V);

    // thermal derivative
    thermal_stress_deriv.update(-1.0, iFin_pStheta_pT_iFinT_V, 0.0);
    thermal_stress_deriv.scale(detFin);


    return thermal_stress_deriv;
  }

  inline ThermalQuantities evaluate_thermal_quantities(const double delta_temperature,
      const ThermalExpansionMaterialType thermal_expansion_mat_type,
      const double thermal_expansion_fac, const Core::LinAlg::Matrix<3, 3>& iFinM, const int gp,
      const int eleGID, const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsumel)
  {
    ThermalQuantities quantities{};

    // verify the thermal expansion material type
    FOUR_C_ASSERT_ALWAYS(thermal_expansion_mat_type == ThermalExpansionMaterialType::isotropic,
        "Only isotropic thermal expansion is enabled currently!");

    // compute the thermal stretch, along with its temperature
    // derivative
    Core::LinAlg::SymmetricTensor<double, 3, 3> thermal_right_cg_tensor{
        Core::LinAlg::TensorGenerators::identity<double, 3, 3>};
    Core::LinAlg::SymmetricTensor<double, 3, 3> thermal_right_cg_temp_deriv_tensor{};
    thermal_right_cg_tensor += 2 * thermal_expansion_fac * delta_temperature *
                               Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
    thermal_right_cg_temp_deriv_tensor +=
        2 * thermal_expansion_fac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

    // compute inverse of the thermal stretch
    Core::LinAlg::SymmetricTensor<double, 3, 3> inv_thermal_right_cg_tensor =
        inv(thermal_right_cg_tensor);

    // get matrices for the thermal stretch
    const Core::LinAlg::Matrix<3, 3> CTM =
        Core::LinAlg::make_matrix(get_full(thermal_right_cg_tensor));
    const Core::LinAlg::Matrix<3, 3> iCTM =
        Core::LinAlg::make_matrix(Core::LinAlg::get_full(inv_thermal_right_cg_tensor));

    // compute terms with iFin
    Core::LinAlg::Matrix<3, 3> iFinCT{};
    iFinCT.multiply(1.0, iFinM, CTM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFinCTiFinT{};
    iFinCTiFinT.multiply_nt(1.0, iFinCT, iFinM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFiniCT{};
    iFiniCT.multiply(1.0, iFinM, iCTM, 0.0);
    Core::LinAlg::Matrix<3, 3> iFiniCTiFinT{};
    iFiniCTiFinT.multiply_nt(1.0, iFiniCT, iFinM, 0.0);

    // add computed tensors to quantities in the specified form
    quantities.CTV = Core::LinAlg::make_stress_like_voigt_view(thermal_right_cg_tensor);
    quantities.iCTV = Core::LinAlg::make_stress_like_voigt_view(inv_thermal_right_cg_tensor);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFinCTiFinT, quantities.iFinCTiFinTV);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(iFiniCTiFinT, quantities.iFiniCTiFinTV);
    quantities.dCTdTV = Core::LinAlg::make_strain_like_voigt_matrix(
        thermal_right_cg_temp_deriv_tensor);  // must be in strain-form for contraction afterwards!

    // compute principal invariants of the thermal stretch
    Core::LinAlg::Voigt::Strains::invariants_principal(quantities.prinv, quantities.CTV);

    // compute derivatives of the thermal stretch principal invariants
    evaluate_invariant_derivatives(
        quantities.prinv, gp, eleGID, potsumel, quantities.dPI, quantities.ddPII);


    return quantities;
  }

#define DEBUG_THERMO_VPLAST ;

}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
