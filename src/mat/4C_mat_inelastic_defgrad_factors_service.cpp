// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::get_detailed_error_message_for_error_type(
    ErrorType err_type)
{
  switch (err_type)
  {
    case ErrorType::negative_plastic_strain:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
    case ErrorType::overflow_error:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
             "the evaluation of the plastic strain increment!";
    case ErrorType::failed_solution_linear_system_lnl:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the Local Newton Loop failed!";
    case ErrorType::no_convergence_local_newton:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
             "converge for the given loop settings!";
    case ErrorType::singular_jacobian:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after "
             "converged Local Newton Loop, which does not allow for the analytical evaluation "
             "of the linearization!";
    case ErrorType::failed_solution_analytic_linearization:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the analytical linearization failed";
    case ErrorType::failed_computation_flow_resistance:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the flow resistance for the viscoplasticity law";
    case ErrorType::failed_computation_flow_resistance_derivs:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the derivatives of the flow resistance for the viscoplasticity law";
    case ErrorType::failed_matrix_log_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix logarithm or its derivative with respect to the argument";
    case ErrorType::failed_matrix_exp_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix exponential or its derivative with respect to the argument";
    case ErrorType::failed_right_cg_interpolation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in interpolating "
             "the right Cauchy-Green deformation tensor";
    case ErrorType::under_yield_surface:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: we are 'under' the yield "
             "surface, sigma < sigma_yield!";
    default:
      FOUR_C_THROW("to_string(ErrorType): {}: No error message provided!", err_type);
  }
}

Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstNonMatTensors::ConstNonMatTensors()
{  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) unit3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);

  // set constant non-material tensors

  // 3x3 identity
  id3x3.update(1.0, unit3x3, 0.0);

  // Voigt stress form of 3x3 identity
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      id3x3, id6x1);

  // symmetric identity four tensor
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(id4_6x6, 1.0, id3x3, id3x3, 0.0);

  // deviatoric operator
  Core::LinAlg::FourTensor<3> dev_op_four_tensor =
      Core::LinAlg::setup_deviatoric_projection_tensor<3>();
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, dev_op_four_tensor);
  dev_op = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // identity four tensor
  id4_9x9.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3, id3x3, id4_9x9);

  // 10x10 identity
  id10x10.clear();
  for (int i = 0; i < 10; ++i) id10x10(i, i) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstMatTensors::
    set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m)
{
  // get instance of constant non-material tensors
  const auto& const_non_mat_tensors = ConstNonMatTensors::instance();

  // set material-dependent tensors (fiber orientation)

  // structural tensor
  mm.multiply_nt(1.0, m, m, 0.0);

  // deviatoric part of the structural tensor
  double tr_mm_ = mm(0, 0) + mm(1, 1) + mm(2, 2);
  mm_dev.update(1.0, mm, -1.0 / 3.0 * tr_mm_, const_non_mat_tensors.id3x3);

  // dyadic product of structural tensors
  Core::LinAlg::Matrix<6, 1> mm_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm, mm_V);
  mm_dyad_mm.multiply_nt(1.0, mm_V, mm_V, 0.0);

  // dyadic product of deviatoric structural tensor with the structural tensor
  Core::LinAlg::Matrix<6, 1> mm_dev_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_dev, mm_dev_V);
  mm_dev_dyad_mm.multiply_nt(1.0, mm_dev_V, mm_V, 0.0);

  // dyadic product of identity with the structural tensor
  id_dyad_mm.multiply_nt(1.0, const_non_mat_tensors.id6x1, mm_V, 0.0);

  // sum of identity with the structural tensor
  id_plus_mm.update(1.0, const_non_mat_tensors.id3x3, 1.0, mm, 0.0);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::reset(
    const double dt)
{
  t_ = 0.0;
  substep_counter_ = 1;
  curr_dt_ = dt;
  time_step_halving_counter_ = 0;
  total_num_of_substeps_ = 1;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::
    increment_substep()
{
  t_ += curr_dt_;
  substep_counter_++;
};

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::halve_substep()
{
  curr_dt_ *= 1.0 / 2.0;
  time_step_halving_counter_ += 1;
  total_num_of_substeps_ += (total_num_of_substeps_ - substep_counter_ + 1);
};


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::init()
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
  for (unsigned int i = 0; i < 3; ++i)
  {
    id3x3(i, i) = 1.0;
  }

  // ----- set last_ and current_ variables referring to values at different time instants
  // ----- for now: the number of Gauss points is unknown -> we set the values only for 1
  // Gauss point and update the number of Gauss points in the setup method

  // default values of the inverse plastic deformation gradient: unit tensor
  last_plastic_defgrad_inverse.resize(1, id3x3);
  current_plastic_defgrad_inverse.resize(1, id3x3);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(1, id3x3);

  // update last_ and current_ values of the plastic strain
  last_plastic_strain.resize(1, 0.0);
  current_plastic_strain.resize(1, 0.0);  // value irrelevant at this point
  last_substep_plastic_strain.resize(1, 0.0);

  // update last_ and current_ values of the equivalent stress
  last_equiv_stress.resize(1, 0.0);
  current_equiv_stress.resize(1, 0.0);  // value irrelevant at this point

  // default values of the right CG tensor: unit tensor
  last_rightCG.resize(1, id3x3);
  current_rightCG.resize(1, id3x3);  // value irrelevant at this point

  // default value for the current deformation gradient: zero tensor \f$ \boldsymbol{0} f$ (to make
  // sure that the inverse inelastic deformation gradient is evaluated in the first method call)
  last_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{id3x3});
  current_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called,
      "You already called resize for the time step quantities! The number of current GP is {} and "
      "you attempt to set it to {}",
      last_plastic_strain.size(), numgp);


  // default values of the inverse plastic deformation gradient for ALL Gauss Points
  last_plastic_defgrad_inverse.resize(numgp, last_plastic_defgrad_inverse[0]);
  current_plastic_defgrad_inverse.resize(numgp,
      last_plastic_defgrad_inverse[0]);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(numgp, last_substep_plastic_defgrad_inverse[0]);

  // default values of the plastic strain for ALL Gauss Points
  last_plastic_strain.resize(numgp, last_plastic_strain[0]);
  current_plastic_strain.resize(numgp, last_plastic_strain[0]);  // value irrelevant at this point
  last_substep_plastic_strain.resize(numgp, last_substep_plastic_strain[0]);

  // default values of the right CG deformation tensor for ALL Gauss Points
  last_rightCG.resize(numgp, last_rightCG[0]);
  current_rightCG.resize(numgp, last_rightCG[0]);  // value irrelevant at this point

  // default values of the equivalent stress for ALL Gauss Points
  last_equiv_stress.resize(numgp, last_equiv_stress[0]);
  current_equiv_stress.resize(numgp, current_equiv_stress[0]);  // value irrelevant at this point

  // default values of the deformation gradient
  last_defgrad.resize(numgp, last_defgrad[0]);
  current_defgrad.resize(numgp, current_defgrad[0]);

  resize_called = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pre_evaluate(
    const unsigned int gp)
{
  // set consistent last substep values
  last_substep_plastic_defgrad_inverse[gp] = last_plastic_defgrad_inverse[gp];
  last_substep_plastic_strain[gp] = last_plastic_strain[gp];
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::update()
{
  // update history variables for the next time step
  last_defgrad = current_defgrad;
  last_rightCG = current_rightCG;
  last_plastic_defgrad_inverse = current_plastic_defgrad_inverse;
  last_substep_plastic_defgrad_inverse = current_plastic_defgrad_inverse;
  last_plastic_strain = current_plastic_strain;
  last_equiv_stress = current_equiv_stress;
  last_substep_plastic_strain = current_plastic_strain;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, last_defgrad);
  add_to_pack(data, last_rightCG);
  add_to_pack(data, last_plastic_defgrad_inverse);
  add_to_pack(data, last_plastic_strain);
  add_to_pack(data, last_equiv_stress);
  add_to_pack(data, last_substep_plastic_defgrad_inverse);
  add_to_pack(data, last_substep_plastic_strain);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, last_defgrad);
  extract_from_pack(buffer, last_rightCG);
  extract_from_pack(buffer, last_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_plastic_strain);
  extract_from_pack(buffer, last_equiv_stress);
  extract_from_pack(buffer, last_substep_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_substep_plastic_strain);

  // fill current_ values with the last_ values
  current_rightCG.resize(last_rightCG.size(),
      last_rightCG[0]);  // value irrelevant
  current_plastic_defgrad_inverse.resize(last_plastic_defgrad_inverse.size(),
      last_plastic_defgrad_inverse[0]);  // value irrelevant
  current_plastic_strain.resize(last_plastic_strain.size(),
      last_plastic_strain[0]);  // value irrelevant
  current_equiv_stress.resize(last_equiv_stress.size(),
      last_equiv_stress[0]);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic deformation
  // gradient is evaluated fully after the restart
  current_defgrad.resize(last_substep_plastic_defgrad_inverse.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::LocalNewtonManager(
    const LocalNewtonParams& lnl_params)
    : params_(lnl_params)
{
  // set number of Gauss points to 1 temporarily, since we don't
  // know it at this point in time
  curr_num_iters_.resize(1, 0);

  // set initial number of iterations to 0
  iter_ = 0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the Local Newton manager! The number of current GP is {} and "
      "you attempt to set it to {}",
      curr_num_iters_.size(), numgp);

  // resize arrays
  curr_num_iters_.resize(numgp, curr_num_iters_[0]);


  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    update_after_local_newton(const unsigned int gp)
{
  // increment number of local Newton iterations for the current timestep at the
  // current GP
  curr_num_iters_[gp] += iter_;
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::reset()
{
  std::ranges::fill(curr_num_iters_, 0);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::InterpolationPointContainer()
{
  current_interp_points_.resize(1, 0.0);
  lower_interp_bounds_.resize(1, 0.0);
  upper_interp_bounds_.resize(1, 1.0);
  last_interp_points_.resize(1, 0.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::reset(const unsigned int gp)
{
  current_interp_points_[gp] = 0.0;
  lower_interp_bounds_[gp] = 0.0;
  upper_interp_bounds_[gp] = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::resize(const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the interpolation point container! The number of "
      "current GP is "
      "{} and "
      "you attempt to set it to {}",
      current_interp_points_.size(), numgp);

  current_interp_points_.resize(numgp, current_interp_points_[0]);
  lower_interp_bounds_.resize(numgp, lower_interp_bounds_[0]);
  upper_interp_bounds_.resize(numgp, upper_interp_bounds_[0]);
  last_interp_points_.resize(numgp, last_interp_points_[0]);

  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::update()
{
  for (unsigned int gp = 0; gp < last_interp_points_.size(); ++gp)
  {
    last_interp_points_[gp] = current_interp_points_[gp];
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, current_interp_points_);
  Core::Communication::add_to_pack(data, lower_interp_bounds_);
  Core::Communication::add_to_pack(data, upper_interp_bounds_);
  Core::Communication::add_to_pack(data, last_interp_points_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    InterpolationPointContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_from_pack(buffer, current_interp_points_);
  Core::Communication::extract_from_pack(buffer, lower_interp_bounds_);
  Core::Communication::extract_from_pack(buffer, upper_interp_bounds_);
  Core::Communication::extract_from_pack(buffer, last_interp_points_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    ElasticDefgradPredictorDecompositions::ElasticDefgradPredictorDecompositions()
{
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    ElasticDefgradPredictorDecompositions::construct_prelim_plastic_pred(const unsigned int gp,
        const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
        const AdaptiveEstimateInterpolationParams& aei_params,
        const Core::LinAlg::Matrix<3, 3>& last_elastic_defgrad)
{
  //  perform polar-spectral decomposition of elastic defgrad within elastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_elast_pred{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(elastic_defgrad_elastic_pred, rot_elast_pred_[gp],
      material_stretch_elast_pred, eigenval_elast_pred_[gp], spectral_pairs_elast_pred);
  for (int i = 0; i < 3; ++i)
  {
    FOUR_C_ASSERT_ALWAYS(eigenval_elast_pred_[gp](i, i) >= 1.0e-8,
        "The eigenvalue {} of the elastic deformation gradient within the elastic predictor at GP "
        "{} is {}, "
        "such that its logarithm can not be computed!",
        i, gp, eigenval_elast_pred_[gp](i, i));
    log_eigenval_elast_pred_[gp][i] = std::log(eigenval_elast_pred_[gp](i, i));
    for (int j = 0; j < 3; ++j)
    {
      eigenvect_rot_elast_pred_[gp](i, j) = spectral_pairs_elast_pred[i].second(j);
    }
  }

  // -->  construct a preliminary plastic predictor based on the parameter specifications

  // we currently only enable the transfer of rotation contributions from the elastic predictor; but
  // make the approach of using different rotation contributions possible in the future
  FOUR_C_ASSERT_ALWAYS(aei_params.plast_pred_elast_rot_type ==
                           PlasticPredictorElasticRotationType::from_elastic_predictor,
      "Elastic rotation type {} not yet enabled for the plastic predictor!",
      EnumTools::enum_name(aei_params.plast_pred_elast_rot_type));
  rel_eigenvect_rot_plast_pred_[gp].scale(0.0);
  Core::LinAlg::Matrix<3, 3> rel_eigenvect_rot_plast_pred_matrix =
      Core::LinAlg::calc_rot_matrix_from_rot_vect(rel_eigenvect_rot_plast_pred_[gp]);
  Core::LinAlg::Matrix<3, 3> eigenvect_rot_plast_pred_matrix{Core::LinAlg::Initialization::zero};
  eigenvect_rot_plast_pred_matrix.multiply(
      1.0, eigenvect_rot_elast_pred_[gp], rel_eigenvect_rot_plast_pred_matrix, 0.0);

  FOUR_C_ASSERT_ALWAYS(aei_params.plast_pred_elast_stretch_eigenvect_type ==
                           PlasticPredictorElasticStretchEigenvectType::from_elastic_predictor,
      "Elastic stretch eigenvector type {} not yet enabled for the plastic predictor!",
      EnumTools::enum_name(aei_params.plast_pred_elast_stretch_eigenvect_type));
  rel_rot_plast_pred_[gp].scale(0.0);
  Core::LinAlg::Matrix<3, 3> rel_rot_plast_pred_matrix =
      Core::LinAlg::calc_rot_matrix_from_rot_vect(rel_rot_plast_pred_[gp]);
  Core::LinAlg::Matrix<3, 3> rot_plast_pred_matrix{Core::LinAlg::Initialization::zero};
  rot_plast_pred_matrix.multiply(1.0, rot_elast_pred_[gp], rel_rot_plast_pred_matrix, 0.0);


  // set elastic eigenvalues based on specification
  const double detF = elastic_defgrad_elastic_pred.determinant();
  switch (aei_params.plast_pred_elast_stretch_eigenval_type)
  {
    case PlasticPredictorElasticStretchEigenvalType::scale_unit:
    {
      const double scaled_detF = std::pow(detF, 1.0 / 3.0);
      for (unsigned int i = 0; i < 3; ++i)
      {
        eigenval_plast_pred_[gp] = scaled_detF;
      }

      break;
    }
    case PlasticPredictorElasticStretchEigenvalType::scale_previous:
    {
      // compute scaling factor \f$\left[ \det(\mathbf{F}_{n+1}) / \det(\mathbf{F}_{\mathrm{e},n})
      // \right]^{1/3} \f$
      const double detF_detFen = detF / last_elastic_defgrad.determinant();
      const double scaled_detF_detFen = std::pow(detF_detFen, 1.0 / 3.0);

      // perform polar-spectral decomposition of the last elastic defgrad
      Core::LinAlg::Matrix<3, 3> last_material_stretch{Core::LinAlg::Initialization::zero};
      Core::LinAlg::Matrix<3, 3> last_rot{Core::LinAlg::Initialization::zero};
      std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> last_spectral_pairs;
      Core::LinAlg::matrix_3x3_polar_decomposition(last_elastic_defgrad, last_rot,
          last_material_stretch, eigenval_plast_pred_[gp], last_spectral_pairs);
      // scale the eigenvalues
      eigenval_plast_pred_[gp].scale(scaled_detF_detFen);

      break;
    }
    default:
    {
      FOUR_C_ASSERT_ALWAYS(
          "Elastic stretch eigenvector type {} not yet enabled for the plastic predictor",
          EnumTools::enum_name(aei_params.plast_pred_elast_stretch_eigenval_type));
    }
  }
  for (int i = 0; i < 3; ++i)
  {
    FOUR_C_ASSERT_ALWAYS(eigenval_plast_pred_[gp](i, i) >= 1.0e-8,
        "The eigenvalue {} of the elastic deformation gradient within the plastic predictor at GP "
        "{} is {}, "
        "such that its logarithm can not be computed!",
        i, gp, eigenval_plast_pred_[gp](i, i));
    log_eigenval_plast_pred_[gp][i] = std::log(eigenval_plast_pred_[gp](i, i));
  }



  defgrad_elast_pred_[gp] = elastic_defgrad_elastic_pred;
  defgrad_plast_pred_[gp] = elastic_defgrad_pair.plastic_predictor_defgrad;



  //  perform polar decomposition of defgrad within plastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_fatrix_plast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> rotation_matrix_plast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_matrix_plast_pred{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_plast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(defgrad_plast_pred, rotation_matrix_plast_pred,
      material_stretch_matrix_plast_pred, eigenval_matrix_plast_pred, spectral_pairs_plast_pred);

  // collect all spectral pairs (elastic and plastic predictors) and use
  // this for ordering and alignment: we will rewrite them back to their
  // original form afterwards
  std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>> all_spectral_pairs{
      spectral_pairs_elast_pred, spectral_pairs_plast_pred};

  // set reference locations for interpolation
  Core::LinAlg::Matrix<1, 1> ref_loc_elast_pred;
  ref_loc_elast_pred(0, 0) = 0.0;
  Core::LinAlg::Matrix<1, 1> ref_loc_plast_pred;
  ref_loc_plast_pred(0, 0) = 1.0;
  std::vector<Core::LinAlg::Matrix<1, 1>> all_ref_locs{ref_loc_elast_pred, ref_loc_plast_pred};

  // align spectral pairs of reference (elastic predictor) suitably
  Core::LinAlg::align_eigenpairs_of_base_matrix(all_spectral_pairs, all_ref_locs, 0);

  // order eigenpairs (plastic predictor) with respect to the reference (elastic predictor)
  if (spectral_pairs_ref.has_value())
  {
    Core::LinAlg::order_eigenpairs_wrt_reference(spectral_pairs_ref.value(), all_spectral_pairs[1]);
    // save spectral pairs within designated objects
    spectral_pairs_elast_pred_ = spectral_pairs_ref.value();
  }
  else
  {
    Core::LinAlg::order_eigenpairs_wrt_reference(all_spectral_pairs[0], all_spectral_pairs[1]);
    // save spectral pairs within designated objects
    spectral_pairs_elast_pred_ = all_spectral_pairs[0];
  }
  spectral_pairs_plast_pred_ = all_spectral_pairs[1];

  // save eigenvalues
  lambda_elast_pred_ = {spectral_pairs_elast_pred_[0].first, spectral_pairs_elast_pred_[1].first,
      spectral_pairs_elast_pred_[2].first};
  lambda_plast_pred_ = {spectral_pairs_plast_pred_[0].first, spectral_pairs_plast_pred_[1].first,
      spectral_pairs_plast_pred_[2].first};
  log_lambda_elast_pred_ = {std::log(spectral_pairs_elast_pred_[0].first),
      std::log(spectral_pairs_elast_pred_[1].first), std::log(spectral_pairs_elast_pred_[2].first)};
  log_lambda_plast_pred_ = {std::log(spectral_pairs_plast_pred_[0].first),
      std::log(spectral_pairs_plast_pred_[1].first), std::log(spectral_pairs_plast_pred_[2].first)};

  // save eigenvector rotation matrices and vectors
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      Qmat_elast_pred_(i, j) = spectral_pairs_elast_pred_[i].second(j);
      Qmat_plast_pred_(i, j) = spectral_pairs_plast_pred_[i].second(j);
    }
  }
  Qmat_plast_pred_rel_.multiply_tn(1.0, Qmat_elast_pred_, Qmat_plast_pred_, 0.0);
  Qvec_plast_pred_rel_ = Core::LinAlg::calc_rot_vect_from_rot_matrix(Qmat_plast_pred_rel_);

  // save rotation matrices and vectors
  Rmat_elast_pred_ = rotation_matrix_elast_pred;
  Rmat_plast_pred_ = rotation_matrix_plast_pred;
  Rmat_plast_pred_rel_.multiply_tn(1.0, Rmat_elast_pred_, Rmat_plast_pred_, 0.0);
  Rvec_plast_pred_rel_ = Core::LinAlg::calc_rot_vect_from_rot_matrix(Rmat_plast_pred_rel_);
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    AdaptiveEstimateInterpolationManager(const AdaptiveEstimateInterpolationParams& aei_params)
    : params_(aei_params), interp_point_container_(),
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit3x3{Core::LinAlg::Initialization::zero};
  for (int i = 0; i < 3; ++i) unit3x3(i, i) = 1.0;


  // initialize class variables (for a single Gauss point for now)
  resize_called_ = false;
  num_of_reestimations_ = 0;
  InterpolationPointContainer default_interp_point_container = InterpolationPointContainer();
  current_elastic_defgrad_decompositions_.resize(
      1, AdaptiveEstimateInterpolationManager::ElasticDefgradDecomposition(PredictorDefgradPair{
             .elastic_predictor_defgrad = unit3x3, .plastic_predictor_defgrad = unit3x3}));
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    resize(const unsigned int num_gp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the adaptive estimate interpolation manager! You attempt to "
      "set it to {}",
      num_gp);

  interp_point_container_.resize(num_gp);
  current_elastic_defgrad_decompositions_.resize(
      num_gp, current_elastic_defgrad_decompositions_[0]);

  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    is_interpolation_possible(const unsigned int num_interp_iters)
{
  // check interpolation interval
  const double diff_bounds = interp_point_container_.upper_interp_bound(gp_) -
                             interp_point_container_.lower_interp_bound(gp_);
  bool check_min_interp_interval = (diff_bounds >= params_.min_interval_length);

  // check number of interpolation iterations
  bool check_interp_iters = (num_interp_iters <= params_.max_num_estimate_interpol_iters);

  // check number of re-estimations
  bool check_num_reestimations = (num_of_reestimations_ <= params_.max_num_reestimations);

  return check_min_interp_interval && check_interp_iters && check_num_reestimations;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    pre_evaluate(const unsigned int gp, const PredictorDefgradPair& elastic_defgrad_pair)
{
  // set Gauss point index
  gp_ = gp;

  // reset certain variables
  num_of_reestimations_ = 0;
  interp_point_container_.reset(gp_);

  // decompose the elastic deformation gradient pair
  current_elastic_defgrad_decompositions_[gp_] = ElasticDefgradDecomposition(elastic_defgrad_pair);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    update()
{
  // update Gauss point values
  interp_point_container_.update();

  last_elastic_defgrad_decompositions_ = current_elastic_defgrad_decompositions_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    pack(Core::Communication::PackBuffer& data) const
{
  interp_point_container_.pack(data);
  for (unsigned int gp = 0; gp < last_elastic_defgrad_pred_decomp_.size(); ++gp)
  {
    last_elastic_defgrad_decompositions_[gp].pack(data);
    current_elastic_defgrad_decompositions_[gp].pack(data);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    unpack(Core::Communication::UnpackBuffer& buffer)
{
  interp_point_container_.unpack(buffer);

  for (unsigned int gp = 0; gp < numgp; ++gp)
  {
    interp_point_containers_[gp].unpack(buffer);
    last_elastic_defgrad_decompositions_[gp].unpack(buffer);
    current_elastic_defgrad_decompositions_[gp].unpack(buffer);
  }
}

FOUR_C_NAMESPACE_CLOSE
