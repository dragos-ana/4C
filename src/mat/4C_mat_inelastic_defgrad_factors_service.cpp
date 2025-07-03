// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_io_runtime_csv_writer.hpp"
#include "4C_mat_inelastic_defgrad_factors.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN

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
    case ErrorType::no_plastic_incompressibility:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: plastic incompressibility "
             "not satisfied!";
    case ErrorType::failed_solution_linear_system_lnl:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the Local Newton Loop failed!";
    case ErrorType::failed_determ_line_search_step:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: could not determine a "
             "suitable line search parameter!";
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
             "the flow resistance of the viscoplasticity law";
    case ErrorType::failed_computation_flow_resistance_derivs:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the derivatives of the flow resistance of the viscoplasticity law";
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
      FOUR_C_THROW("to_string(ErrorType): You should not be here!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::local_iteration_status_enum_to_double(
    const LocalIterationStatus iter_status)
{
  switch (iter_status)
  {
    case LocalIterationStatus::converged:
      return 0.0;
    case LocalIterationStatus::final_error:
      return 1.0;
    case LocalIterationStatus::not_evaluated:
      return -1.0;
    case LocalIterationStatus::residual_evaluation_successful:
      return 2.0;
    case LocalIterationStatus::residual_evaluation_failed:
      return 3.0;
    default:
      FOUR_C_THROW("Unhandled IterationStatus {}", EnumTools::enum_name(iter_status));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    reset()
{
  eval_num_of_LNL_steps_ = 0;
  eval_num_of_iters_ = 0;
  eval_num_of_repredict_ = 0;
  eval_num_of_pred_adapt_iters_ = 0;
  eval_num_of_repredict_iters_ = 0;
  eval_num_of_line_search_ = 0;
  eval_num_of_line_search_iters_ = 0;
  eval_teuchos_timer_.reset();
  eval_teuchos_timer_LNL_.reset();
  eval_teuchos_timer_pred_adapt_.reset();
  eval_teuchos_timer_repredict_.reset();
  eval_teuchos_timer_line_search_.reset();
  eval_time_ = 0;
  eval_time_LNL_ = 0;
  eval_time_pred_adapt_ = 0;
  eval_time_repredict_ = 0;
  eval_time_line_search_ = 0;
  eval_error_map_ = {
      {ErrorType::negative_plastic_strain, 0},
      {ErrorType::overflow_error, 0},
      {ErrorType::no_plastic_incompressibility, 0},
      {ErrorType::failed_solution_linear_system_lnl, 0},
      {ErrorType::failed_determ_line_search_step, 0},
      {ErrorType::no_convergence_local_newton, 0},
      {ErrorType::singular_jacobian, 0},
      {ErrorType::failed_solution_analytic_linearization, 0},
      {ErrorType::failed_matrix_log_evaluation, 0},
      {ErrorType::failed_matrix_exp_evaluation, 0},
      {ErrorType::under_yield_surface, 0},
  };
  eval_num_of_alpha_neq_1 = 0;
  eval_num_of_alpha_neq_1_last_iter = 0;
  eval_num_of_first_iter_convergences = 0;
  curr_pred_interp_factor_ = -1.0;
  curr_max_pred_interp_factor_ = -1.0;
  optimal_pred_interp_factor_ = -1.0;
  lnl_res_optimal_pred_interp_factor_ = -1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    update_total()
{
  total_num_of_LNL_steps_ += eval_num_of_LNL_steps_;
  total_num_of_iters_ += eval_num_of_iters_;
  total_num_of_repredict_ += eval_num_of_repredict_;
  total_num_of_pred_adapt_iters_ += eval_num_of_pred_adapt_iters_;
  total_num_of_repredict_iters_ += eval_num_of_repredict_iters_;
  total_num_of_line_search_ += eval_num_of_line_search_;
  total_num_of_line_search_iters_ += eval_num_of_line_search_iters_;
  total_num_of_alpha_neq_1 += eval_num_of_alpha_neq_1;
  total_num_of_alpha_neq_1_last_iter += eval_num_of_alpha_neq_1_last_iter;
  total_time_ += eval_time_;
  total_time_LNL_ += eval_time_LNL_;
  total_time_pred_adapt_ += eval_time_pred_adapt_;
  total_time_repredict_ += eval_time_repredict_;
  total_time_line_search_ += eval_time_line_search_;
  for (const auto& [error_type, error_count] : eval_error_map_)
  {
    total_error_map_[error_type] += error_count;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    write_to_csv()
{
  // get structure discretization
  std::shared_ptr<Core::FE::Discretization> structure_dis =
      Global::Problem::instance()->get_dis("structure");

  // check whether we are using a single processor! (no implementation for multiple processors
  // yet, and also not really required)
  int my_rank = Core::Communication::my_mpi_rank(structure_dis->get_comm());
  FOUR_C_ASSERT_ALWAYS(my_rank == 0,
      "InelasticDefgradTransvIsotropElastViscoplast: No implementation of time integration "
      "output "
      "for multiple processors");

  // create csv_writer and register its columns
  Core::IO::RuntimeCsvWriter csv_writer{
      my_rank, *Global::Problem::instance()->output_control_file(), "timint_output"};
  csv_writer.register_data_vector("Eval. steps (LNL)", 1, 16);
  csv_writer.register_data_vector("Eval. iterations (LNL)", 1, 16);
  csv_writer.register_data_vector("Eval. repredictorizations (LNL)", 1, 16);
  csv_writer.register_data_vector("Eval. iterations (predictor adaptation)", 1, 16);
  csv_writer.register_data_vector("Eval. iterations (repredictorization)", 1, 16);
  csv_writer.register_data_vector("Eval. line searches (LNL)", 1, 16);
  csv_writer.register_data_vector("Eval. iterations (line search)", 1, 16);
  csv_writer.register_data_vector("Eval. # of times: alpha neq 1 (all LNL iters)", 1, 16);
  csv_writer.register_data_vector("Eval. # of times: alpha neq 1 (last LNL iter)", 1, 16);
  csv_writer.register_data_vector("Eval. # of first LNL iter. convergences", 1, 16);
  csv_writer.register_data_vector("Eval. time (full: preevaluate -> update)", 1, 16);
  csv_writer.register_data_vector("Eval. time (LNL)", 1, 16);
  csv_writer.register_data_vector("Eval. time (predictor adaptation)", 1, 16);
  csv_writer.register_data_vector("Eval. time (repredictorization)", 1, 16);
  csv_writer.register_data_vector("Eval. time (line search)", 1, 16);
  csv_writer.register_data_vector("Total steps (LNL)", 1, 16);
  csv_writer.register_data_vector("Total iterations (LNL)", 1, 16);
  csv_writer.register_data_vector("Total repredictorizations (LNL)", 1, 16);
  csv_writer.register_data_vector("Total iterations (predictor adaptation)", 1, 16);
  csv_writer.register_data_vector("Total iterations (repredictorization)", 1, 16);
  csv_writer.register_data_vector("Total line searches (LNL)", 1, 16);
  csv_writer.register_data_vector("Total iterations (line search)", 1, 16);
  csv_writer.register_data_vector("Total # of times: alpha neq 1 (all LNL iters)", 1, 16);
  csv_writer.register_data_vector("Total # of times: alpha neq 1 (last LNL iter)", 1, 16);
  csv_writer.register_data_vector("Total # of first LNL iter. convergences", 1, 16);
  csv_writer.register_data_vector("Total time (full: preevaluate -> update)", 1, 16);
  csv_writer.register_data_vector("Total time (LNL)", 1, 16);
  csv_writer.register_data_vector("Total time (predictor adaptation)", 1, 16);
  csv_writer.register_data_vector("Total time (repredictorization)", 1, 16);
  csv_writer.register_data_vector("Total time (line search)", 1, 16);
  csv_writer.register_data_vector(
      "Interpolation factor of GP 0 of Ele 0 (last global iteration)", 1, 16);
  csv_writer.register_data_vector(
      "Interpolation factor of GP 0 of Ele 0 (maximum over all global "
      "iterations)",
      1, 16);
  csv_writer.register_data_vector("Interpolation factor of GP 0 of Ele 0 (optimal)", 1, 16);
  csv_writer.register_data_vector(
      "LNL Residual: Interpolation factor of GP 0 of Ele 0 (optimal)", 1, 16);
  for (ErrorType err_type : magic_enum::enum_values<ErrorType>())
  {
    csv_writer.register_data_vector(
        "Eval. Error " + std::string(magic_enum::enum_name(err_type)), 1, 16);
    csv_writer.register_data_vector(
        "Total Error " + std::string(magic_enum::enum_name(err_type)), 1, 16);
  }

  // output data
  std::map<std::string, std::vector<double>> output_data;
  output_data["Eval. steps (LNL)"] = {static_cast<double>(eval_num_of_LNL_steps_)};
  output_data["Total steps (LNL)"] = {static_cast<double>(total_num_of_LNL_steps_)};
  output_data["Eval. iterations (LNL)"] = {static_cast<double>(eval_num_of_iters_)};
  output_data["Total iterations (LNL)"] = {static_cast<double>(total_num_of_iters_)};
  output_data["Eval. repredictorizations (LNL)"] = {static_cast<double>(eval_num_of_repredict_)};
  output_data["Total repredictorizations (LNL)"] = {static_cast<double>(total_num_of_repredict_)};
  output_data["Eval. iterations (predictor adaptation)"] = {
      static_cast<double>(eval_num_of_pred_adapt_iters_)};
  output_data["Total iterations (predictor adaptation)"] = {
      static_cast<double>(total_num_of_pred_adapt_iters_)};
  output_data["Eval. iterations (repredictorization)"] = {
      static_cast<double>(eval_num_of_repredict_iters_)};
  output_data["Total iterations (repredictorization)"] = {
      static_cast<double>(total_num_of_repredict_iters_)};
  output_data["Eval. iterations (line search)"] = {
      static_cast<double>(eval_num_of_line_search_iters_)};
  output_data["Total iterations (line search)"] = {
      static_cast<double>(total_num_of_line_search_iters_)};
  output_data["Eval. line searches (LNL)"] = {static_cast<double>(eval_num_of_line_search_)};
  output_data["Total line searches (LNL)"] = {static_cast<double>(total_num_of_line_search_)};
  output_data["Eval. time (full: preevaluate -> update)"] = {static_cast<double>(eval_time_)};
  output_data["Total time (full: preevaluate -> update)"] = {static_cast<double>(total_time_)};
  output_data["Eval. time (LNL)"] = {static_cast<double>(eval_time_LNL_)};
  output_data["Total time (LNL)"] = {static_cast<double>(total_time_LNL_)};
  output_data["Eval. time (predictor adaptation)"] = {static_cast<double>(eval_time_pred_adapt_)};
  output_data["Total time (predictor adaptation)"] = {static_cast<double>(total_time_pred_adapt_)};
  output_data["Eval. time (repredictorization)"] = {static_cast<double>(eval_time_repredict_)};
  output_data["Total time (repredictorization)"] = {static_cast<double>(total_time_repredict_)};
  output_data["Eval. time (line search)"] = {static_cast<double>(eval_time_line_search_)};
  output_data["Total time (line search)"] = {static_cast<double>(total_time_line_search_)};
  output_data["Eval. # of times: alpha neq 1 (all LNL iters)"] = {
      static_cast<double>(eval_num_of_alpha_neq_1)};
  output_data["Eval. # of times: alpha neq 1 (last LNL iter)"] = {
      static_cast<double>(eval_num_of_alpha_neq_1_last_iter)};
  output_data["Eval. # of first LNL iter. convergences"] = {
      static_cast<double>(eval_num_of_first_iter_convergences)};
  output_data["Total # of times: alpha neq 1 (all LNL iters)"] = {
      static_cast<double>(total_num_of_alpha_neq_1)};
  output_data["Total # of times: alpha neq 1 (last LNL iter)"] = {
      static_cast<double>(total_num_of_alpha_neq_1_last_iter)};
  output_data["Total # of first LNL iter. convergences"] = {
      static_cast<double>(total_num_of_first_iter_convergences)};


  for (ErrorType err_type : magic_enum::enum_values<ErrorType>())
  {
    output_data["Eval. Error " + std::string(magic_enum::enum_name(err_type))] = {
        static_cast<double>(eval_error_map_[err_type])};
    output_data["Total Error " + std::string(magic_enum::enum_name(err_type))] = {
        static_cast<double>(total_error_map_[err_type])};
  }

  // predictor interpolation factors
  output_data
      ["Interpolation factor of GP 0 of Ele 0 (last global "
       "iteration)"] = {static_cast<double>(curr_pred_interp_factor_)};
  output_data
      ["Interpolation factor of GP 0 of Ele 0 (maximum over all global "
       "iterations)"] = {static_cast<double>(curr_max_pred_interp_factor_)};
  output_data["Interpolation factor of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(optimal_pred_interp_factor_)};
  output_data["LNL Residual: Interpolation factor of GP 0 of Ele 0 (optimal)"] = {
      static_cast<double>(lnl_res_optimal_pred_interp_factor_)};


  // write output data to csv
  csv_writer.write_data_to_file(sim_time_, sim_timestep_, output_data);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::GeneralLocalTimIntAnalysisUtils::
    output_error_local_newton_loop(unsigned int step_counter)
{  // add current number of steps
  eval_num_of_LNL_steps_ += step_counter;

  // stop (already started!) LNL timer
  eval_time_LNL_ += eval_teuchos_timer_LNL_.stop();

  // output routine
  eval_time_ = eval_teuchos_timer_.stop();
  update_total();
  write_to_csv();
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputPredAdaptMicroIterData::
    CSVOutputPredAdaptMicroIterData(
        const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputTrackingData
            csv_output_tracking_data)
    : csv_output_tracking_data_(csv_output_tracking_data)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputPredAdaptMicroIterData::
    append_micro_iter_data(
        const MicroIterDataCollector mi_data_collector, const unsigned micro_iter)
{
  // consistency check: verify whether data has already been written for
  // the given micro iteration
  FOUR_C_ASSERT_ALWAYS(
      std::find(all_microiter_.begin(), all_microiter_.end(), micro_iter) == all_microiter_.end(),
      "You have already written predictor adaptation data for microiteration {}", micro_iter);

  // append data for the given microiteration
  all_microiter_.push_back(micro_iter);
  all_current_xi_.push_back(mi_data_collector.current_xi_);
  all_current_equiv_stress_.push_back(mi_data_collector.current_equiv_stress_);
  all_current_plastic_strain_.push_back(mi_data_collector.current_plastic_strain_);
  all_current_error_status_.push_back(mi_data_collector.current_error_status_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputLineSearchMicroIterData::
    CSVOutputLineSearchMicroIterData(
        const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputTrackingData
            csv_output_tracking_data)
    : csv_output_tracking_data_(csv_output_tracking_data)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputLineSearchMicroIterData::
    append_micro_iter_data(
        const MicroIterDataCollector mi_data_collector, const unsigned micro_iter)
{
  // consistency check: verify whether data has already been written for
  // the given micro iteration
  FOUR_C_ASSERT_ALWAYS(
      std::find(all_microiter_.begin(), all_microiter_.end(), micro_iter) == all_microiter_.end(),
      "You have already written line search data for microiteration {}", micro_iter);

  // append data for the given microiteration
  all_microiter_.push_back(micro_iter);
  all_current_alpha_.push_back(mi_data_collector.current_alpha_);
  all_max_alpha_.push_back(mi_data_collector.max_alpha_);
  all_current_equiv_stress_.push_back(mi_data_collector.current_equiv_stress_);
  all_current_plastic_strain_.push_back(mi_data_collector.current_plastic_strain_);
  all_current_quadratic_residual_norm_.push_back(
      mi_data_collector.current_quadratic_residual_norm_);
  all_max_quadratic_residual_norm_.push_back(mi_data_collector.max_quadratic_residual_norm_);
  all_current_error_status_.push_back(mi_data_collector.current_error_status_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputPredAdaptMicroIterData::
    write_pred_adapt_micro_iter_data_to_csv()
{
  // get structure discretization
  std::shared_ptr<Core::FE::Discretization> structure_dis =
      Global::Problem::instance()->get_dis("structure");

  // check whether we are using a single processor! (no implementation for multiple
  // processors yet, and also not really required)
  int my_rank = Core::Communication::my_mpi_rank(structure_dis->get_comm());
  FOUR_C_ASSERT_ALWAYS(my_rank == 0,
      "InelasticDefgradTransvIsotropElastViscoplast: No implementation of time integration "
      "output "
      "for multiple processors");

  // create csv_writer and register its columns
  Core::IO::RuntimeCsvWriter csv_writer{my_rank,
      *Global::Problem::instance()->output_control_file(),
      "pred-adapt-micro-iter-output-ele-gid-" + std::to_string(csv_output_tracking_data_.ele_gid) +
          "-gp-" + std::to_string(csv_output_tracking_data_.gp) + "-tn-" +
          std::to_string(csv_output_tracking_data_.tn) + "-globiter-or-timestep-index-" +
          std::to_string(csv_output_tracking_data_.globiter_or_timestep_index_) + "-lnl-iter-" +
          std::to_string(csv_output_tracking_data_.lnl_iter)};
  csv_writer.register_data_vector("element_gid", 1, 16);
  csv_writer.register_data_vector("gauss_point", 1, 16);
  csv_writer.register_data_vector("previous_time", 1, 16);
  csv_writer.register_data_vector("globiter_or_timestep_index", 1, 16);
  csv_writer.register_data_vector("lnl_iter", 1, 16);
  csv_writer.register_data_vector("current_xi", 1, 16);
  csv_writer.register_data_vector("current_equiv_stress", 1, 16);
  csv_writer.register_data_vector("current_plastic_strain", 1, 16);
  csv_writer.register_data_vector("current_err_status", 1, 16);

  // already fill the columns containing solely the tracking data
  for (unsigned int mi = 0; mi < all_microiter_.size(); ++mi)
  {
    std::map<std::string, std::vector<double>> output_data;
    output_data["element_gid"] = {static_cast<double>(csv_output_tracking_data_.ele_gid)};
    output_data["gauss_point"] = {static_cast<double>(csv_output_tracking_data_.gp)};
    output_data["previous_time"] = {static_cast<double>(csv_output_tracking_data_.tn)};
    output_data["globiter_or_timestep_index"] = {
        static_cast<double>(csv_output_tracking_data_.globiter_or_timestep_index_)};
    output_data["lnl_iter"] = {static_cast<double>(csv_output_tracking_data_.lnl_iter)};
    output_data["current_xi"] = {static_cast<double>(all_current_xi_[mi])};
    output_data["current_equiv_stress"] = {static_cast<double>(all_current_equiv_stress_[mi])};
    output_data["current_plastic_strain"] = {static_cast<double>(all_current_plastic_strain_[mi])};
    switch (all_current_error_status_[mi])
    {
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors:
        output_data["current_err_status"] = {0.0};
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::overflow_error:
        output_data["current_err_status"] = {1.0};
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::under_yield_surface:
        output_data["current_err_status"] = {2.0};
        break;
      default:
        output_data["current_err_status"] = {-1.0};
        break;
    }

    // write output data to csv
    csv_writer.write_data_to_file(csv_output_tracking_data_.tnp, mi, output_data);
  }
}

//! writes data from each microiteration of a single line search (specified via tracking data)
//! to a dedicated csv file
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::CSVOutputLineSearchMicroIterData::
    write_line_search_micro_iter_data_to_csv()
{
  // get structure discretization
  std::shared_ptr<Core::FE::Discretization> structure_dis =
      Global::Problem::instance()->get_dis("structure");

  // check whether we are using a single processor! (no implementation for multiple
  // processors yet, and also not really required)
  int my_rank = Core::Communication::my_mpi_rank(structure_dis->get_comm());
  FOUR_C_ASSERT_ALWAYS(my_rank == 0,
      "InelasticDefgradTransvIsotropElastViscoplast: No implementation of time integration "
      "output "
      "for multiple processors");

  // create csv_writer and register its columns
  Core::IO::RuntimeCsvWriter csv_writer{my_rank,
      *Global::Problem::instance()->output_control_file(),
      "line-search-micro-iter-output-ele-gid-" + std::to_string(csv_output_tracking_data_.ele_gid) +
          "-gp-" + std::to_string(csv_output_tracking_data_.gp) + "-tn-" +
          std::to_string(csv_output_tracking_data_.tn) + "-globiter-or-timestep-index-" +
          std::to_string(csv_output_tracking_data_.globiter_or_timestep_index_) + "-lnl-iter-" +
          std::to_string(csv_output_tracking_data_.lnl_iter)};
  csv_writer.register_data_vector("element_gid", 1, 16);
  csv_writer.register_data_vector("gauss_point", 1, 16);
  csv_writer.register_data_vector("previous_time", 1, 16);
  csv_writer.register_data_vector("globiter_or_timestep_index", 1, 16);
  csv_writer.register_data_vector("lnl_iter", 1, 16);
  csv_writer.register_data_vector("current_alpha", 1, 16);
  csv_writer.register_data_vector("max_alpha", 1, 16);
  csv_writer.register_data_vector("current_equiv_stress", 1, 16);
  csv_writer.register_data_vector("current_plastic_strain", 1, 16);
  csv_writer.register_data_vector("current_quadratic_residual_norm", 1, 16);
  csv_writer.register_data_vector("max_quadratic_residual_norm", 1, 16);
  csv_writer.register_data_vector("current_err_status", 1, 16);

  // already fill the columns containing solely the tracking data
  for (unsigned int mi = 0; mi < all_microiter_.size(); ++mi)
  {
    std::map<std::string, std::vector<double>> output_data;
    output_data["element_gid"] = {static_cast<double>(csv_output_tracking_data_.ele_gid)};
    output_data["gauss_point"] = {static_cast<double>(csv_output_tracking_data_.gp)};
    output_data["previous_time"] = {static_cast<double>(csv_output_tracking_data_.tn)};
    output_data["globiter_or_timestep_index"] = {
        static_cast<double>(csv_output_tracking_data_.globiter_or_timestep_index_)};
    output_data["lnl_iter"] = {static_cast<double>(csv_output_tracking_data_.lnl_iter)};
    output_data["current_alpha"] = {static_cast<double>(all_current_alpha_[mi])};
    output_data["max_alpha"] = {static_cast<double>(all_max_alpha_[mi])};
    output_data["current_equiv_stress"] = {static_cast<double>(all_current_equiv_stress_[mi])};
    output_data["current_plastic_strain"] = {static_cast<double>(all_current_plastic_strain_[mi])};
    output_data["current_quadratic_residual_norm"] = {
        static_cast<double>(all_current_quadratic_residual_norm_[mi])};
    output_data["max_quadratic_residual_norm"] = {
        static_cast<double>(all_max_quadratic_residual_norm_[mi])};
    switch (all_current_error_status_[mi])
    {
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors:
        output_data["current_err_status"] = {0.0};
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::overflow_error:
        output_data["current_err_status"] = {1.0};
        break;
      case InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::under_yield_surface:
        output_data["current_err_status"] = {2.0};
        break;
      default:
        output_data["current_err_status"] = {-1.0};
        break;
    }

    // write output data to csv
    csv_writer.write_data_to_file(csv_output_tracking_data_.tnp, mi, output_data);
  }
}


FOUR_C_NAMESPACE_CLOSE
