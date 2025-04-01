// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <magic_enum/magic_enum.hpp>

#include <map>
#include <ostream>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. repredictorization, substepping) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      NoErrors,
      NegativePlasticStrain,  ///< negative plastic strain which does not allow for evaluations
                              ///< inside the viscoplasticity laws
      OverflowError,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                      ///< (and \f$ \mathsymbol{E}^{\text{p}}  = \exp(- \Delta t
                      ///< \dot{\varepsilon}^{\text{p}} \mathsymbol{N}^{\text{p}}) \f$)
      NoFlowResistance,  ///< the material has no flow resistance anymore, such that the evaluations
                         ///< model non-physical phenomena
      NoPlasticIncompressibility,  ///< no plastic incompressibility, meaning that our determinant
                                   ///< of the inelastic defgrad is far from
                                   ///< 1
      FailedSolLinSystLNL,  ///< solution of the linear system in the Local Newton-Raphson Loop
                            ///< failed
      FailedDetermLineSearchParam,  ///< the computation of a suitable line search parameter failed
      NoConvergenceLNL,  ///< the Local Newton Loop did not converge for the given loop settings
      SingularJacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                         ///< analytical evaluation of the linearization
      FailedSolAnalytLinearization,     ///< solution of the linear system in the analytical
                                        ///< linearization failed
      FailedComputationFlowResistance,  ///< failed in the computation of the flow resistance via
                                        ///< time integration of the hardening-rate equation (e.g.,
                                        ///< Anand model)
      FailedComputationFlowResistanceDerivs,  ///< failed in the computation of the flow resistance
                                              ///< derivatives (e.g., Anand model)
      FailedLogEval,        ///< failed evaluation of the matrix logarithm or its derivative
      FailedExpEval,        ///< failed evaluation of the matrix exponential or its derivative
      FailedRightCGInterp,  ///< failed interpolation of the right Cauchy-Green tensor
    };

    /// enum class for error management actions in InelasticDefgradTransvIsotropElastViscoplast
    enum class ErrorAction
    {
      Continue,             ///< continue without any errors (NoErrors)
      ReturnSolWithErrors,  ///< return the current solution with errors (if the current simulation
                            ///< settings cannot lead to a solution)
      NextIter,             ///< go to next iteration after performing certain reset steps
    };


    /// to_string: error types to error messages in InelasticDefgradTransvIsotropElastViscoplast
    inline std::string to_string(ErrorType err_type)
    {
      switch (err_type)
      {
        case ErrorType::NegativePlasticStrain:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
        case ErrorType::OverflowError:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
                 "the evaluation of the plastic strain increment!";
        case ErrorType::NoPlasticIncompressibility:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: plastic incompressibility "
                 "not satisfied!";
        case ErrorType::FailedSolLinSystLNL:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
                 "system in the Local Newton Loop failed!";
        case ErrorType::FailedDetermLineSearchParam:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: could not determine a "
                 "suitable line search parameter!";
        case ErrorType::NoConvergenceLNL:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
                 "converge for the given loop settings!";
        case ErrorType::SingularJacobian:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after "
                 "converged Local Newton Loop, which does not allow for the analytical evaluation "
                 "of "
                 "the linearization!";
        case ErrorType::FailedSolAnalytLinearization:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
                 "system "
                 "in the analytical linearization failed";
          break;
        case ErrorType::FailedComputationFlowResistance:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
                 "the "
                 "flow resistance of the viscoplasticity law";
          break;
        case ErrorType::FailedComputationFlowResistanceDerivs:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
                 "the "
                 "derivatives of the flow resistance of the viscoplasticity law";
          break;
        case ErrorType::FailedLogEval:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
                 "matrix logarithm or its derivative with respect to the argument";
          break;
        case ErrorType::FailedExpEval:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
                 "matrix exponential or its derivative with respect to the argument";
          break;
        case ErrorType::FailedRightCGInterp:
          return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in interpolating "
                 "the "
                 "right Cauchy-Green deformation tensor";
          break;
        default:
          FOUR_C_THROW("to_string(ErrorType): You should not be here!");
      }
    }

    /// enum class for material behavior types
    /// (InelasticDefgradTransvIsotropElastViscoplast)
    enum class MatBehavior
    {
      isotrop,         ///< isotropic material behavior
      transv_isotrop,  ///< isotropic material behavior
    };

    /// enum class for time integration types (integration of internal
    /// variables in the Local Newton Loop of InelasticDefgradTransvIsotropElastViscoplast)
    enum class TimIntType
    {
      standard,     ///< standard time integration,
      logarithmic,  ///< time integration with logarithmically transformed residual equation for the
                    ///< evolution of the plastic deformation gradient
    };

    /// enum class for material linearization types (linearization of
    /// InelasticDefgradTransvIsotropElastViscoplast)
    enum class LinearizationType
    {
      analytic,       ///< analytical linearization involving the solution of a linear system of
                      ///< equations,
      perturb_based,  ///< linearization based on perturbing the current state
    };

    // names of the various error types
    inline std::map<ErrorType, std::string> ErrorNames = {
        {ErrorType::NegativePlasticStrain, "NegativePlasticStrain"},
        {ErrorType::OverflowError, "OverflowError"},
        {ErrorType::NoPlasticIncompressibility, "NoPlasticIncompressibility"},
        {ErrorType::FailedSolLinSystLNL, "FailedSolLinSystLNL"},
        {ErrorType::FailedDetermLineSearchParam, "FailedDetermLineSearchParam"},
        {ErrorType::NoConvergenceLNL, "NoConvergenceLNL"},
        {ErrorType::SingularJacobian, "SingularJacobian"},
        {ErrorType::FailedSolAnalytLinearization, "FailedSolAnalytLinearization"},
        {ErrorType::FailedLogEval, "FailedLogEval"},
        {ErrorType::FailedExpEval, "FailedExpEval"},
        {ErrorType::FailedRightCGInterp, "FailedRightCGInterp"},
    };

    //! class containing utilities for analyzing the material time integration:
    //! error types, number of line searches, ...
    //! (InelastDefgradTransvIsotropElastViscoplast)
    class TimIntAnalysisUtils
    {
     public:
      //! number of LNL steps for the current timestep evaluation (LNL)
      unsigned int eval_num_of_LNL_steps_ = 0;

      //! total number of steps
      unsigned int total_num_of_LNL_steps_ = 0;

      //! number of iterations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_iters_ = 0;

      //! total number of LNL iterations
      unsigned int total_num_of_iters_ = 0;

      //! number of repredictorizations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_repredict_ = 0;

      //! total number of LNL repredictorizations
      unsigned int total_num_of_repredict_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (LNL)
      unsigned int eval_num_of_pred_adapt_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation
      unsigned int total_num_of_pred_adapt_iters_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (LNL), in the specific case of repredictorization
      unsigned int eval_num_of_repredict_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation,
      //! in the specific case of repredictorization
      unsigned int total_num_of_repredict_iters_ = 0;

      //! number of line searches for the current timestep evaluation (LNL)
      unsigned int eval_num_of_line_search_ = 0;

      //! line search: the number of times the step size \f$ \alpha \f$ of
      //! the last iteration (Local Newton Loop) deviates from 1.0
      //! (currently evaluated time step)
      unsigned int eval_num_of_alpha_neq_1_last_iter = 0;

      //! line search: the number of times the step size \f$ \alpha \f$
      //! deviates from 1.0 in all iterations of the Local Newton Loop
      //! (currently evaluated time step)
      unsigned int eval_num_of_alpha_neq_1 = 0;

      //! total number of LNL line searches
      unsigned int total_num_of_line_search_ = 0;

      //! line search: total number of times the step size \f$ \alpha \f$ of
      //! the last iteration (Local Newton Loop) deviates from 1.0
      unsigned int total_num_of_alpha_neq_1_last_iter = 0;

      //! line search: the number of times the step size \f$ \alpha \f$
      //! deviates from 1.0 in all iterations of the Local Newton Loop
      //! (currently evaluated time step)
      unsigned int total_num_of_alpha_neq_1 = 0;

      //! number of iterations of the line searches for the current timestep evaluation (LNL)
      unsigned int eval_num_of_line_search_iters_ = 0;

      //! total number of line search iterations
      unsigned int total_num_of_line_search_iters_ = 0;

      //! number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!) for the current timestep evaluation
      unsigned int eval_num_of_first_iter_convergences = 0;

      //! total number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!)
      unsigned int total_num_of_first_iter_convergences = 0;

      //! timer for the current timestep evaluation, from the start of preevaluate to the end of
      //! update
      Teuchos::Time eval_teuchos_timer_{
          "InelasticDefgradTransvIsotropElastViscoplast::from_preevaluate_to_update"};

      //! timer for the time spent in the LNL
      Teuchos::Time eval_teuchos_timer_LNL_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the LNL"};

      //! timer for the time spent adapting the predictor
      Teuchos::Time eval_teuchos_timer_pred_adapt_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the predictor adaptation"};

      //! timer for the time spent adapting the predictor in the
      //! specific case of repredictorization
      Teuchos::Time eval_teuchos_timer_repredict_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the predictor adaptation "
          "(repredictorization)"};

      //! timer for the time spent in the line search
      Teuchos::Time eval_teuchos_timer_line_search_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the line search"};


      //! evaluation time
      double eval_time_;

      //! total time
      double total_time_;

      //! evaluation time spent in the LNL (current
      //! time step)
      double eval_time_LNL_;

      //! total time spent in the LNL
      double total_time_LNL_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step)
      double eval_time_pred_adapt_;

      //! total time spent in the predictor adaptation
      double total_time_pred_adapt_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step), in the specific case of repredictorization
      double eval_time_repredict_;

      //! total time spent in the predictor adaptation, in the specific
      //! case of repredictorization
      double total_time_repredict_;

      //! evaluation time spent in the line search (current
      //! time step)
      double eval_time_line_search_;

      //! total time spent in the line search
      double total_time_line_search_;

      //! error map of the current timestep evaluation, from the first preevaluate of this time step
      //! to the first preevaluate of the next
      std::map<ErrorType, unsigned int> eval_error_map_ = {
          {ErrorType::NegativePlasticStrain, 0},
          {ErrorType::OverflowError, 0},
          {ErrorType::NoPlasticIncompressibility, 0},
          {ErrorType::FailedSolLinSystLNL, 0},
          {ErrorType::FailedDetermLineSearchParam, 0},
          {ErrorType::NoConvergenceLNL, 0},
          {ErrorType::SingularJacobian, 0},
          {ErrorType::FailedSolAnalytLinearization, 0},
          {ErrorType::FailedLogEval, 0},
          {ErrorType::FailedExpEval, 0},
          {ErrorType::FailedRightCGInterp, 0},
      };

      //! error map of the total evaluation
      std::map<ErrorType, unsigned int> total_error_map_ = {
          {ErrorType::NegativePlasticStrain, 0},
          {ErrorType::OverflowError, 0},
          {ErrorType::NoPlasticIncompressibility, 0},
          {ErrorType::FailedSolLinSystLNL, 0},
          {ErrorType::FailedDetermLineSearchParam, 0},
          {ErrorType::NoConvergenceLNL, 0},
          {ErrorType::SingularJacobian, 0},
          {ErrorType::FailedSolAnalytLinearization, 0},
          {ErrorType::FailedLogEval, 0},
          {ErrorType::FailedExpEval, 0},
          {ErrorType::FailedRightCGInterp, 0},
      };

      //! runtime csv writer
      std::optional<Core::IO::RuntimeCsvWriter> csv_writer_;

      //! simulation time instant and time step
      double sim_time_ = 0.0;
      int sim_timestep_ = 0.0;

      //! was the pre_evaluate method of the first element called?
      bool pre_eval_called_ = false;

      //! how often was the update method called? (max. num_of_global_elements if one processor is
      //! considered)
      int num_update_calls_ = 0;

      //! reset method
      void reset()
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
            {ErrorType::NegativePlasticStrain, 0},
            {ErrorType::OverflowError, 0},
            {ErrorType::NoPlasticIncompressibility, 0},
            {ErrorType::FailedSolLinSystLNL, 0},
            {ErrorType::FailedDetermLineSearchParam, 0},
            {ErrorType::NoConvergenceLNL, 0},
            {ErrorType::SingularJacobian, 0},
            {ErrorType::FailedSolAnalytLinearization, 0},
            {ErrorType::FailedLogEval, 0},
            {ErrorType::FailedExpEval, 0},
            {ErrorType::FailedRightCGInterp, 0},
        };
        eval_num_of_alpha_neq_1 = 0;
        eval_num_of_alpha_neq_1_last_iter = 0;
        eval_num_of_first_iter_convergences = 0;
      }

      //! initialize csv_writer
      void init_csv_writer()
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
        csv_writer_.emplace(
            my_rank, *Global::Problem::instance()->output_control_file(), "timint_output");
        csv_writer_->register_data_vector("Eval. steps (LNL)", 1, 16);
        csv_writer_->register_data_vector("Eval. iterations (LNL)", 1, 16);
        csv_writer_->register_data_vector("Eval. repredictorizations (LNL)", 1, 16);
        csv_writer_->register_data_vector("Eval. iterations (predictor adaptation)", 1, 16);
        csv_writer_->register_data_vector("Eval. iterations (repredictorization)", 1, 16);
        csv_writer_->register_data_vector("Eval. line searches (LNL)", 1, 16);
        csv_writer_->register_data_vector("Eval. iterations (line search)", 1, 16);
        csv_writer_->register_data_vector("Eval. # of times: alpha neq 1 (all LNL iters)", 1, 16);
        csv_writer_->register_data_vector("Eval. # of times: alpha neq 1 (last LNL iter)", 1, 16);
        csv_writer_->register_data_vector("Eval. # of first LNL iter. convergences", 1, 16);
        csv_writer_->register_data_vector("Eval. time (full: preevaluate -> update)", 1, 16);
        csv_writer_->register_data_vector("Eval. time (LNL)", 1, 16);
        csv_writer_->register_data_vector("Eval. time (predictor adaptation)", 1, 16);
        csv_writer_->register_data_vector("Eval. time (repredictorization)", 1, 16);
        csv_writer_->register_data_vector("Eval. time (line search)", 1, 16);
        csv_writer_->register_data_vector("Total steps (LNL)", 1, 16);
        csv_writer_->register_data_vector("Total iterations (LNL)", 1, 16);
        csv_writer_->register_data_vector("Total repredictorizations (LNL)", 1, 16);
        csv_writer_->register_data_vector("Total iterations (predictor adaptation)", 1, 16);
        csv_writer_->register_data_vector("Total iterations (repredictorization)", 1, 16);
        csv_writer_->register_data_vector("Total line searches (LNL)", 1, 16);
        csv_writer_->register_data_vector("Total iterations (line search)", 1, 16);
        csv_writer_->register_data_vector("Total # of times: alpha neq 1 (all LNL iters)", 1, 16);
        csv_writer_->register_data_vector("Total # of times: alpha neq 1 (last LNL iter)", 1, 16);
        csv_writer_->register_data_vector("Total # of first LNL iter. convergences", 1, 16);
        csv_writer_->register_data_vector("Total time (full: preevaluate -> update)", 1, 16);
        csv_writer_->register_data_vector("Total time (LNL)", 1, 16);
        csv_writer_->register_data_vector("Total time (predictor adaptation)", 1, 16);
        csv_writer_->register_data_vector("Total time (repredictorization)", 1, 16);
        csv_writer_->register_data_vector("Total time (line search)", 1, 16);
        for (const auto& [key, value] : ErrorNames)
        {
          csv_writer_->register_data_vector(
              "Eval. Error " + std::to_string(static_cast<int>(key)) + ": " + value, 1, 16);
          csv_writer_->register_data_vector(
              "Total Error " + std::to_string(static_cast<int>(key)) + ": " + value, 1, 16);
        }
      }

      //! update total values
      void update_total()
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

      //! write to csv after each timestep
      void write_to_csv()
      {
        // output data
        std::map<std::string, std::vector<double>> output_data;
        output_data["Eval. steps (LNL)"] = {static_cast<double>(eval_num_of_LNL_steps_)};
        output_data["Total steps (LNL)"] = {static_cast<double>(total_num_of_LNL_steps_)};
        output_data["Eval. iterations (LNL)"] = {static_cast<double>(eval_num_of_iters_)};
        output_data["Total iterations (LNL)"] = {static_cast<double>(total_num_of_iters_)};
        output_data["Eval. repredictorizations (LNL)"] = {
            static_cast<double>(eval_num_of_repredict_)};
        output_data["Total repredictorizations (LNL)"] = {
            static_cast<double>(total_num_of_repredict_)};
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
        output_data["Total time (full: preevaluate -> update)"] = {
            static_cast<double>(total_time_)};
        output_data["Eval. time (LNL)"] = {static_cast<double>(eval_time_LNL_)};
        output_data["Total time (LNL)"] = {static_cast<double>(total_time_LNL_)};
        output_data["Eval. time (predictor adaptation)"] = {
            static_cast<double>(eval_time_pred_adapt_)};
        output_data["Total time (predictor adaptation)"] = {
            static_cast<double>(total_time_pred_adapt_)};
        output_data["Eval. time (repredictorization)"] = {
            static_cast<double>(eval_time_repredict_)};
        output_data["Total time (repredictorization)"] = {
            static_cast<double>(total_time_repredict_)};
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


        for (const auto& [key, value] : ErrorNames)
        {
          output_data["Eval. Error " + std::to_string(static_cast<int>(key)) + ": " + value] = {
              static_cast<double>(eval_error_map_[key])};
          output_data["Total Error " + std::to_string(static_cast<int>(key)) + ": " + value] = {
              static_cast<double>(total_error_map_[key])};
        }

        // write output data to csv
        csv_writer_->write_data_to_file(sim_time_, sim_timestep_, output_data);
      }

      //! output routine routine of the csv writer in the case of an error
      //! during the Local Newton Loop routine
      void output_error_local_newton_loop(unsigned int step_counter)
      {  // add current number of steps
        eval_num_of_LNL_steps_ += step_counter;

        // stop (already started!) LNL timer
        eval_time_LNL_ += eval_teuchos_timer_LNL_.stop();

        // output routine
        eval_time_ = eval_teuchos_timer_.stop();
        update_total();
        write_to_csv();
      }
    };

    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      PlasticStrainRateOnly,  ///< return in evaluate_state_quantities once the plastic strain rate
                              ///< has been evaluated
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives method)
      PlasticStrainRateDerivsOnly,  ///< return in evaluate_state_quantity_derivatives once the
                                    ///< derivatives of the plastic strain rate have been evaluated
    };

    /// make sure StateQuantityDerivEvalType is stream-insertable
    inline std::ostream& operator<<(
        std::ostream& stream, const StateQuantityDerivEvalType& state_quant_deriv_eval_type)
    {
      stream << magic_enum::enum_name(state_quant_deriv_eval_type);
      return stream;
    }

    /// defines
    // flag for debug output (viscoplastic material) related to
    // time integration
    // #define DEBUGVPLAST_TIMINT

    // flag for debug output (viscoplastic material) related to the
    // inverse inelastic defgrad computation; less detailed than
    // DEBUGVPLAST_TIMINT
    // #define DEBUGVPLAST_INELDEFGRAD

    // flag for debug output (viscoplastic material) related to the
    // stiffness contribution
    // #define DEBUGVPLAST_LINEARIZATION

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif