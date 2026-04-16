// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <format>
#include <map>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// declare numerical tolerance to be used in the verification of (numerically) zero plastic
    /// strain increments
    constexpr double zero_plastic_strain_increment{1.0e-14};

    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. Reinterpolation,
    /// substepping, line search) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      no_errors,                ///< no errors
      negative_plastic_strain,  ///< negative plastic strain which does not allow for evaluations
                                ///< inside the viscoplasticity laws
      overflow_error,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                       ///< (and \f$ \boldsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \boldsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,  ///< the material has no flow resistance anymore, such that the
                           ///< evaluations model non-physical phenomena
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
      no_convergence_local_newton,  ///< the Local Newton Loop did not converge for the given loop
                                    ///< settings
      singular_jacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                          ///< analytical evaluation of the linearization
      failed_solution_analytic_linearization,  ///< solution of the linear system in the analytical
                                               ///< linearization failed
      failed_computation_flow_resistance,  ///< failed in the computation of the flow resistance via
                                           ///< time integration of the hardening-rate equation
                                           ///(e.g., when using the Anand law)
      failed_computation_flow_resistance_derivs,  ///< failed in the computation of the flow
                                                  ///< resistance derivatives (e.g., when using the
                                                  ///< Anand law)
      failed_matrix_log_evaluation,   ///< failed evaluation of the matrix logarithm or its
                                      ///< derivative
      failed_matrix_exp_evaluation,   ///< failed evaluation of the matrix exponential or its
                                      ///< derivative
      failed_right_cg_interpolation,  ///< failed interpolation of the right Cauchy-Green tensor
      under_yield_surface  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                           ///< stress is smaller than the yield stress
    };


    /// enum class for evaluation management actions in the iterations of the
    /// Local Newton loop
    enum class EvaluationAction
    {
      continue_current_iteration,    ///< continue current iteration
      continue_with_next_iteration,  ///< go to next iteration after performing certain reset steps
      exit_with_error,               ///< exit Local Newton Loop with the set error status
    };

    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotropic,         ///< isotropic material behavior
      transv_isotropic,  ///< transversely isotropic material behavior
    };

    /// enum class for time integration types (Local Newton integration)
    enum class TimIntType
    {
      standard,     ///< standard time integration,
      logarithmic,  ///< time integration with logarithmically transformed residual equation for the
                    ///< evolution of the plastic deformation gradient
    };

    /// enum class for material linearization types
    enum class LinearizationType
    {
      analytic,  ///< analytical linearization involving the solution of a linear system of
                 ///< equations,
      perturbation_based,  ///< linearization based on perturbing the current state
    };

    //! matrix exponential and logarithm evaluation utilities
    struct MatrixExpLogUtils
    {
      //! Pade approximation order (to be used consistently: the
      //! derivative of the matrix functions should use the same Pade
      //! order as the evaluation of the matrix functions)
      unsigned int pade_order = 16;  // by default we set the highest order currently implemented
    };

    //! struct containing time step settings and time trackers
    struct TimeStepTracker
    {
      //! time step length
      double dt;
      //! currently computed time instant \f$ t_{n+1} \f$
      double tnp;
      //! minimum substep length
      double min_dt;
    };


    //! struct containing quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively). The quantities are tracked at all Gauss points, in order to
    //! update them simultaneously during the update method call
    struct TimeStepQuantities
    {
      //! right Cauchy-Green deformation tensor at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_rightCG;

      //! inverse plastic deformation gradient at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse;

      //! (equivalent) plastic strain at the last time step (for all Gauss points)
      std::vector<double> last_plastic_strain;

      //! equivalent stress at the previous time instant (for all Gauss points)
      std::vector<double> last_equiv_stress;


      //! last (reduced) deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_defgrad;

      //! temporary variable, for which we store the right Cauchy-Green deformation tensor at each
      //! evaluation (used in order to update last_rightCG_ once outer NR converges) (for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_rightCG;

      //! current (reduced) deformation gradient: used to check whether the inverse inelastic
      //! deformation gradient has already been evaluated (to improve the computation performance)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_defgrad;

      //! current inverse plastic deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrad_inverse;

      //! current plastic strain (for all Gauss points)
      std::vector<double> current_plastic_strain;

      //! current equivalent stress (for all Gauss points)
      std::vector<double> current_equiv_stress;


      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain;

      /*!
       * @brief Set meaningful initial values. Done first for one single Gauss point (extended later
       * on using the resizing function).
       *
       */
      void init();

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @note The value saved within the first item is taken for all items during resizing. We only
       * enable resizing if the current sizes of the involved vectors are 1
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Perform pre-evaluation tasks at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void pre_evaluate(const unsigned int gp);

      //!  Update values between time steps: last <- current
      void update();

      //! Pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! Unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called{false};
    };



    //! struct: constant non-material tensors, such as different
    //! identity tensors
    struct ConstNonMatTensors
    {
      static const ConstNonMatTensors& instance()
      {
        static ConstNonMatTensors instance;
        return instance;
      }

      //! constructor
      ConstNonMatTensors();
      //! second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
      //! second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 1> id6x1{Core::LinAlg::Initialization::zero};
      //! symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
      Core::LinAlg::Matrix<6, 6> id4_6x6{Core::LinAlg::Initialization::zero};
      //! deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
      //! \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 6> dev_op{Core::LinAlg::Initialization::zero};
      //! identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
      Core::LinAlg::Matrix<9, 9> id4_9x9{Core::LinAlg::Initialization::zero};
      //! second-order 10x10 identity tensor in matrix form
      Core::LinAlg::Matrix<10, 10> id10x10{Core::LinAlg::Initialization::zero};
    };



    //! struct containing constant tensors which depend on the constant fiber direction \f$
    //! \boldsymbol{m} \f$
    struct ConstMatTensors
    {
      //! \f$ \boldsymbol{I} + \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_mm;
      //! \f$ \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> mm{Core::LinAlg::Initialization::zero};
      //! deviatoric part \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //! \right)_\text{dev}\f$
      Core::LinAlg::Matrix<3, 3> mm_dev{Core::LinAlg::Initialization::zero};
      //! \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right) \otimes \left( \boldsymbol{m}
      //! \otimes \boldsymbol{m} \right) \f$ (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right)_\text{dev} \otimes \left(
      //!  \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dev_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \boldsymbol{I} \otimes \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> id_dyad_mm;

      //! set tensors for a given fiber direction \f$ \boldsymbol{m} \f$
      void set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m);
    };

    //! class with local substepping utilities
    class LocalSubsteppingUtils
    {
     public:
      LocalSubsteppingUtils() = delete;
      //! Constructor (calling reset under the hood)
      explicit LocalSubsteppingUtils(double dt) { reset(dt); }

      //! reset routine: set a single substep of a given size dt
      void reset(const double dt);

      //! verify whether the substepping routine has reached its end
      [[nodiscard]] bool end_substepping() const
      {
        return substep_counter_ > total_num_of_substeps_;
      };

      //! increment substep
      void increment_substep();

      //! halve current substep and update relevant quantities
      void halve_substep();

      //! get substep size
      [[nodiscard]] double get_substep_size() const { return curr_dt_; }

      //! retrieve the normalized time parameter at the next time instant during substepping, i.e.,
      //! \f$ \frac{\left(t_{m} + \Delta t_{m}\right)}{\Delta t} \f$, where \f$t_{m}\f$ denotes the
      //! previously converged time instant, \f$\Delta t_{m}\f$ denotes the current substep size,
      //! and \f$\Delta t\f$ specifies the global timestep
      [[nodiscard]] double get_normalized_next_time_param(const double dt) const
      {
        return (t_ + curr_dt_) / dt;
      }

      //! get counter for the current number of time step halving procedures
      [[nodiscard]] unsigned int get_halving_counter() const { return time_step_halving_counter_; }

      //! get substepping info as string
      [[nodiscard]] std::string get_info() const
      {
        std::string out;
        out += "Substepping info: \n";
        out += std::format(
            "t: {}, substep_counter: {}, curr_dt: {}, time_step_halving_counter: {}, "
            "total_num_of_substeps: {} \n",
            t_, substep_counter_, curr_dt_, time_step_halving_counter_, total_num_of_substeps_);
        return out;
      };

     private:
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t_;
      //! counter of evaluated substeps
      unsigned int substep_counter_;
      //! current substep size
      double curr_dt_;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter_;
      //!  total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always directly proportional to time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps_;
    };

    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      plastic_strain_rate_only,  ///< return in evaluate_state_quantities once the plastic strain
                                 ///< rate has been evaluated
      equiv_stress_only,         ///< return in evaluate_state_quantities once the
                                 ///< equivalent stress has been evaluated
    };



    //! struct containing quantities computed from a given elasticity/plasticity state;
    //! given: current right Cauchy-Green deformation tensor, inelastic deformation gradient and
    //! plastic strain at the previous time instant
    struct StateQuantities
    {
      // ----- current state quantities (for the evaluated Gauss points) ----- //

      //! elastic right Cauchy-Green deformation tensor
      Core::LinAlg::Matrix<3, 3> curr_CeM{Core::LinAlg::Initialization::zero};

      //! isotropic stress factors
      Core::LinAlg::Matrix<3, 1> curr_gamma{Core::LinAlg::Initialization::zero};

      //! isotropic constitutive tensor factors
      Core::LinAlg::Matrix<8, 1> curr_delta{Core::LinAlg::Initialization::zero};

      //! elastic 2nd PK stress tensors (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<3, 3> curr_SeM{Core::LinAlg::Initialization::zero};

      //! elastic stiffness tensor (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<6, 6> curr_dSedCe{Core::LinAlg::Initialization::zero};

      //! deviatoric, symmetric part of the Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Me_dev_sym_M{Core::LinAlg::Initialization::zero};

      //! equivalent tensile stress
      double curr_equiv_stress{0.0};

      //! equivalent plastic strain rate
      double curr_equiv_plastic_strain_rate{0.0};

      //! plastic flow direction tensor
      Core::LinAlg::Matrix<3, 3> curr_NpM{Core::LinAlg::Initialization::zero};

      //! plastic stretching tensor
      Core::LinAlg::Matrix<3, 3> curr_dpM{Core::LinAlg::Initialization::zero};

      //! plastic velocity gradient tensor
      Core::LinAlg::Matrix<3, 3> curr_lpM{Core::LinAlg::Initialization::zero};

      //! plastic update tensor
      Core::LinAlg::Matrix<3, 3> curr_EpM{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityEvalType eval_type;
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives
                  ///< method)
      plastic_strain_rate_derivs_only,  ///< return in evaluate_state_quantity_derivatives once the
                                        ///< derivatives of the plastic strain rate have been
                                        ///< evaluated
      equiv_stress_derivs_only,  ///< return in evaluate_state_quantities once the derivatives of
                                 ///< the equivalent stress has been evaluated
    };



    //! struct containing specific derivatives of quantities computed from a given
    //! elasticity/plasticity state; given: current right Cauchy-Green deformation tensor,
    //! inelastic deformation gradient and plastic strain at the previous time instant
    struct StateQuantityDerivatives
    {
      // ----- current state variable derivatives (for the evaluated Gauss points)----- //

      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the inverse
      //! inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dCediFin{Core::LinAlg::Initialization::zero};
      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dCedC{Core::LinAlg::Initialization::zero};

      //! derivatives of the equivalent tensile stress w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<1, 9> curr_dequiv_stress_diFin{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress form)
      Core::LinAlg::Matrix<1, 6> curr_dequiv_stress_dC{Core::LinAlg::Initialization::zero};

      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMe_dev_sym_diFin{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMe_dev_sym_dC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityDerivEvalType eval_type;
    };

    //! struct containing specific derivatives of the scalar plastic strain rate used in
    //! InelasticDefgradTransvIsotropElastViscoplast
    struct PlasticStrainRateDerivs
    {
      //! derivative with respect to the equivalent stress
      double deriv_equiv_stress;


      //! derivative with respect to the plastic strain
      double deriv_plastic_strain;
    };


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonConvCheck
    {
      residual,         ///< verify convergence based on the absolute value of the Local Newton
                        ///< residual 2-norm
      increment_ratio,  ///< verify convergence based on the ratio of solution increment to current
                        ///< solution: \f$ \frac{\left| \Delta \boldsymbol{s}^{l+1} \right|}{\left|
                        ///< \boldsymbol{s}^{l} \right|}  \f$
      residual_and_increment_ratio,  ///< verify convergence based on both the absolute Local Newton
                                     ///< residual and the ratio of solution increment to current
                                     ///< solution
    };


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonDiverCont
    {
      stop,          ///< stop the simulation entirely
      continue_sim,  ///<  continue the simulation, and display warning in regards to the current
                     ///<  state within the Local Newton Loop
      continue_sim_with_safeguard  ///< continue the simulation only if the convergence tolerances
                                   ///< are not exceeded excessively, as specified with specific
                                   ///< exceedance factors for the tolerances
    };

    /// enum: quantities relevant for convergence checking within the Local Newton Loop
    struct LocalNewtonConvQuantities
    {
      //! residual 2-norm
      double residual_norm;

      //! ratio of solution increment to current solution: \f$ \frac{\left| \Delta
      //! \boldsymbol{s}^{l+1} \right|}{\left| \boldsymbol{s}^{l} \right|}  \f$
      double increment_norm;
    };



    //! struct containing parameter specifications for the Local Newton loop
    struct LocalNewtonParams
    {
      //! convergence tolerance: absolute residual value
      const double res_tol;

      //! convergence tolerance: ratio of solution increment to current solution
      const double incr_tol;

      //! convergence check strategy
      const LocalNewtonConvCheck conv_check;

      //! strategy for dealing with divergence
      const LocalNewtonDiverCont diver_cont;

      //! maximum number of local iterations
      const unsigned int max_iter;

      //! maximum exceedance factor for the residual tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_res_tol;

      //! maximum exceedance factor for the solution increment tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_incr_tol;
    };

    //! class for managing the Local Newton loop, containing the utilized parameters and iteration
    //! data
    class LocalNewtonManager
    {
     public:
      LocalNewtonManager() = delete;
      /*!
       * @brief Constructor
       *
       * @param[in] lnl_params Local Newton parameters
       *
       */
      explicit LocalNewtonManager(const LocalNewtonParams& lnl_params);

      /// getter for Local Newton parameters
      [[nodiscard]] LocalNewtonParams params() const { return params_; }

      /// getter for local iteration count
      [[nodiscard]] unsigned int iter() const { return iter_; }

      /// setter for local iteration count
      void set_iteration_count(const unsigned int iter) { iter_ = iter; }

      /// getter for total number of local iterations evaluated in this time step (vector over all
      /// Gauss points)
      [[nodiscard]] const std::vector<unsigned int>& curr_num_iters() const
      {
        return curr_num_iters_;
      }

      /// increment iteration count by 1
      void increment_iteration_count() { iter_++; }

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Routine to be run after the Local Newton-Raphson at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void update_after_local_newton(const unsigned int gp);

      //! reset method
      void reset();

      //! pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

     private:
      //! Local Newton parameters
      const LocalNewtonParams params_;

      //! current local iteration
      unsigned int iter_;

      //! total number of local iterations for the current timestep; vector of Gauss point values
      std::vector<unsigned int> curr_num_iters_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};
    };

    //! class: manager for the adaptive estimate interpolation proposed in
    //! Ana, Schmidt, Wall: Adaptive Estimate Interpolation: Accelerating Local Newton-Raphson
    //! Schemes in Computational (Visco)Plasticity, Preprint
    class AdaptiveEstimateInterpolationManager
    {
     public:
      //! plastic predictor: strategies for choosing the elastic stretch eigenvalues \f$
      //! \boldsymbol{\Lambda} \f$
      enum class PlasticPredictorElasticStretchEigenvalType
      {
        scale_previous,  ///< the elastic stretch eigenvalues from the previous time instant are
                         ///< scaled with the deformation gradient determinant to maintain plastic
                         ///< incompressibility
        scale_unit,      ///< the unit tensor is scaled with the deformation gradient determinant to
                         ///< maintain plastic incompressibility
      };

      //! plastic predictor: strategies for choosing elastic stretch eigenvectors \f$ \boldsymbol{Q}
      //! \f$
      enum class PlasticPredictorElasticStretchEigenvectRotType
      {
        from_elastic_predictor,  ///< eigenvector rotation is taken from the trial elastic state /
                                 ///< the elastic predictor (for isotropic materials, this
                                 ///< assumption is consistent with the solution of the Local
                                 ///< Newton)
      };


      //! plastic predictor: strategies for choosing the elastic rotation \f$ \boldsymbol{R} \f$
      enum class PlasticPredictorRotationType
      {
        from_elastic_predictor,  ///< elastic rotation = trial elastic rotation (elastic
                                 ///< predictor)
                                 /// within the plastic predictor (for isotropic models this is
                                 /// generally consistent with the solution of the Local Newton)
      };


      /*!
       * @brief Constructor
       *
       * @param[in] max_num_reestimations Maximum number of allowed re-estimations before the local
       * integration is deemed infeasible
       * @param[in] elastic_stretch_eigenval_type Strategy for choosing the elastic stretch
       * eigenvalues of the plastic predictor
       * @param[in] elastic_stretch_eigenvect_rot_type Strategy for choosing the elastic stretch
       * eigenvectors of the plastic predictor
       * @param[in] elastic_rot_type Strategy for choosing the elastic rotation
       *  @param[in] min_interp_interval Minimum interpolation interval upper - lower (2-norm in
       * interpolation space) which leads to an infeasible estimate interpolation
       */
      AdaptiveEstimateInterpolationManager(const unsigned int max_num_reestimations,
          const PlasticPredictorElasticStretchEigenvalType elastic_stretch_eigenval_type,
          const PlasticPredictorElasticStretchEigenvectRotType elastic_stretch_eigenvect_rot_type,
          const PlasticPredictorRotationType rot_type, const double min_interp_interval);



      //! starting point type
      enum class StartingPointType
      {
        user_set,                  ///< User-set constant factor
        last_interpolation_point,  ///< Takes the interpolation point from the last global iteration
                                   ///< of the previous timestep, which led to a valid initial
                                   ///< guess, as the starting point for the interpolation within
                                   ///< the current timestep
        optimal_equiv_stress       ///< Computes the interpolation factor based on the equivalent
        ///< stress for the previous timestep with respect to the previous
        ///< elastic and plastic predictors. All (generally different) interpolation point
        ///< components are set to this one factor.
      };


      //! struct: decomposition of elastic deformation gradient using combined spectral-polar
      //! decomposition as in Satheesh et al. 2023 (10.1002/nme.7373)
      struct ElasticDefgradDecomposition
      {
        //! elastic predictor: full, non-decomposed specific deformation
        // gradient considered
        Core::LinAlg::Matrix<3, 3> specific_defgrad_elast_pred_;
        //! plastic predictor: full, non-decomposed specific deformation
        // gradient considered
        Core::LinAlg::Matrix<3, 3> specific_defgrad_plast_pred_;
        //! elastic predictor: eigenvalues \f$ \lambda_{\mathrm{elast}, i} \f$
        std::array<double, 3> lambda_elast_pred_;
        //! plastic predictor: eigenvalues \f$ \lambda_{\mathrm{plast}, i} \f$
        std::array<double, 3> lambda_plast_pred_;
        //! elastic predictor: logarithm of eigenvalues \f$ \log(\lambda_{\mathrm{elast}, i})
        //! \f$
        std::array<double, 3> log_lambda_elast_pred_;
        //! plastic predictor: logarithm eigenvalues \f$ \log(\lambda_{\mathrm{plast}, i}) \f$
        std::array<double, 3> log_lambda_plast_pred_;
        //! elastic predictor: eigenvector rotation matrix \f$ \mathbf{Q}_{\mathrm{elast}} \f$
        Core::LinAlg::Matrix<3, 3> Qmat_elast_pred_;
        //! plastic predictor: eigenvector rotation matrix \f$ \mathbf{Q}_{\mathrm{plast}} \f$
        Core::LinAlg::Matrix<3, 3> Qmat_plast_pred_;
        //! plastic predictor: relative eigenvector rotation matrix \f$
        //! \mathbf{Q}_{\mathrm{plast, rel}} \f$ with respect to the eigenvector rotation of the
        //! elastic predictor
        Core::LinAlg::Matrix<3, 3> Qmat_plast_pred_rel_;
        //! plastic predictor: relative eigenvector rotation vector \f$
        //! \mathbf{q}_{\mathrm{plast, rel}} \f$ with respect to the eigenvector rotation of the
        //! elastic predictor
        Core::LinAlg::Matrix<3, 1> Qvec_plast_pred_rel_;
        //! elastic predictor: rotation matrix \f$ \mathbf{R}_{\mathrm{elast}} \f$
        Core::LinAlg::Matrix<3, 3> Rmat_elast_pred_;
        //! plastic predictor: rotation matrix \f$ \mathbf{R}_{\mathrm{plast}} \f$
        Core::LinAlg::Matrix<3, 3> Rmat_plast_pred_;
        //! plastic predictor: relative rotation matrix \f$ \mathbf{R}_{\mathrm{plast, rel}} \f$
        //! with respect to rotation of the elastic predictor
        Core::LinAlg::Matrix<3, 3> Rmat_plast_pred_rel_;
        //! plastic predictor: relative rotation vector \f$ \mathbf{r}_{\mathrm{plast, rel}} \f$
        //! with respect to rotation of the elastic predictor
        Core::LinAlg::Matrix<3, 1> Rvec_plast_pred_rel_;
        //! elastic predictor: spectral pairs containing the eigenvectors and eigenvalues
        std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred_;
        //! plastic predictor: spectral pairs containing the eigenvectors and eigenvalues
        std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_plast_pred_;

        /**
         * @brief Constructor.
         *
         * @note The specific deformation gradient within the elastic predictor
         * is considered as the reference when aligning eigenpairs, and
         * determining the relative rotation.
         *
         * @param[in] defgrad_elast_pred Deformation gradient (
         * elastic, inverse plastic, ...) within the elastic predictor
         * @param[in] defgrad_plast_pred Deformation gradient (elastic,
         * inverse plastic, ...) within the plastic predictor
         * @param[in] spectral_pairs_ref Given reference / elastic predictor spectral pairs to
         * be used for determining relative rotation and aligning eigenpairs for
         * the plastic predictor. Only to be used in special cases, such as when
         * we
         * determine optimal interpolation factors to ensure consistent
         * reference spectral pairs for interpolation and solution.
         */
        PredictorDefgradDecomposition(const Core::LinAlg::Matrix<3, 3>& defgrad_elast_pred,
            const Core::LinAlg::Matrix<3, 3>& defgrad_plast_pred,
            std::optional<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>
                spectral_pairs_ref = std::nullopt);

        //! pack method
        void pack(Core::Communication::PackBuffer& data) const
        {
          Core::Communication::add_to_pack(data, specific_defgrad_elast_pred_);
          Core::Communication::add_to_pack(data, specific_defgrad_plast_pred_);
          Core::Communication::add_to_pack(data, lambda_elast_pred_);
          Core::Communication::add_to_pack(data, lambda_plast_pred_);
          Core::Communication::add_to_pack(data, log_lambda_elast_pred_);
          Core::Communication::add_to_pack(data, log_lambda_plast_pred_);
          Core::Communication::add_to_pack(data, Qmat_elast_pred_);
          Core::Communication::add_to_pack(data, Qmat_plast_pred_);
          Core::Communication::add_to_pack(data, Qmat_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Qvec_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Rmat_elast_pred_);
          Core::Communication::add_to_pack(data, Rmat_plast_pred_);
          Core::Communication::add_to_pack(data, Rmat_plast_pred_rel_);
          Core::Communication::add_to_pack(data, Rvec_plast_pred_rel_);
          Core::Communication::add_to_pack(data, specific_defgrad_elast_pred_);
          Core::Communication::add_to_pack(data, specific_defgrad_plast_pred_);
        }

        //! unpack method
        void unpack(Core::Communication::UnpackBuffer& buffer)
        {
          Core::Communication::extract_from_pack(buffer, specific_defgrad_elast_pred_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_plast_pred_);
          Core::Communication::extract_from_pack(buffer, lambda_elast_pred_);
          Core::Communication::extract_from_pack(buffer, lambda_plast_pred_);
          Core::Communication::extract_from_pack(buffer, log_lambda_elast_pred_);
          Core::Communication::extract_from_pack(buffer, log_lambda_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_elast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Qmat_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Qvec_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Rmat_elast_pred_);
          Core::Communication::extract_from_pack(buffer, Rmat_plast_pred_);
          Core::Communication::extract_from_pack(buffer, Rmat_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, Rvec_plast_pred_rel_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_elast_pred_);
          Core::Communication::extract_from_pack(buffer, specific_defgrad_plast_pred_);
        }

        //! print method
        void print(std::ostream& os) const
        {
          std::cout << "PredictorDefgradDecomposition: \n";
          std::cout << "elastic - plastic: " << std::endl;
          std::cout << "lambda: [" << lambda_elast_pred_[0] << ", " << lambda_elast_pred_[1] << ", "
                    << lambda_elast_pred_[2] << "] - [" << lambda_plast_pred_[0] << ", "
                    << lambda_plast_pred_[1] << ", " << lambda_plast_pred_[2] << "]\n";
          std::cout << "Qvec_rel: [" << "0" << ", " << "0" << ", "
                    << "0" << "] - [" << Qvec_plast_pred_rel_(0) << ", " << Qvec_plast_pred_rel_(1)
                    << ", " << Qvec_plast_pred_rel_(2) << "]\n";
        }
      };

      //! struct: specified point in interpolation space used to interpolate the
      //! inverse plastic deformation gradient (or the elastic deformation
      //! gradient, depending on the user specification)
      struct InterpolationPoint
      {
        //! interpolation parameters for the elastic stretch eigenvalues \f$ \boldsymbol{\Lambda}
        //! \f$
        std::array<double, 3> xi_lambda;
        //! interpolation parameters for the rotation vector associated with the relative elastic
        //! stretch eigenvector rotation \f$ \boldsymbol{Q}_{rel} \f$
        std::array<double, 3> xi_rel_eigenvect_rot;
        //! interpolation parameters for the rotation vector associated with the relative elastic
        //! stretch rotation \f$ \boldsymbol{R}_{rel} \f$
        std::array<double, 3> xi_rel_rot;

        //! calculate 2-norm
        double norm() const
        {
          const double squared_lambda = xi_lambda[0] * xi_lambda[0] + xi_lambda[1] * xi_lambda[1] +
                                        xi_lambda[2] * xi_lambda[2];

          const double squared_rel_eigenvect_rot =
              xi_rel_eigenvect_rot[0] * xi_rel_eigenvect_rot[0] +
              xi_rel_eigenvect_rot[1] * xi_rel_eigenvect_rot[1] +
              xi_rel_eigenvect_rot[2] * xi_rel_eigenvect_rot[2];

          const double squared_rel_rot = xi_rel_rot[0] * xi_rel_rot[0] +
                                         xi_rel_rot[1] * xi_rel_rot[1] +
                                         xi_rel_rot[2] * xi_rel_rot[2];


          return std::sqrt(squared_lambda + squared_rel_eigenvect_rot + squared_rel_rot);
        }

        //! print method
        void print(std::ostream& os) const
        {
          os << std::format(
              "Interpolation point: \n lambda: [{}, {}, {}] \n rel_eigenvect_rot: [{}, {}, {}] \n "
              "rel_rot: [{}, {}, {}] \n",
              xi_lambda[0], xi_lambda[1], xi_lambda[2], xi_rel_eigenvect_rot[0],
              xi_rel_eigenvect_rot[1], xi_rel_eigenvect_rot[2], xi_rel_rot[0], xi_rel_rot[1],
              xi_rel_rot[2]);
        }
      };

      //! enum class: shift direction for the interpolation
      enum class InterpolationShiftAction
      {
        shift_towards_elastic_pred,  ///< shift towards the elastic predictor (e.g., in case of
                                     ///< vanishing plastic strain increments)
        shift_towards_plastic_pred,  ///< shift towards the plastic predictor (e.g., in case of
                                     ///< overflow errors due to high overstresses)
      };

      //! get shift direction for errors encountered in the plastic predictor construction
      static inline InterpolationShiftAction get_plastic_pred_construction_shift_action(
          ErrorType error_type)
      {
        switch (error_type)
        {
          case ErrorType::overflow_error:
            return InterpolationShiftAction::shift_towards_plastic_pred;
            break;
          case ErrorType::under_yield_surface:
            return InterpolationShiftAction::shift_towards_elastic_pred;
            break;
          default:
            FOUR_C_THROW(
                "No action specified for error {} within the plastic predictor construction!",
                EnumTools::enum_name(error_type));
        }
      }

      //! get shift direction for errors encountered in the estimate interpolation procedure
      static inline InterpolationShiftAction get_estimate_interpolation_shift_action(
          ErrorType error_type)
      {
        switch (error_type)
        {
          case ErrorType::overflow_error:
          case ErrorType::failed_computation_flow_resistance:
          case ErrorType::failed_computation_flow_resistance_derivs:
          case ErrorType::failed_matrix_log_evaluation:
          case ErrorType::failed_matrix_exp_evaluation:
            return InterpolationShiftAction::shift_towards_plastic_pred;
            break;
          case ErrorType::under_yield_surface:
            return InterpolationShiftAction::shift_towards_elastic_pred;
            break;
          default:
            FOUR_C_THROW("No action specified for error {} within the estimate interpolation!",
                EnumTools::enum_name(error_type));
        }
      }

      //! resize method: set the correct number of Gauss Points to track the internal variables of
      //! the class
      void resize(const unsigned int num_gp);

      /*!
       * @brief Verify whether interpolation is still possible, based on the
       * set minimum interpolation interval, the set maximum number of interpolation
       * iterations, and the set maximum number of reinterpolations
       *
       */
      bool is_interpolation_possible(const unsigned int gp, const unsigned int num_interp_iters)
      {
        // check interpolation interval
        const double diff_bounds = add_interpolation_points(
            1.0, get_lower_bound_interp_point(gp), -1.0, get_upper_bound_interp_point(gp))
                                       .norm();
        bool check_min_interp_interval = (diff_bounds >= min_interp_interval_);

        // check number of interpolation iterations
        bool check_interp_iters = (num_interp_iters <= max_num_interp_iters_);

        // check number of re-estimations
        bool check_num_reestimations = (num_of_reestimations_ <= max_num_reestimations_);

        return check_min_interp_interval && check_interp_iters && check_num_reestimations;
      }

      /*!
       * @brief Pre-evaluation: setting element and Gauss point, and the reference quantities used
       * for estimate interpolation
       *
       * @param[in] gp Gauss point index
       * @param[in] ele_gid element index
       * @param[in] elastic_defgrad_elast_pred elastic deformation gradient within the elastic
       * predictor
       * @param[in] elastic_defgrad_plast_pred elastic deformation gradient within the plastic
       * predictor
       */
      void pre_evaluate(const unsigned int gp, const unsigned int ele_gid,
          const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_plast_pred);

      //! update method
      void update();

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      /*!
       * @brief Perform decompositions of inverse plastic | elastic deformation
       * gradient within the elastic and plastic predictors.
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_elast_pred inverse plastic deformation
       * gradient within the elastic predictor
       * @param[in] inv_plastic_defgrad_plast_pred inverse plastic deformation
       * gradient within the plastic predictor
       * @param[in] defgrad Current deformation gradient (current Local Newton
       * iteration of \f$ \left[ t_n, t_{n+1} \right] \f$)
       *
       */
      void perform_predictor_decomposition(const unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred,
          const Core::LinAlg::Matrix<3, 3>& defgrad);

      //! getter for the current interpolation point \f$ \xi \f$ at a specified Gauss point gp
      InterpolationPoint get_current_interp_point(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < current_interp_point_.size(),
            "Cannot retrieve current interpolation point for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, current_interp_point_.size());
        return current_interp_point_[gp];
      };
      //! setter for the current interpolation point \f$ \xi \f$ at a specified Gauss point gp
      void set_current_interp_point(const unsigned gp, const InterpolationPoint interp_point)
      {
        FOUR_C_ASSERT_ALWAYS(gp < current_interp_point_.size(),
            "Cannot set current interpolation point for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, current_interp_point_.size());
        current_interp_point_[gp] = interp_point;
      }

      //! getter for the lower interpolation bound \f$ \xi_{\text{E}} \f$ at a specified Gauss point
      //! gp
      InterpolationPoint get_lower_interp_bound(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < lower_interp_bound_.size(),
            "Cannot retrieve lower interpolation bound for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, lower_interp_bound_.size());
        return lower_interp_bound_[gp];
      };
      //! setter for the lower interpolation bound \f$ \xi_{\text{E}} \f$ at a specified Gauss point
      //! gp
      void set_lower_interp_bound(const unsigned gp, const InterpolationPoint interp_point)
      {
        FOUR_C_ASSERT_ALWAYS(gp < lower_interp_bound_.size(),
            "Cannot set lower interpolation bound for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, current_interp_point_.size());
        lower_interp_bound_[gp] = interp_point;
      }

      //! getter for the upper interpolation bound \f$ \xi_{\text{P}} \f$ at a specified Gauss point
      //! gp
      InterpolationPoint get_upper_interp_bound(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < upper_interp_bound_.size(),
            "Cannot retrieve upper interpolation bound for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, upper_interp_bound_.size());
        return upper_interp_bound_[gp];
      };
      //! setter for the upper interpolation bound \f$ \xi_{\text{P}} \f$ at a specified Gauss point
      //! gp
      void set_upper_interp_bound(const unsigned gp, const InterpolationPoint interp_point)
      {
        FOUR_C_ASSERT_ALWAYS(gp < upper_interp_bound_.size(),
            "Cannot set upper interpolation bound for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, upper_interp_bound_.size());
        upper_interp_bound_[gp] = interp_point;
      }

      //! getter for the last interpolation point \f$ \xi_{n} \f$ at a specified Gauss point gp
      InterpolationPoint get_last_interp_point(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < last_interp_point_.size(),
            "Cannot retrieve last interpolation point for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, last_interp_point_.size());
        return last_interp_point_[gp];
      };
      //! setter for the last interpolation point \f$ \xi_{n} \f$ at a specified Gauss point gp
      void set_last_interp_point(const unsigned gp, const InterpolationPoint interp_point)
      {
        FOUR_C_ASSERT_ALWAYS(gp < current_interp_point_.size(),
            "Cannot set last interpolation point for Gauss point with index {}! The "
            "assigned number of Gauss points is {}",
            gp, last_interp_point_.size());
        last_interp_point_[gp] = interp_point;
      }


      /*!
       * @brief Add interpolation points.
       *
       * Performs \e this = \e scalar_a * \e interp_point_a + \e scalar_b * \e interp_point_b
       *
       */
      static InterpolationPoint add_interpolation_points(const double scalar_a,
          const InterpolationPoint& interp_point_a, const double scalar_b,
          const InterpolationPoint& interp_point_b)
      {
        return InterpolationPoint{
            .xi_lambda = {scalar_a * interp_point_a.xi_lambda[0] +
                              scalar_b * interp_point_b.xi_lambda[0],
                scalar_a * interp_point_a.xi_lambda[1] + scalar_b * interp_point_b.xi_lambda[1],
                scalar_a * interp_point_a.xi_lambda[2] + scalar_b * interp_point_b.xi_lambda[2]},
            .xi_rel_eigenvect_rot = {scalar_a * interp_point_a.xi_rel_eigenvect_rot[0] +
                                         scalar_b * interp_point_b.xi_rel_eigenvect_rot[0],
                scalar_a * interp_point_a.xi_rel_eigenvect_rot[1] +
                    scalar_b * interp_point_b.xi_rel_eigenvect_rot[1],
                scalar_a * interp_point_a.xi_rel_eigenvect_rot[2] +
                    scalar_b * interp_point_b.xi_rel_eigenvect_rot[2]},
            .xi_rel_rot = {scalar_a * interp_point_a.xi_rel_rot[0] +
                               scalar_b * interp_point_b.xi_rel_rot[0],
                scalar_a * interp_point_a.xi_rel_rot[1] + scalar_b * interp_point_b.xi_rel_rot[1],
                scalar_a * interp_point_a.xi_rel_rot[2] + scalar_b * interp_point_b.xi_rel_rot[2]},
        };
      }


      /*!
       * @brief Interpolate elastic deformation gradient between the
       * elastic and the plastic predictor, given the
       * current several interpolation factors.
       *
       * @note The eigenvalues are interpolated using the logarithmic weighted average
       * (see Satheesh et al. 2022, 10.1002/nme.7373) with linear weighting
       * between the predictors.
       *
       * @param[in] interp_point Interpolation point to be used.
       * @param[in] inv_defgrad Inverse of the current deformation gradient (current Local Newton
       * iteration within \f$ \left[ t_n, t_{n+1} \right] \f$)
       *
       */
      Core::LinAlg::Matrix<3, 3> interpolate_elastic_defgrad(
          const InterpolationPoint& interp_point, const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

      /*!
       * @brief Adapt interpolation interval \f$ \left[\xi_{\mathrm{E}}, \xi_{\mathrm{P} \right] \f$
       * based on evaluation error
       *
       * @param[in] eval_err_type type of evaluation error necessitating
       * an adaptation of the parameter and the interval
       *
       */
      void adapt_interpolation_interval(const ErrorType eval_err_type);

      /*!
       * @brief Adapt current interpolation point \f$ \xi \f$ using the current interpolation
       * interval
       *
       */
      void adapt_current_interpolation_point();

      [[nodiscard]] const std::vector<PredictorDefgradDecomposition>&
      get_curr_pred_decomp_specific_defgrad() const
      {
        return curr_pred_decomp_specific_defgrad_;
      }

      [[nodiscard]] const std::vector<PredictorDefgradDecomposition>&
      get_last_pred_decomp_specific_defgrad() const
      {
        return last_pred_decomp_specific_defgrad_;
      }


     private:
      //! current Gauss point index
      int gp_{-1};

      //! current element id
      int ele_gid_{-1};

      //! type of elastic stretch eigenvalues within plastic predictor
      PlasticPredictorElasticStretchEigenvalType plast_pred_elast_stretch_eigenval_type_;

      //! type of elastic stretch eigenvector rotation within plastic predictor
      PlasticPredictorElasticStretchEigenvectRotType plast_pred_elast_stretch_eigenvect_rot_type_;

      //! type of rotation within plastic predictor
      PlasticPredictorRotationType plast_pred_rot_type_;

      //! interval scanning parameter for interpolation: we currently utilize bisection
      const double scan_param_ = 1.0 / 2.0;

      //! current number of re-estimations
      unsigned int num_of_reestimations_;

      //! maximum number of allowed re-estimations before the local integration is deemed infeasible
      const unsigned int max_num_reestimations_;

      //! minimum interpolation interval as a 2-norm \f$ \|  \mathbf{\xi}_{\text{upper}} -
      //! \mathbf{\xi}_{\text{upper}} \| \f$
      const double min_interp_interval_;

      //! set maximum relative deviation between equivalent stress and yield stress: if
      //! elastic predictor has a smaller stress deviation than this this value, then it is directly
      //! used as the initial LNL guess without performing the LNGI; otherwise, the plastic
      //! predictor is updated such that its relative stress deviation is smaller than this value
      static constexpr double max_rel_stress_deviation_ = 1.0e-6;

      // maximum number of interpolation iterations
      static constexpr unsigned int max_num_interp_iters_ = 50;

      // maximum number of plastic predictor construction iterations
      static constexpr unsigned int max_plastic_pred_construct_iters_ = 50;

      //! current estimate containing the inverse inelastic deformation
      //! gradient (components 0-8) and the plastic strain (component 9)
      Core::LinAlg::Matrix<10, 1> current_interp_estimate_;

      //! current timestep predictor decompositions for each GP of deformation gradient (with
      //! specified type: elastic, inverse plastic, ...) within elastic and plastic predictors
      //!--> MOVE TO PRIVATE AFTER REMOVING CONSISTENCY CHECKS
      std::vector<PredictorDefgradDecomposition> curr_pred_decomp_specific_defgrad_;

      //! last timestep predictor decompositions for each GP of deformation gradient (with
      //! specified type: elastic, inverse plastic, ...) within elastic and plastic predictors
      //!--> MOVE TO PRIVATE AFTER REMOVING CONSISTENCY CHECKS
      std::vector<PredictorDefgradDecomposition> last_pred_decomp_specific_defgrad_;

      //! current interpolation points \f$ \xi \f$ at all Gauss points
      std::vector<InterpolationPoint> current_interp_point_;

      //! lower interpolation bounds \f$ \xi_{\text{E}} \f$ at all Gauss points
      std::vector<InterpolationPoint> lower_interp_bound_;

      //! upper interpolation bounds \f$ \xi_{\text{P}} \f$ at all Gauss points
      std::vector<InterpolationPoint> upper_interp_bound_;


      //! interpolation points \f$ \xi_{n} \f$ at all Gauss points, used in the last converged
      //! global iteration of the previous time step
      std::vector<InterpolationPoint> last_interp_point_;

      //! control variable: should relative elastic rotations be interpolated (if
      //! not, take the component from the elastic predictor)
      const bool interpolate_rot_;

      //! control variable: should relative elastic eigenvector rotations be interpolated (if
      //! not, take the component from the elastic predictor)
      const bool interpolate_eigenvect_rot_;
    };

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
