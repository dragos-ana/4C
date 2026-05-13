// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <map>
#include <optional>
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
      under_yield_surface,  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                            ///< stress is smaller than the yield stress
    };


    /// struct: settings for registering errors within the procedures used for return mapping
    struct ErrorRegistrationSettings
    {
      //! should overflow error be registered via ErrorType when the plastic strain increment
      //! exceeds the specified tolerance
      const bool register_plastic_strain_incr_overflow;

      //! maximum, numerically evaluable plastic strain increment before overflow error is
      //! registered
      const double max_plastic_strain_incr;

      //! should overflow error be registered via ErrorType when any of the plastic strain
      //! derivative increments exceeds the specified tolerance
      const bool register_plastic_strain_incr_derivs_overflow;

      //! maximum, numerically evaluable increment of
      //! plastic strain derivatives (time_step * derivative)
      const double max_plastic_strain_deriv_incr;
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

      //!  Update values between time steps: last <- current, at a given Gauss points
      void update(const unsigned int gp);

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
        out += "\nSubstepping info: \n";
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

      //! deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Mtheta_dev_sym_M{Core::LinAlg::Initialization::zero};

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

      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMtheta_dev_sym_diFin{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMtheta_dev_sym_dC{Core::LinAlg::Initialization::zero};

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

      //! derivative with respect to the temperature
      double deriv_temperature;
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
      //! \boldsymbol{s}^{l} \right|}{\left| \boldsymbol{s}^{l} \right|}  \f$
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

      /// getter for total number of local iterations evaluated in this time step (vector over all
      /// Gauss points)
      [[nodiscard]] const std::vector<unsigned int>& curr_num_iters() const
      {
        return curr_num_iters_;
      }

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Initialize the solution vector, and the iteration counter (optional), for the
       * subsequent Local Newton at the currently considered Gauss point
       *
       * @param[in] init_estimate initial estimate \f$ \boldsymbol{s}^{(0)} \f$
       * @param[in] reset_iter_counter reset the iteration counter?
       */
      void init_local_newton(
          const Core::LinAlg::Matrix<10, 1>& init_estimate, const bool reset_iter_counter);

      /// sets the residual norm based on the given residual vector
      void set_residual_norm(const Core::LinAlg::Matrix<10, 1>& residual)
      {
        convergence_quantities_.residual_norm = residual.norm2();
      }

      /*!
       * @brief Determine whether the Local Newton Loop has converged, based on the saved
       * convergence quantities and the specified convergence checks.
       *
       * @return boolean: true = converged
       */
      [[nodiscard]] bool is_local_newton_converged() const;


      /*!
       * @brief   After an unsuccessful convergence check: determine whether the Local Newton is
       * stuck, i.e., the relative solution increment is nearly 0, but there is no convergence yet,
       * based on the saved convergence quantities.
       *
       * @return boolean: true = stuck
       */
      [[nodiscard]] bool is_local_newton_stuck() const;


      /// is the maximum number of iterations exceeded?
      [[nodiscard]] bool is_max_iter_exceeded() { return iter_ > params_.max_iter; }


      /*!
       * @brief Increments the solution vector \f$ \boldsymbol{s}^{(l+1)} = \boldsymbol{s}^{(l)}
       * +
       * \Delta \boldsymbol{s}^{(l+1)} \f$ after the current iteration \f$ l \f$, along with the
       * iteration counter
       *
       * @note Also updates the increment norm (ratio of increment to solution) internally based on
       * the provided increment
       *
       * @param[in] delta_sol increment vector for the next iteration \f$\Delta
       * \boldsymbol{s}^{(l+1)}\f$
       */
      void increment_solution_vector_and_iter(const Core::LinAlg::Matrix<10, 1>& delta_sol);

      /// getter for the solution vector
      [[nodiscard]] Core::LinAlg::Matrix<10, 1> sol() const { return sol_; }


      /// getter for the convergence quantities
      [[nodiscard]] LocalNewtonConvQuantities convergence_quantities() const
      {
        return convergence_quantities_;
      }

      /*!
       * @brief Routine to be run after the Local Newton-Raphson at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void update_after_local_newton(const unsigned int gp);

      //! reset the saved number of iterations at a given Gauss point
      void reset_curr_num_iters(const unsigned int gp);

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

      //! solution vector in the current iteration \f$ \boldsymbol{s}^{(l)} \f$ (used at the
      //! currently considered Gauss point)
      Core::LinAlg::Matrix<10, 1> sol_;

      //! quantities used for convergence checks
      LocalNewtonConvQuantities convergence_quantities_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};
    };

    //! helper struct containing tensors associated with the deformation, passed as input
    //! for the local time integration
    struct LocalIntegrationDeformationTensors
    {
      /*!
       * @brief constructor
       *
       * @param[in] F deformation gradient \f$ \mathbf{F}_{n+1} \f$
       * @param[in] last_iFp previous inelastic/plastic deformation gradient \f$
       \mathbf{F}_{\text{p},n}^{-1} \f$
       *
       */
      LocalIntegrationDeformationTensors(
          const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& last_iFp);

      //! deformation gradient \f$ \mathbf{F}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> defgrad;

      //! inverse deformation gradient \f$ \mathbf{F}_{n+1}^{-1} \f$
      Core::LinAlg::Matrix<3, 3> inv_defgrad;

      //! right Cauchy-Green deformation tensor \f$ \mathbf{C}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> right_cg;

      //! inverse plastic deformation gradient within the elastic predictor \f$
      //! \left[ \mathbf{F}_{\mathrm{p},n+1}^{(\mathrm{E})} \right]^{-1} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_inverse_plastic_defgrad;

      //! elastic deformation gradient within the elastic predictor \f$
      //! \mathbf{F}_{\mathrm{e},n+1}^{(\mathrm{E})} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_elastic_defgrad;
    };

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
