// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRU_MULTI_MICROSTATIC_HPP
#define FOUR_C_STRU_MULTI_MICROSTATIC_HPP



#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <Teuchos_Time.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationVisualizationWriterMesh;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace MultiScale
{
  /*!
  \brief Stop nested parallelism support by sending a message to the
  supporting procs

  */
  void stop_np_multiscale();

  /*!
  \brief Quasi-static control for microstructural analysis
  in case of multi-scale problems

  Note that implementation currently only holds for imr-like generalized
  alpha time integration. Corresponding functions (e.g. UpdateNewTimeStep,
  but also calls to SurfaceStressManager!) need to be adapted accordingly
  if usage of other time integration schemes should be enabled.

  */

  class MicroStatic
  {
   public:
    /*!
    \brief Standard Constructor

    */
    MicroStatic(const int microdisnum, const double V0);

    /*!
    \brief Destructor

    */
    virtual ~MicroStatic() = default;

    /*!
    \brief Read restart

    */
    void read_restart(const int step, std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::unordered_map<int, std::vector<char>>& history_data, const std::string& name);

    /// get corresponding time to step
    double get_time_to_step(const int step, const std::string& name);

    /*!
    \brief Return time from parameter list

    */
    double time_old() { return time_; }

    /*!
    \brief Predictor step

    */
    void predictor(const Core::LinAlg::Matrix<3, 3>* defgrd);

    /*!
    \brief Predictor step

    */
    void predict_const_dis(const Core::LinAlg::Matrix<3, 3>* defgrd);

    /*!
   \brief Predictor step

   */
    void predict_tang_dis(const Core::LinAlg::Matrix<3, 3>* defgrd);

    /*!
    \brief Full Newton iteration

    */
    void full_newton();

    /*!
    \brief Calculate stresses and strains

    */
    void runtime_pre_output_step_state();

    /*!
     * @brief Write runtime output
     *
     */
    void runtime_output_step_state_microscale(
        std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> micro_visualization_writer,
        const std::pair<double, int>& output_time_and_step, const std::string& section_name) const;

    /*!
     * \brief Update on element level(e.g.internal variables)
     *
     */
    void update_step_element();

    /*!
    \brief Write restart

    */
    void write_restart(std::shared_ptr<Core::IO::DiscretizationWriter> output, const double time,
        const int step, const double dt) const;

    /*!
    \brief Determine toggle vector identifying prescribed boundary dofs

    */
    void determine_toggle();

    /*!
    \brief Evaluate microscale boundary displacement according to
    associated macroscale deformation gradient

    */
    void evaluate_micro_bc(
        const Core::LinAlg::Matrix<3, 3>* defgrd, Core::LinAlg::Vector<double>& disp);

    /*!
    \brief Set old state given from micromaterialgp

    */
    void set_state(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::Vector<double>> disn,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_node_postprocessed,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_element_postprocessed,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_node_postprocessed,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_element_postprocessed,
        std::shared_ptr<std::vector<char>> plstrain,
        std::shared_ptr<Core::LinAlg::Matrix<6, 6>> macro_cmat_output);

    /*!
    \brief Set time and step

    */
    void set_time(
        const double time, const double timen, const double dt, const int step, const int stepn);

    /*!
    \brief Clear all displacement states

    */
    void clear_state();

    /*!
    \brief Set up everything for homogenization
    (e.g. calculation of matrix D containing reference boundary coordinates)

    */
    void set_up_homogenization();

    /*!
    \brief Perform homogenization, i.e. calculate second Piola-Kirchhoff
    stresses and constitutive tensor by averaging over RVE

    */
    void static_homogenization(Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat,
        const Core::LinAlg::Matrix<3, 3>* defgrd, const bool mod_newton, bool& build_stiff);

    /*!
    \brief Convert constitutive tensor relating first Piola-Kirchhoff
    stresses and deformation gradient to tensor relating second
    Piola-Kirchhoff stresses and Green-Lagrange strains

    For details cf.

    Marsden and Hughes, Mathematical Foundations of Elasticity,
    Dover, pg. 215
    */
    void convert_mat(const Core::LinAlg::MultiVector<double>& cmatpf,
        const Core::LinAlg::Matrix<3, 3>& F_inv, const Core::LinAlg::Matrix<6, 1>& S,
        Core::LinAlg::Matrix<6, 6>& cmat);


    /*!
    \brief Check for Newton convergence
    */

    bool converged();

    /*!
    \brief Calculate reference norms for relative convergence checks

    */
    void calc_ref_norms();

    /*!
    \brief Output of Newton details

    Note that this is currently disabled for the sake of clearness
    */
    void print_newton(bool print_unconv, Teuchos::Time timer);

    /*!
    \brief Output of predictor details

    Note that this is currently disabled for the sake of clearness
    */
    void print_predictor();

    double density() const { return density_; };

   private:
    // don't want = operator and cctor
    MicroStatic operator=(const MicroStatic& old);
    MicroStatic(const MicroStatic& old);

    std::shared_ptr<Core::FE::Discretization> discret_;
    std::shared_ptr<Core::LinAlg::Solver> solver_;
    int myrank_;
    int maxentriesperrow_;

    double dt_;
    double time_;
    double timen_;

    Inpar::Solid::PredEnum pred_;  //!< predictor

    bool isadapttol_;
    double adaptolbetter_;

    int maxiter_;
    int numiter_;
    int numstep_;
    int step_;
    int stepn_;

    /// @name variables controlling output
    /// @{

    /// whether to write displacement output
    bool output_displacement_state_;

    /// whether to write the owner of elements
    bool output_element_owner_;

    /// whether to write the element material IDs
    bool output_element_material_id_;

    /// whether to write stress and / or strain data
    bool output_stress_strain_;

    /// Output type of Gauss point data
    Inpar::Solid::GaussPointDataOutputType gauss_point_data_output_type_;
    //@}
    int results_every_;
    Inpar::Solid::StressType iostress_;
    Inpar::Solid::StrainType iostrain_;
    Inpar::Solid::StrainType ioplstrain_;
    int restart_;
    int restart_every_;
    int printscreen_;

    Inpar::Solid::VectorNorm iternorm_;
    double tolfres_;
    double toldisi_;

    enum Inpar::Solid::BinaryOp combdisifres_;  //!< binary operator to
                                                // combine displacement and forces
    enum Inpar::Solid::ConvNorm normtypedisi_;  //!< convergence check for residual displacements
    enum Inpar::Solid::ConvNorm normtypefres_;  //!< convergence check for residual forces
    double normcharforce_;
    double normfres_;
    double normchardis_;
    double normdisi_;

    std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_dirich_;

    std::shared_ptr<Core::LinAlg::Vector<double>> dirichtoggle_;
    std::shared_ptr<Core::LinAlg::Vector<double>> invtoggle_;
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;
    std::shared_ptr<Core::LinAlg::Vector<double>>
        dis_;  //!< displacements at t_{n} (needed for convergence check only)
    std::shared_ptr<Core::LinAlg::Vector<double>> disn_;  //!< displacements at t_{n+1}
    std::shared_ptr<Core::LinAlg::Vector<double>> disi_;
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;
    std::shared_ptr<Core::LinAlg::Vector<double>> fresn_;
    std::shared_ptr<Core::LinAlg::Vector<double>> freactn_;

    std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_node_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> stress_data_element_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_node_postprocessed_;
    std::shared_ptr<Core::LinAlg::MultiVector<double>> strain_data_element_postprocessed_;
    std::shared_ptr<std::vector<char>> plstrain_;
    std::shared_ptr<Core::LinAlg::Matrix<6, 6>>
        macro_cmat_output_;  //!< Averaged tangent stiffness tensor

    std::shared_ptr<Core::LinAlg::MultiVector<double>>
        d_matrix_;  //!< D Matrix following Miehe et al., 2002
    std::shared_ptr<Core::LinAlg::MultiVector<double>>
        rhs_;  //!< exported transpose of D (pdof -> dofrowmap)

    int microdisnum_;  //!< number of RVE

    double initial_volume_;  //!< initial volume of RVE
    double density_;         //!< initial density of RVE

    int ndof_;  //!< number of dofs overall
    int np_;    //!< number of boundary dofs
    std::shared_ptr<Core::LinAlg::Vector<double>>
        material_coords_boundary_nodes_;       //!< vector containing material
                                               //!< coordinates of boundary nodes
    std::shared_ptr<Core::LinAlg::Map> pdof_;  //!< prescribed dofs
    std::shared_ptr<Core::LinAlg::Map> fdof_;  //!< free dofs
    std::shared_ptr<Core::LinAlg::Import> importp_;
    std::shared_ptr<Core::LinAlg::Import> importf_;
  };


  class MicroStaticParObjectType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "MicroStaticParObjectType"; }

    static MicroStaticParObjectType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MicroStaticParObjectType instance_;
  };

  class MicroStaticParObject : public Core::Communication::ParObject
  {
   public:
    [[nodiscard]] inline int unique_par_object_id() const override
    {
      return MultiScale::MicroStaticParObjectType::instance().unique_par_object_id();
    };

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    struct MicroStaticData
    {
      int gp_{};
      int microdisnum_{};
      int eleowner_{};
      double initial_volume_{};
      Core::LinAlg::SerialDenseMatrix defgrd_;
      Core::LinAlg::SerialDenseMatrix stress_;
      Core::LinAlg::SerialDenseMatrix cmat_;
    };

    [[nodiscard]] inline const MicroStaticData* get_micro_static_data_ptr() const
    {
      return std::addressof(microstatic_data_);
    };

    inline void set_micro_static_data(MicroStaticData& micro_data)
    {
      microstatic_data_ = micro_data;
    };

   private:
    MicroStaticData microstatic_data_{};
  };

  //! Micro material nested parallelism action
  enum class MicromaterialNestedParallelismAction : int
  {
    read_restart,       ///< read restart
    post_setup,         ///< perform post setup routine for micro material
    evaluate,           ///< evaluate micro material
    update,             ///< update micro material
    prepare_output,     ///< prepare output for micro material
    output_step_state,  ///< write output for micro material
    write_restart,      ///< write restart output for micro material
    stop_multiscale     ///< after time loop stop multiscale simulation
  };
}  // namespace MultiScale
FOUR_C_NAMESPACE_CLOSE

#endif
