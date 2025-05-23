// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TURBULENCE_STATISTICS_BFS_HPP
#define FOUR_C_FLUID_TURBULENCE_STATISTICS_BFS_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  class TurbulenceStatisticsBfs
  {
   public:
    /*!
    \brief Standard Constructor (public)

        o Create sets for lines

    o Allocate distributed vector for squares

    */
    TurbulenceStatisticsBfs(std::shared_ptr<Core::FE::Discretization> actdis,
        Teuchos::ParameterList& params, const std::string& statistics_outfilename,
        const std::string& geotype);

    /*!
    \brief Destructor

    */
    virtual ~TurbulenceStatisticsBfs() = default;


    //! @name functions for averaging

    /*!
    \brief The values of velocity and its squared values are added to
    global vectors. This method allows to do the time average of the
    nodal values after a certain amount of timesteps.
    */
    void do_time_sample(
        Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& stresses);

    /*!
    \brief The values of velocity, pressure, temperature and its squared
    values are added to global vectors. This method allows to do the time
    average of the nodal values after a certain amount of timesteps.
    */
    void do_loma_time_sample(Core::LinAlg::Vector<double>& velnp,
        Core::LinAlg::Vector<double>& scanp, const double eosfac);

    /*!
    \brief The values of velocity, pressure, phi and its squared
    values are added to global vectors. This method allows to do the time
    average of the nodal values after a certain amount of timesteps.
    */
    void do_scatra_time_sample(
        Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& scanp);

    /*!
    \brief Dump the result to file.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_statistics(int step);

    /*!
    \brief Dump the result to file for low-Mach-number flow.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_loma_statistics(int step);

    /*!
    \brief Dump the result to file for turbulent flow with passive scalar.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_scatra_statistics(int step);


   protected:
    /*!
    \brief sort criterium for double values up to a tolerance of 10-9

    This is used to create sets of doubles (e.g. coordinates)

    */
    class LineSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

   private:
    //! geometry of DNS of incompressible flow over bfs by Le, Moin and Kim or geometry of Avancha
    //! and Pletcher of LES of flow over bfs with heating
    enum GeoType
    {
      none,
      geometry_DNS_incomp_flow,
      geometry_LES_flow_with_heating,
      geometry_EXP_vogel_eaton
    };

    //! number of samples taken
    int numsamp_;

    //! number of coordinates in x1- and x2-direction
    int numx1coor_;
    int numx2coor_;

    //! number of locations in x1- and x2-direction for statistical evaluation
    int numx1statlocations_;
    int numx2statlocations_;
    int numx1supplocations_;

    //! bounds for extension of backward-facing step in x2-direction
    double x2min_;
    double x2max_;

    //! bounds for extension of backward-facing step in x3-direction
    double x3min_;
    double x3max_;

    //! The discretisation (required for nodes, dofs etc;)
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! parameter list
    Teuchos::ParameterList& params_;

    //! geometry of DNS of incompressible flow over bfs by Le, Moin and Kim or geometry of Avancha
    //! and Pletcher of LES of flow over bfs with heating
    FLD::TurbulenceStatisticsBfs::GeoType geotype_;

    //! boolean indicating turbulent inflow channel discretization
    const bool inflowchannel_;
    //! x-coordinate of outflow of inflow channel
    const double inflowmax_;

    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;

    //! pointer to vel/pres^2 field (space allocated in constructor)
    std::shared_ptr<Core::LinAlg::Vector<double>> squaredvelnp_;
    //! pointer to T^2 field (space allocated in constructor)
    std::shared_ptr<Core::LinAlg::Vector<double>> squaredscanp_;
    //! pointer to 1/T field (space allocated in constructor)
    std::shared_ptr<Core::LinAlg::Vector<double>> invscanp_;
    //! pointer to (1/T)^2 field (space allocated in constructor)
    std::shared_ptr<Core::LinAlg::Vector<double>> squaredinvscanp_;

    //! toggle vectors: sums are computed by scalarproducts
    std::shared_ptr<Core::LinAlg::Vector<double>> toggleu_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglev_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglew_;
    std::shared_ptr<Core::LinAlg::Vector<double>> togglep_;

    //! available x1- and x2-coordinates
    std::shared_ptr<std::vector<double>> x1coordinates_;
    std::shared_ptr<std::vector<double>> x2coordinates_;

    //! coordinates of locations in x1- and x2-direction for statistical evaluation
    Core::LinAlg::Matrix<21, 1> x1statlocations_;
    Core::LinAlg::Matrix<2, 1> x2statlocations_;

    //! coordinates of supplementary locations in x2-direction for velocity derivative
    Core::LinAlg::Matrix<2, 1> x2supplocations_;

    //! coordinates of supplementary locations in x1-direction for statistical evaluation
    //! (check of inflow profile)
    Core::LinAlg::Matrix<10, 1> x1supplocations_;

    //! matrices containing values
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x1sumu_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x1sump_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x1sumrho_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x1sum_t_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x1sumtauw_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumu_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumv_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumw_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sump_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumrho_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sum_t_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsqu_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsqv_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsqw_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsqp_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsqrho_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumsq_t_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumuv_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumuw_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumvw_;

    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumrhou_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumu_t_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumrhov_;
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> x2sumv_t_;

    void convert_string_to_geo_type(const std::string& geotype);
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
