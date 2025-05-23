// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodetopoint.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::PoroMultiPhaseScaTraArtCouplNodeToPoint(
    std::shared_ptr<Core::FE::Discretization> arterydis,
    std::shared_ptr<Core::FE::Discretization> contdis, const Teuchos::ParameterList& couplingparams,
    const std::string& condname, const std::string& artcoupleddofname,
    const std::string& contcoupleddofname)
    : PoroMultiPhaseScaTraArtCouplNonConforming(
          arterydis, contdis, couplingparams, condname, artcoupleddofname, contcoupleddofname)
{
  // user info
  if (myrank_ == 0)
  {
    std::cout << "<                                                  >" << std::endl;
    print_out_coupling_method();
    std::cout << "<                                                  >" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "\n";
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::setup()
{
  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::setup();


  // preevaluate coupling pairs
  pre_evaluate_coupling_pairs();

  // print out summary of pairs
  if (contdis_->name() == "porofluid" && couplingparams_.get<bool>("PRINT_OUT_SUMMARY_PAIRS"))
    output_coupling_pairs();

  // error-checks
  if (has_varying_diam_)
    FOUR_C_THROW("Varying diameter not yet possible for node-to-point coupling");
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not yet possible for node-to-point coupling");

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::pre_evaluate_coupling_pairs()
{
  // pre-evaluate
  for (auto& coupl_elepair : coupl_elepairs_) coupl_elepair->pre_evaluate(nullptr);

  // delete the inactive pairs
  coupl_elepairs_.erase(
      std::remove_if(coupl_elepairs_.begin(), coupl_elepairs_.end(),
          [](const std::shared_ptr<PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>
                  coupling_pair) { return not coupling_pair->is_active(); }),
      coupl_elepairs_.end());

  // output
  int total_numactive_pairs = 0;
  int numactive_pairs = static_cast<int>(coupl_elepairs_.size());
  Core::Communication::sum_all(&numactive_pairs, &total_numactive_pairs, 1, get_comm());
  if (myrank_ == 0)
  {
    std::cout << total_numactive_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs are active" << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::evaluate(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  if (!issetup_) FOUR_C_THROW("setup() has not been called");


  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::evaluate(sysmat, rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::setup_system(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art)
{
  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::setup_system(*sysmat, rhs,
      *sysmat_cont, *sysmat_art, rhs_cont, rhs_art, *dbcmap_cont, *dbcmap_art->cond_map(),
      *dbcmap_art->cond_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::apply_mesh_movement()
{
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not possible for node-to-point coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("Output of vessel volume fraction not possible for node-to-point coupling");

  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::print_out_coupling_method() const
{
  std::cout << "<Coupling-Method: 1D node to coincident point in 3D>" << std::endl;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeToPoint::output_coupling_pairs() const
{
  if (myrank_ == 0)
  {
    std::cout << "\nSummary of coupling pairs (segments):" << std::endl;
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
  }
  Core::Communication::barrier(get_comm());
  for (const auto& coupl_elepair : coupl_elepairs_)
  {
    std::cout << "Proc " << std::right << std::setw(2) << myrank_ << ": Artery-ele " << std::right
              << std::setw(5) << coupl_elepair->ele1_gid() << ": <---> continuous-ele "
              << std::right << std::setw(7) << coupl_elepair->ele2_gid() << std::endl;
  }
  Core::Communication::barrier(get_comm());
  if (myrank_ == 0) std::cout << "\n";
}

FOUR_C_NAMESPACE_CLOSE
