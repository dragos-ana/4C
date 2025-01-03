// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_immersed.hpp"

#include "4C_fluid_ele_immersed_base.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_tds.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcImmersed<distype>*
Discret::Elements::FluidEleCalcImmersed<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidEleCalcImmersed<distype>>(
            new Discret::Elements::FluidEleCalcImmersed<distype>());
      });

  return singleton_owner.instance(action);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcImmersed<distype>::FluidEleCalcImmersed()
    : Discret::Elements::FluidEleCalc<distype>::FluidEleCalc(), immersedele_(nullptr), gp_iquad_(0)
{
  my::fldpara_ = Discret::Elements::FluidEleParameterStd::instance();
}

/*----------------------------------------------------------------------*
 * Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcImmersed<distype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, bool offdiag)
{
  // get integration rule for fluid elements cut by structural boundary
  int num_gp_fluid_bound =
      Global::Problem::instance()->immersed_method_params().get<int>("NUM_GP_FLUID_BOUND");
  int degree_gp_fluid_bound = 3;
  if (num_gp_fluid_bound == 8)
    degree_gp_fluid_bound = 3;
  else if (num_gp_fluid_bound == 64)
    degree_gp_fluid_bound = 7;
  else if (num_gp_fluid_bound == 125)
    degree_gp_fluid_bound = 9;
  else if (num_gp_fluid_bound == 343)
    degree_gp_fluid_bound = 13;
  else if (num_gp_fluid_bound == 729)
    degree_gp_fluid_bound = 17;
  else if (num_gp_fluid_bound == 1000)
    degree_gp_fluid_bound = 19;
  else
    FOUR_C_THROW(
        "Invalid value for parameter NUM_GP_FLUID_BOUND (valid parameters are 8, 64, 125, 343, 729 "
        "and 1000).");

  // initialize integration rules
  const Core::FE::GaussIntegration intpoints_fluid_bound(distype, degree_gp_fluid_bound);
  const Core::FE::GaussIntegration intpoints_std(distype);

  // store current element
  immersedele_ = dynamic_cast<Discret::Elements::FluidImmersedBase*>(ele);

  // use different integration rule for fluid elements that are cut by the structural boundary
  if (immersedele_->is_boundary_immersed())
  {
    return my::evaluate(ele, discretization, lm, params, mat, elemat1_epetra, elemat2_epetra,
        elevec1_epetra, elevec2_epetra, elevec3_epetra, intpoints_fluid_bound, offdiag);
  }
  else
  {
    return my::evaluate(ele, discretization, lm, params, mat, elemat1_epetra, elemat2_epetra,
        elevec1_epetra, elevec2_epetra, elevec3_epetra, intpoints_std, offdiag);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcImmersed<distype>::compute_subgrid_scale_velocity(
    const Core::LinAlg::Matrix<nsd_, nen_>& eaccam, double& fac1, double& fac2, double& fac3,
    double& facMtau, int iquad, double* saccn, double* sveln, double* svelnp)
{
  // set number of current gp
  gp_iquad_ = iquad;
  // compute convective conservative term from previous iteration u_old(u_old*nabla)
  Core::LinAlg::Matrix<nsd_, 1> conv_old_cons(true);
  conv_old_cons.update(my::vdiv_, my::convvelint_, 0.0);

  //----------------------------------------------------------------------
  // compute residual of momentum equation
  // -> different for generalized-alpha and other time-integration schemes
  //----------------------------------------------------------------------
  if (my::fldparatimint_->is_genalpha())
  {
    // rhs of momentum equation: density*bodyforce at n+alpha_F
    if (my::fldpara_->physical_type() == Inpar::FLUID::boussinesq)
    {
      // safety check
      if (my::fldparatimint_->alpha_f() != 1.0 or my::fldparatimint_->gamma() != 1.0)
        FOUR_C_THROW(
            "Boussinesq approximation in combination with generalized-alpha time integration "
            "has only been tested for BDF2-equivalent time integration parameters! "
            "Feel free to remove this error at your own risk!");

      my::rhsmom_.update(my::deltadens_, my::bodyforce_, 0.0);
    }
    else
      my::rhsmom_.update(my::densaf_, my::bodyforce_, 0.0);

    // add pressure gradient prescribed as body force (caution: not density weighted)
    my::rhsmom_.update(1.0, my::generalbodyforce_, 1.0);

    // get acceleration at time n+alpha_M at integration point
    my::accint_.multiply(eaccam, my::funct_);

    // evaluate momentum residual once for all stabilization right hand sides
    for (int rr = 0; rr < nsd_; ++rr)
    {
      if (immersedele_->has_projected_dirichlet())
        my::momres_old_(rr) = my::densam_ * my::accint_(rr) +
                              my::densaf_ * (my::conv_old_(rr) + conv_old_cons(rr)) +
                              my::gradp_(rr) - 2 * my::visceff_ * my::visc_old_(rr) +
                              my::reacoeff_ * my::velint_(rr) - my::rhsmom_(rr);
      else
        my::momres_old_(rr) = my::densam_ * my::accint_(rr) + my::densaf_ * my::conv_old_(rr) +
                              my::gradp_(rr) - 2 * my::visceff_ * my::visc_old_(rr) +
                              my::reacoeff_ * my::velint_(rr) - my::rhsmom_(rr);
    }

    // add consistency terms for MFS if applicable
    my::multfrac_sub_grid_scales_consistent_residual();
  }
  else
  {
    if (not my::fldparatimint_->is_stationary())
    {
      // rhs of instationary momentum equation:
      // density*theta*bodyforce at n+1 + density*(histmom/dt)
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g Changed density from densn_
      // to densaf_. Makes the OST consistent with the gen-alpha.
      if (my::fldpara_->physical_type() == Inpar::FLUID::boussinesq)
        my::rhsmom_.update((my::densaf_ / my::fldparatimint_->dt() / my::fldparatimint_->theta()),
            my::histmom_, my::deltadens_, my::bodyforce_);
      else
        my::rhsmom_.update((my::densaf_ / my::fldparatimint_->dt() / my::fldparatimint_->theta()),
            my::histmom_, my::densaf_, my::bodyforce_);

      // add pressure gradient prescribed as body force (caution: not density weighted)
      my::rhsmom_.update(1.0, my::generalbodyforce_, 1.0);

      // compute instationary momentum residual:
      // momres_old = u_(n+1)/dt + theta ( ... ) - histmom_/dt - theta*bodyforce_
      for (int rr = 0; rr < nsd_; ++rr)
      {
        if (immersedele_->has_projected_dirichlet())
          my::momres_old_(rr) = ((my::densaf_ * my::velint_(rr) / my::fldparatimint_->dt() +
                                     my::fldparatimint_->theta() *
                                         (my::densaf_ * (my::conv_old_(rr) + conv_old_cons(rr)) +
                                             my::gradp_(rr) - 2 * my::visceff_ * my::visc_old_(rr) +
                                             my::reacoeff_ * my::velint_(rr))) /
                                    my::fldparatimint_->theta()) -
                                my::rhsmom_(rr);
        else
          my::momres_old_(rr) =
              ((my::densaf_ * my::velint_(rr) / my::fldparatimint_->dt() +
                   my::fldparatimint_->theta() * (my::densaf_ * my::conv_old_(rr) + my::gradp_(rr) -
                                                     2 * my::visceff_ * my::visc_old_(rr) +
                                                     my::reacoeff_ * my::velint_(rr))) /
                  my::fldparatimint_->theta()) -
              my::rhsmom_(rr);
      }
    }
    else
    {
      // rhs of stationary momentum equation: density*bodyforce
      // in the case of a Boussinesq approximation: f = rho_0*[(rho - rho_0)/rho_0]*g = (rho -
      // rho_0)*g else:                                      f = rho * g and pressure gradient
      // prescribed as body force (not density weighted)
      if (my::fldpara_->physical_type() == Inpar::FLUID::boussinesq)
        my::rhsmom_.update(my::deltadens_, my::bodyforce_, 1.0, my::generalbodyforce_);
      else
        my::rhsmom_.update(my::densaf_, my::bodyforce_, 1.0, my::generalbodyforce_);

      // compute stationary momentum residual:
      for (int rr = 0; rr < nsd_; ++rr)
      {
        if (immersedele_->has_projected_dirichlet())
          my::momres_old_(rr) = my::densaf_ * (my::conv_old_(rr) + conv_old_cons(rr)) +
                                my::gradp_(rr) - 2 * my::visceff_ * my::visc_old_(rr) +
                                my::reacoeff_ * my::velint_(rr) - my::rhsmom_(rr);
        else
          my::momres_old_(rr) = my::densaf_ * my::conv_old_(rr) + my::gradp_(rr) -
                                2 * my::visceff_ * my::visc_old_(rr) +
                                my::reacoeff_ * my::velint_(rr) - my::rhsmom_(rr);
      }

      // add consistency terms for MFS if applicable
      my::multfrac_sub_grid_scales_consistent_residual();
    }
  }

  //----------------------------------------------------------------------
  // compute subgrid-scale velocity
  //----------------------------------------------------------------------
  // 1) quasi-static subgrid scales
  // Definition of subgrid-scale velocity is not consistent for the SUPG term and Franca, Valentin,
  // ... Definition of subgrid velocity used by Hughes
  if (my::fldpara_->tds() == Inpar::FLUID::subscales_quasistatic)
  {
    my::sgvelint_.update(-my::tau_(1), my::momres_old_, 0.0);
  }
  // 2) time-dependent subgrid scales
  else
  {
    // some checking
    if (my::fldparatimint_->is_stationary())
      FOUR_C_THROW("there is no time dependent subgrid scale closure for stationary problems\n");
    if (saccn == nullptr or sveln == nullptr or svelnp == nullptr)
      FOUR_C_THROW("no subscale array provided");

    // parameter definitions
    double alphaF = my::fldparatimint_->alpha_f();
    double alphaM = my::fldparatimint_->alpha_m();
    double gamma = my::fldparatimint_->gamma();
    double dt = my::fldparatimint_->dt();

    /*
                                            1.0
       facMtau =  -------------------------------------------------------
                     n+aM                      n+aF
                  rho     * alphaM * tauM + rho     * alphaF * gamma * dt
    */
    facMtau =
        1.0 / (my::densam_ * alphaM * my::tau_(1) + my::densaf_ * my::fldparatimint_->afgdt());

    /*
       factor for old subgrid velocities:

                 n+aM                      n+aF
       fac1 = rho     * alphaM * tauM + rho     * gamma * dt * (alphaF-1)
    */
    fac1 =
        (my::densam_ * alphaM * my::tau_(1) + my::densaf_ * gamma * dt * (alphaF - 1.0)) * facMtau;
    /*
      factor for old subgrid accelerations

                 n+aM
       fac2 = rho     * tauM * dt * (alphaM-gamma)
    */
    fac2 = (my::densam_ * dt * my::tau_(1) * (alphaM - gamma)) * facMtau;
    /*
      factor for residual in current subgrid velocities:

       fac3 = gamma * dt * tauM
    */
    fac3 = (gamma * dt * my::tau_(1)) * facMtau;

    // warning: time-dependent subgrid closure requires generalized-alpha time
    // integration
    if (!my::fldparatimint_->is_genalpha())
    {
      FOUR_C_THROW("the time-dependent subgrid closure requires a genalpha time integration\n");
    }

    /*         +-                                       -+
        ~n+1   |        ~n           ~ n            n+1  |
        u    = | fac1 * u  + fac2 * acc  -fac3 * res     |
         (i)   |                                    (i)  |
               +-                                       -+
    */

    /* compute the intermediate value of subscale velocity

            ~n+af            ~n+1                   ~n
            u     = alphaF * u     + (1.0-alphaF) * u
             (i)              (i)

    */

    static Core::LinAlg::Matrix<1, nsd_> sgvelintaf(true);
    sgvelintaf.clear();
    for (int rr = 0; rr < nsd_; ++rr)
    {
      my::tds_->update_svelnp_in_one_direction(fac1, fac2, fac3, my::momres_old_(rr),
          my::fldparatimint_->alpha_f(), rr, iquad,
          my::sgvelint_(
              rr),  // sgvelint_ is set to sgvelintnp, but is then overwritten below anyway!
          sgvelintaf(rr));

      int pos = rr + nsd_ * iquad;

      /*
       *  ~n+1           ~n           ~ n            n+1
       *  u    =  fac1 * u  + fac2 * acc  -fac3 * res
       *   (i)
       *
       */

      svelnp[pos] = fac1 * sveln[pos] + fac2 * saccn[pos] - fac3 * my::momres_old_(rr);

      /* compute the intermediate value of subscale velocity
       *
       *          ~n+af            ~n+1                   ~n
       *          u     = alphaF * u     + (1.0-alphaF) * u
       *           (i)              (i)
       *
       */
      my::sgvelint_(rr) = alphaF * svelnp[pos] + (1.0 - alphaF) * sveln[pos];
    }
  }  // end time dependent subgrid scale closure

  //----------------------------------------------------------------------
  // include computed subgrid-scale velocity in convective term
  // -> only required for cross- and Reynolds-stress terms
  //----------------------------------------------------------------------
  if (my::fldpara_->cross() != Inpar::FLUID::cross_stress_stab_none or
      my::fldpara_->reynolds() != Inpar::FLUID::reynolds_stress_stab_none or
      my::fldpara_->conti_cross() != Inpar::FLUID::cross_stress_stab_none or
      my::fldpara_->conti_reynolds() != Inpar::FLUID::reynolds_stress_stab_none)
    my::sgconv_c_.multiply_tn(my::derxy_, my::sgvelint_);
  else
    my::sgconv_c_.clear();

  return;
}


template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcImmersed<distype>::lin_gal_mom_res_u(
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, const double& timefacfac)
{
  /*
      instationary                                 conservative          cross-stress, part 1
       +-----+                             +-------------------------+   +-------------------+
       |     |                             |                         |   |                   |

                 /       n+1       \           /               n+1  \     /      ~n+1       \
       rho*Du + |   rho*u   o nabla | Du  + Du |   rho*nabla o u     | + |   rho*u   o nabla | Du +
                 \      (i)        /           \               (i)  /     \       (i)        /

                 /                \  n+1     n+1  /                 \
              + |   rho*Du o nabla | u    + u    | rho*nabla o Du    |  + sigma*Du
                 \                /   (i)    (i)  \                 /
                |                        | |                         |    |       |
                +------------------------+ +-------------------------+    +-------+
                        Newton                Newton (conservative)       reaction
  */


  // convective form of momentum residual
  my::lin_gal_mom_res_u(lin_resM_Du, timefacfac);

  // add conservative terms for covered fluid integration points
  const double timefacfac_densaf = timefacfac * my::densaf_;
  std::array<int, nsd_> idim_nsd_p_idim;
  for (int idim = 0; idim < nsd_; ++idim)
  {
    idim_nsd_p_idim[idim] = idim * nsd_ + idim;
  }


  // convection, convective part (conservative addition)
  if (immersedele_->has_projected_dirichlet())
  {
    for (int ui = 0; ui < nen_; ++ui)
    {
      const double v = timefacfac_densaf * my::vdiv_;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        lin_resM_Du(idim_nsd_p_idim[idim], ui) += my::funct_(ui) * v;
      }
    }

    //  convection, reactive part (conservative addition) (only for Newton)
    if (my::fldpara_->is_newton())
    {
      for (int ui = 0; ui < nen_; ++ui)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          const double temp = timefacfac_densaf * my::velint_(idim);
          const int idim_nsd = idim * nsd_;

          for (int jdim = 0; jdim < nsd_; ++jdim)
          {
            lin_resM_Du(idim_nsd + jdim, ui) += temp * my::derxy_(jdim, ui);
          }
        }
      }
    }
  }

  return;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcImmersed<distype>::inertia_convection_reaction_gal_part(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce,
    Core::LinAlg::Matrix<nsd_ * nsd_, nen_>& lin_resM_Du, Core::LinAlg::Matrix<nsd_, 1>& resM_Du,
    const double& rhsfac)
{
  my::inertia_convection_reaction_gal_part(estif_u, velforce, lin_resM_Du, resM_Du, rhsfac);

  if (immersedele_->has_projected_dirichlet())
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      /* convection (conservative addition) on right-hand side */
      double v = -rhsfac * my::densaf_ * my::velint_(idim) * my::vdiv_;

      if (my::fldpara_->physical_type() == Inpar::FLUID::loma)
        v += rhsfac * my::velint_(idim) * my::densaf_ * my::scaconvfacaf_ * my::conv_scaaf_;
      else if (my::fldpara_->physical_type() == Inpar::FLUID::varying_density)
        v -= rhsfac * my::velint_(idim) * my::conv_scaaf_;

      for (int vi = 0; vi < nen_; ++vi) velforce(idim, vi) += v * my::funct_(vi);
    }
  }

  return;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcImmersed<distype>::continuity_gal_part(
    Core::LinAlg::Matrix<nen_, nen_ * nsd_>& estif_q_u, Core::LinAlg::Matrix<nen_, 1>& preforce,
    const double& timefacfac, const double& timefacfacpre, const double& rhsfac)
{
  for (int vi = 0; vi < nen_; ++vi)
  {
    const double v = timefacfacpre * my::funct_(vi);
    for (int ui = 0; ui < nen_; ++ui)
    {
      const int fui = nsd_ * ui;

      for (int idim = 0; idim < nsd_; ++idim)
      {
        /* continuity term */
        /*
             /                \
            |                  |
            | nabla o Du  , q  |
            |                  |
             \                /
        */
        estif_q_u(vi, fui + idim) += v * my::derxy_(idim, ui);
      }
    }
  }  // end for(idim)

  // continuity term on right-hand side
  if (not immersedele_->is_immersed()) preforce.update(-rhsfac * my::vdiv_, my::funct_, 1.0);

  if (immersedele_->is_boundary_immersed())
    preforce.update(
        rhsfac * immersedele_->projected_int_point_divergence(gp_iquad_), my::funct_, 1.0);


  return;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcImmersed<distype>::conservative_formulation(
    Core::LinAlg::Matrix<nen_ * nsd_, nen_ * nsd_>& estif_u,
    Core::LinAlg::Matrix<nsd_, nen_>& velforce, const double& timefacfac, const double& rhsfac)
{
  if (not(immersedele_->has_projected_dirichlet()))
    my::conservative_formulation(estif_u, velforce, timefacfac, rhsfac);
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// Ursula is responsible for this comment!
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::hex8>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::hex20>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::hex27>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::tet4>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::tet10>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::wedge6>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::wedge15>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::pyramid5>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::quad4>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::quad8>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::quad9>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::tri3>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::tri6>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::nurbs9>;
template class Discret::Elements::FluidEleCalcImmersed<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
