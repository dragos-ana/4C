/*----------------------------------------------------------------------*/
/*!
 \file scatra_ele_calc_cardiac_monodomain.cpp

 \brief

 <pre>
   Maintainer: Lasse Jagschies
               jagschies@mhpc.mw.tum.de
               http://www.lnm.mw.tum.de
               089 - 289-10365
 </pre>
 *----------------------------------------------------------------------*/


#include "scatra_ele_calc_cardiac_monodomain.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"

#include "scatra_ele_parameter_timint.H"

#include "../drt_mat/myocard.H"
#include "../drt_mat/matlist.H"



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype>::ScaTraEleCalcCardiacMonodomain(const int numdofpernode,const int numscal)
  : DRT::ELEMENTS::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode,numscal),
    DRT::ELEMENTS::ScaTraEleCalcAniso<distype>::ScaTraEleCalcAniso(numdofpernode,numscal),
    DRT::ELEMENTS::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(numdofpernode,numscal)
{

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype> * DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype>::Instance(
  const int numdofpernode,
  const int numscal,
  bool create )
{
  static ScaTraEleCalcCardiacMonodomain<distype> * instance;
  if ( create )
  {
    if ( instance==NULL )
    {
      instance = new ScaTraEleCalcCardiacMonodomain<distype>(numdofpernode,numscal);
    }
  }
  else
  {
    if ( instance!=NULL )
      delete instance;
    instance = NULL;
  }
  return instance;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype>::Done()
{
  // delete this pointer! Afterwards we have to go! But since this is a
  // cleanup call, we can do it this way.
  Instance( 0, 0, false );
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    ehrl 11/13 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype>::Materials(
  const Teuchos::RCP<const MAT::Material> material, //!< pointer to current material
  const int                               k,        //!< id of current scalar
  double&                                 densn,    //!< density at t_(n)
  double&                                 densnp,   //!< density at t_(n+1) or t_(n+alpha_F)
  double&                                 densam,   //!< density at t_(n+alpha_M)
  Teuchos::RCP<ScaTraEleDiffManager>      diffmanager,  //!< diffusion manager handling diffusivity / diffusivities (in case of systems) or (thermal conductivity/specific heat) in case of loma
  Teuchos::RCP<ScaTraEleReaManager>       reamanager,   //!< reaction manager
  double&                                 visc,         //!< fluid viscosity
  const int                               iquad         //!< id of current gauss point

  )
{
  if (material->MaterialType() == INPAR::MAT::m_myocard)
    MatMyocard(material,k,densn,densnp,densam,diffmanager,reamanager,visc,iquad);
  else dserror("Material type is not supported");

  return;
}


/*----------------------------------------------------------------------*
 |  Material ScaTra                                          ehrl 11/13 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<distype>::MatMyocard(
  const Teuchos::RCP<const MAT::Material> material, //!< pointer to current material
  const int                               k,        //!< id of current scalar
  double&                                 densn,    //!< density at t_(n)
  double&                                 densnp,   //!< density at t_(n+1) or t_(n+alpha_F)
  double&                                 densam,   //!< density at t_(n+alpha_M)
  Teuchos::RCP<ScaTraEleDiffManager>      diffmanager,  //!< diffusion manager handling diffusivity / diffusivities (in case of systems) or (thermal conductivity/specific heat) in case of loma
  Teuchos::RCP<ScaTraEleReaManager>       reamanager,   //!< reaction manager
  double&                                 visc,     //!< fluid viscosity
  const int                               iquad   //!< id of current gauss point (default = -1)
  )
{
  const Teuchos::RCP<const MAT::Myocard>& actmat
    = Teuchos::rcp_dynamic_cast<const MAT::Myocard>(material);

  // dynamic cast to Advanced_Reaction-specific reaction manager
  Teuchos::RCP<ScaTraEleReaManagerAdvReac> advreamanager = Teuchos::rcp_dynamic_cast<ScaTraEleReaManagerAdvReac>(reamanager);

  // dynamic cast to anisotropic diffusion manager
  Teuchos::RCP<ScaTraEleDiffManagerAniso<my::nsd_> > diffmanageraniso = Teuchos::rcp_dynamic_cast<ScaTraEleDiffManagerAniso<my::nsd_> >(diffmanager);

  // get constant diffusivity
  LINALG::Matrix<my::nsd_,my::nsd_> difftensor(true);
  actmat->Diffusivity(difftensor);

  diffmanageraniso->SetAnisotropicDiff(difftensor,k);

  // get membrane potential at n+1 or n+alpha_F at integration point
  const double phinp = my::funct_.Dot(my::ephinp_[k]);

  // get reaction coefficient
  advreamanager->SetReaBodyForce(-actmat->ReaCoeff(phinp, my::scatraparatimint_->Dt()),k);
  advreamanager->SetReaBodyForceDerivMatrix(-actmat->ReaCoeffDeriv(phinp, my::scatraparatimint_->Dt()),k,k);
  advreamanager->SetReaCoeff(0.0, k);
  advreamanager->SetReaCoeffDerivMatrix(0.0,k,k);
  reamanager->SetReaCoeff(0.0, k);
  reamanager->SetReaCoeffDerivMatrix(0.0, k, k);

  return;
} // ScaTraEleCalcCardiacMonodomain<distype>::MatMyocard



// template classes

// 2D elements
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::line2>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::tri3>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::quad4>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::quad9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::hex8>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::hex27>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::tet4>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::tet10>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::wedge6>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::nurbs9>;
//template class DRT::ELEMENTS::ScaTraEleCalcCardiacMonodomain<DRT::Element::nurbs27>;
