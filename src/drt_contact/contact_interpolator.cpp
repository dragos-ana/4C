/*!----------------------------------------------------------------------
\file contact_interpolator.cpp

<pre>
-------------------------------------------------------------------------
                        BACI Contact library
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Prof. Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Philipp Farah
            farah@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Header                                                    farah 09/14|
 *----------------------------------------------------------------------*/
#include "contact_interpolator.H"
#include "contact_integrator.H"
#include "contact_element.H"
#include "contact_defines.H"
#include "friction_node.H"

#include "../drt_mortar/mortar_projector.H"
#include "../drt_mortar/mortar_defines.H"

#include "../linalg/linalg_serialdensematrix.H"
#include "../linalg/linalg_serialdensevector.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 09/14|
 *----------------------------------------------------------------------*/
CONTACT::CoInterpolator::CoInterpolator(Teuchos::ParameterList& params) :
iparams_(params),
pwslip_(DRT::INPUT::IntegralValue<int>(iparams_,"GP_SLIP_INCR")),
wearlaw_(DRT::INPUT::IntegralValue<INPAR::CONTACT::WearLaw>(iparams_,"WEARLAW")),
wearimpl_(false),
wearside_(INPAR::CONTACT::wear_slave),
weartype_(INPAR::CONTACT::wear_expl),
wearshapefcn_(INPAR::CONTACT::wear_shape_standard),
wearcoeff_(-1.0),
wearcoeffm_(-1.0),
sswear_(DRT::INPUT::IntegralValue<int>(iparams_,"SSWEAR")),
ssslip_(iparams_.get<double>("SSSLIP"))
{
  // wear specific
  if(wearlaw_!=INPAR::CONTACT::wear_none)
  {
    // set wear contact status
    INPAR::CONTACT::WearType wtype = DRT::INPUT::IntegralValue<INPAR::CONTACT::WearType>(params,"WEARTYPE");
    if (wtype == INPAR::CONTACT::wear_impl)
      wearimpl_ = true;

    // wear surface
    wearside_ = DRT::INPUT::IntegralValue<INPAR::CONTACT::WearSide>(iparams_,"BOTH_SIDED_WEAR");

    // wear algorithm
    weartype_ = DRT::INPUT::IntegralValue<INPAR::CONTACT::WearType>(iparams_,"WEARTYPE");

    // wear shape function
    wearshapefcn_ = DRT::INPUT::IntegralValue<INPAR::CONTACT::WearShape>(iparams_,"WEAR_SHAPEFCN");

    // wear coefficient
    wearcoeff_ = iparams_.get<double>("WEARCOEFF");

    // wear coefficient
    wearcoeffm_ = iparams_.get<double>("WEARCOEFF_MASTER");
  }

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::Interpolate2D(MORTAR::MortarElement& sele,
                                            std::vector<MORTAR::MortarElement*> meles)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i=0;i<(int)meles.size();++i)
  {
    if ((!sele.IsSlave()) || (meles[i]->IsSlave()))
      dserror("ERROR: IntegrateAndDerivSegment called on a wrong type of MortarElement pair!");
  }

  // contact with wear
  bool wear = false;
  if(iparams_.get<double>("WEARCOEFF")!= 0.0)
    wear = true;

  //**************************************************************
  //                loop over all Slave nodes
  //**************************************************************
  for(int snodes = 0; snodes<sele.NumNode() ;++snodes)
  {
    CoNode* mynode = dynamic_cast<CoNode*>(sele.Nodes()[snodes]);
    bool kink_projection=false;

    // skip this node if already considered
    if (mynode->MoData().GetD().size()!=0)
      continue;

    double area = 0.0;
    for (int ele=0;ele<mynode->NumElement();++ele)
      area+=dynamic_cast<CONTACT::CoElement*>(mynode->Elements()[ele])->MoData().Area();

    area=area/mynode->NumElement();

    //**************************************************************
    //                loop over all Master Elements
    //**************************************************************
    for (int nummaster=0;nummaster<(int)meles.size();++nummaster)
    {
      // project Gauss point onto master element
      double mxi[2] = {0.0, 0.0};
      MORTAR::MortarProjector::Impl(*meles[nummaster])->ProjectNodalNormal(
          *mynode, *meles[nummaster], mxi);

      // node on mele?
      if ((mxi[0]>=-1.0) && (mxi[0]<=1.0) && (kink_projection==false))
      {
        kink_projection   = true;
        mynode->HasProj() = true;

        int ndof = 2;
        int ncol = meles[nummaster]->NumNode();
        LINALG::SerialDenseVector mval(ncol);
        LINALG::SerialDenseMatrix mderiv(ncol,1);
        meles[nummaster]->EvaluateShape(mxi,mval,mderiv,ncol,false);

        // get slave and master nodal coords for Jacobian / GP evaluation
        LINALG::SerialDenseMatrix scoord(3,sele.NumNode());
        LINALG::SerialDenseMatrix mcoord(3,ncol);
        sele.GetNodalCoords(scoord);
        meles[nummaster]->GetNodalCoords(mcoord);

        // nodal coords from previous time step and lagrange mulitplier
        Teuchos::RCP<LINALG::SerialDenseMatrix> scoordold;
        Teuchos::RCP<LINALG::SerialDenseMatrix> mcoordold;
        Teuchos::RCP<LINALG::SerialDenseMatrix> lagmult;

        scoordold = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,sele.NumNode()));
        mcoordold = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,ncol));
        lagmult   = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,sele.NumNode()));
        sele.GetNodalCoordsOld(*scoordold);
        meles[nummaster]->GetNodalCoordsOld(*mcoordold);
        sele.GetNodalLagMult(*lagmult);

        // TODO: calculate reasonable linsize
        int linsize    = 100;
        double gpn[3]  = {0.0, 0.0, 0.0};
        double jumpval = 0.0;
        GEN::pairedvector<int,double> dgap(linsize+ndof*ncol);
        GEN::pairedvector<int,double> dslipmatrix(linsize+ndof*ncol);
        GEN::pairedvector<int,double> dwear(linsize+ndof*ncol);
        //**************************************************************
        double sxi[2] = {0.0, 0.0};

        if(sele.Shape() == DRT::Element::line2)
        {
          if(snodes==0)
            sxi[0] = -1;
          if(snodes==1)
            sxi[0] = 1;
        }
        else if (sele.Shape() == DRT::Element::line3)
        {
          if(snodes==0)
            sxi[0] = -1;
          if(snodes==1)
            sxi[0] = 1;
          if(snodes == 2)
            sxi[0] = 0;
        }
        else
        {
          dserror("ERROR: Chosen element type not supported for nts!");
        }
        //**************************************************************

        // evalute the GP slave coordinate derivatives --> no entries
        GEN::pairedvector<int,double> dsxi(linsize+ndof*ncol);

        // evalute the GP master coordinate derivatives
        GEN::pairedvector<int,double> dmxi(linsize+ndof*ncol);
        DerivXiGP2D(sele,*meles[nummaster],sxi[0],mxi[0],dsxi,dmxi,linsize);

        // calculate node-wise DM
        nwDM2D(*mynode,*meles[nummaster],mval,mderiv,dmxi);

        // calculate node-wise un-weighted gap
        nwGap2D(*mynode, *meles[nummaster], mval, mderiv, dmxi, gpn);

        // calculate node-wise wear
        if(wear)
          nwWear2D(*mynode, *meles[nummaster], mval, mderiv, scoord, mcoord, scoordold, mcoordold,
                   lagmult,snodes,linsize, jumpval, area, gpn, dmxi, dslipmatrix, dwear);

        // calculate node-wise slip
        if(pwslip_)
          nwSlip2D(*mynode, *meles[nummaster], mval, mderiv, scoord, mcoord, scoordold, mcoordold,
                   snodes, linsize, dmxi);

        // calculate node-wise wear (prim. var.)
        if(weartype_ == INPAR::CONTACT::wear_discr)
          nwTE2D(*mynode, area, jumpval, dslipmatrix);
      }//End hit ele
    }//End Loop over all Master Elements
  }
  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  interpolate (public)                                     farah 09/14|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::Interpolate3D(MORTAR::MortarElement& sele,
                                            std::vector<MORTAR::MortarElement*> meles)
{
  // ********************************************************************
  // Check integrator input for non-reasonable quantities
  // *********************************************************************
  // check input data
  for (int i=0;i<(int)meles.size();++i)
  {
    if ((!sele.IsSlave()) || (meles[i]->IsSlave()))
      dserror("ERROR: IntegrateAndDerivSegment called on a wrong type of MortarElement pair!");
  }

  //**************************************************************
  //                loop over all Slave nodes
  //**************************************************************
  for(int snodes = 0; snodes<sele.NumNode() ;++snodes)
  {
    CoNode* mynode = dynamic_cast<CoNode*>(sele.Nodes()[snodes]);
    bool kink_projection=false;

    // skip this node if already considered
    if (mynode->MoData().GetD().size()!=0)
      continue;

    double area = 0.0;
    for (int ele=0;ele<mynode->NumElement();++ele)
      area+=dynamic_cast<CONTACT::CoElement*>(mynode->Elements()[ele])->MoData().Area();

    area=area/mynode->NumElement();

    double sxi[2] = {0.0, 0.0};

    if(sele.Shape() == DRT::Element::quad4 or
       sele.Shape() == DRT::Element::quad8 or
       sele.Shape() == DRT::Element::quad9)
    {
      if(snodes==0)       {sxi[0] = -1; sxi[1] = -1;}
      else if(snodes==1)  {sxi[0] =  1; sxi[1] = -1;}
      else if(snodes==2)  {sxi[0] =  1; sxi[1] =  1;}
      else if(snodes==3)  {sxi[0] = -1; sxi[1] =  1;}
      else if(snodes==4)  {sxi[0] =  0; sxi[1] = -1;}
      else if(snodes==5)  {sxi[0] =  1; sxi[1] =  0;}
      else if(snodes==6)  {sxi[0] =  0; sxi[1] =  1;}
      else if(snodes==7)  {sxi[0] = -1; sxi[1] =  0;}
      else if(snodes==8)  {sxi[0] =  0; sxi[1] =  0;}
      else dserror("ERORR: wrong node LID");
    }
    else if (sele.Shape() == DRT::Element::tri3 or
             sele.Shape() == DRT::Element::tri6)
    {
      if(snodes==0)       {sxi[0] = 0;    sxi[1] = 0;}
      else if(snodes==1)  {sxi[0] = 1;    sxi[1] = 0;}
      else if(snodes==2)  {sxi[0] = 0;    sxi[1] = 1;}
      else if(snodes==3)  {sxi[0] = 0.5;  sxi[1] = 0;}
      else if(snodes==4)  {sxi[0] = 0.5;  sxi[1] = 0.5;}
      else if(snodes==5)  {sxi[0] = 0;    sxi[1] = 0.5;}
      else dserror("ERORR: wrong node LID");
    }
    else
    {
      dserror("ERROR: Chosen element type not supported for nts!");
    }
    //**************************************************************
    //                loop over all Master Elements
    //**************************************************************
    for (int nummaster=0;nummaster<(int)meles.size();++nummaster)
    {
      // project Gauss point onto master element
      double mxi[2]    = {0.0, 0.0};
      double projalpha = 0.0;
      MORTAR::MortarProjector::Impl(sele,*meles[nummaster])->ProjectGaussPoint3D(
          sele,sxi,*meles[nummaster],mxi,projalpha);

      bool is_on_mele = true;

      // check GP projection
      DRT::Element::DiscretizationType dt = meles[nummaster]->Shape();
      const double tol = 0.00;
      if (dt==DRT::Element::quad4 || dt==DRT::Element::quad8 || dt==DRT::Element::quad9)
      {
        if (mxi[0]<-1.0-tol || mxi[1]<-1.0-tol || mxi[0]>1.0+tol || mxi[1]>1.0+tol)
        {
          is_on_mele=false;
        }
      }
      else
      {
        if (mxi[0]<-tol || mxi[1]<-tol || mxi[0]>1.0+tol || mxi[1]>1.0+tol || mxi[0]+mxi[1]>1.0+2*tol)
        {
          is_on_mele=false;
        }
      }

      // node on mele?
      if ((kink_projection==false) && (is_on_mele))
      {
        kink_projection   = true;
        mynode->HasProj() = true;

        int ndof = 3;
        int ncol = meles[nummaster]->NumNode();
        LINALG::SerialDenseVector mval(ncol);
        LINALG::SerialDenseMatrix mderiv(ncol,2);
        meles[nummaster]->EvaluateShape(mxi,mval,mderiv,ncol,false);

        // get slave and master nodal coords for Jacobian / GP evaluation
        LINALG::SerialDenseMatrix scoord(3,sele.NumNode());
        LINALG::SerialDenseMatrix mcoord(3,ncol);
        sele.GetNodalCoords(scoord);
        meles[nummaster]->GetNodalCoords(mcoord);

        // nodal coords from previous time step and lagrange mulitplier
        Teuchos::RCP<LINALG::SerialDenseMatrix> scoordold;
        Teuchos::RCP<LINALG::SerialDenseMatrix> mcoordold;
        Teuchos::RCP<LINALG::SerialDenseMatrix> lagmult;

        scoordold = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,sele.NumNode()));
        mcoordold = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,ncol));
        lagmult   = Teuchos::rcp(new LINALG::SerialDenseMatrix(3,sele.NumNode()));
        sele.GetNodalCoordsOld(*scoordold);
        meles[nummaster]->GetNodalCoordsOld(*mcoordold);
        sele.GetNodalLagMult(*lagmult);

        int linsize    = mynode->GetLinsize();
        double gpn[3]  = {0.0, 0.0, 0.0};
        //**************************************************************

        // evalute the GP slave coordinate derivatives --> no entries
        std::vector<GEN::pairedvector<int,double> > dsxi(2,0);
        std::vector<GEN::pairedvector<int,double> > dmxi(2,4*linsize+ncol*ndof);
        DerivXiGP3D(sele,*meles[nummaster],sxi,mxi, dsxi, dmxi, projalpha);

        // calculate node-wise DM
        nwDM3D(*mynode,*meles[nummaster],mval,mderiv,dmxi);

        // calculate node-wise un-weighted gap
        nwGap3D(*mynode, *meles[nummaster], mval, mderiv, dmxi, gpn);

      }//End hit ele
    }//End Loop over all Master Elements
  }
  //**************************************************************

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise TE for primary variable wear                  farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwTE2D(CoNode& mynode,
                                   double& area,
                                   double& jumpval,
                                   GEN::pairedvector<int,double>& dslipmatrix)
{
  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  // multiply the two shape functions
  double prod1 = abs(jumpval);
  double prod2 = 1.0*area;

  int col = mynode.Dofs()[0];
  int row = 0;

  if(abs(prod1)>MORTARINTTOL) dynamic_cast<CONTACT::FriNode&>(mynode).AddTValue(row,col,prod1);
  if(abs(prod2)>MORTARINTTOL) dynamic_cast<CONTACT::FriNode&>(mynode).AddEValue(row,col,prod2);

  std::map<int,double>& tmmap_jk =
      dynamic_cast<CONTACT::FriNode&>(mynode).FriDataPlus().GetDerivTw()[mynode.Id()];

  if(!sswear_)
  {
    double fac = 1.0;
    for (_CI p=dslipmatrix.begin(); p!=dslipmatrix.end(); ++p)
      tmmap_jk[p->first] += fac*(p->second);
  }
  return;
}


/*----------------------------------------------------------------------*
 |  node-wise slip                                          farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwSlip2D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   LINALG::SerialDenseMatrix& scoord,
                                   LINALG::SerialDenseMatrix& mcoord,
                                   Teuchos::RCP<LINALG::SerialDenseMatrix> scoordold,
                                   Teuchos::RCP<LINALG::SerialDenseMatrix> mcoordold,
                                   int& snodes,
                                   int& linsize,
                                   GEN::pairedvector<int,double>& dmxi)
{
  const int ncol = mele.NumNode();
  const int ndof = mynode.NumDof();

  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  GEN::pairedvector<int,double> dslipgp(linsize+ndof*ncol);

  // LIN OF TANGENT
  GEN::pairedvector<int,double> dmap_txsl_gp(ncol*ndof + linsize);
  GEN::pairedvector<int,double> dmap_tysl_gp(ncol*ndof + linsize);

  // build interpolation of slave GP normal and coordinates
  double sjumpv[3] = {0.0, 0.0, 0.0};
  double mjumpv[3] = {0.0, 0.0, 0.0};
  double jumpv[3]  = {0.0, 0.0, 0.0};
  double tanv[3]   = {0.0, 0.0, 0.0};

  double tanlength = 0.0;
  double pwjump = 0.0;

   //nodal tangent interpolation
   tanv[0]+=mynode.CoData().txi()[0];
   tanv[1]+=mynode.CoData().txi()[1];
   tanv[2]+=mynode.CoData().txi()[2];

   // delta D
   sjumpv[0]+=(scoord(0,snodes)-(*scoordold)(0,snodes));
   sjumpv[1]+=(scoord(1,snodes)-(*scoordold)(1,snodes));
   sjumpv[2]+=(scoord(2,snodes)-(*scoordold)(2,snodes));


  for (int i=0;i<ncol;++i)
  {
    mjumpv[0]+=mval[i]*(mcoord(0,i)-(*mcoordold)(0,i));
    mjumpv[1]+=mval[i]*(mcoord(1,i)-(*mcoordold)(1,i));
    mjumpv[2]+=mval[i]*(mcoord(2,i)-(*mcoordold)(2,i));
  }

  // normalize interpolated GP tangent back to length 1.0 !!!
  tanlength = sqrt(tanv[0]*tanv[0]+tanv[1]*tanv[1]+tanv[2]*tanv[2]);
  if (tanlength<1.0e-12) dserror("ERROR: IntegrateAndDerivSegment: Divide by zero!");

  for (int i=0;i<3;i++)
    tanv[i]/=tanlength;

  // jump
  jumpv[0] = sjumpv[0] - mjumpv[0];
  jumpv[1] = sjumpv[1] - mjumpv[1];
  jumpv[2] = sjumpv[2] - mjumpv[2];

  //multiply with tangent
  // value of relative tangential jump
  for (int i=0;i<3;++i)
    pwjump += tanv[i]*jumpv[i];


  // *****************************************************************************
  // add everything to dslipgp                                                   *
  // *****************************************************************************
  GEN::pairedvector<int,double>& dmap_txsl_i = mynode.CoData().GetDerivTxi()[0];
  GEN::pairedvector<int,double>& dmap_tysl_i = mynode.CoData().GetDerivTxi()[1];

  for (_CI p=dmap_txsl_i.begin();p!=dmap_txsl_i.end();++p)
    dmap_txsl_gp[p->first] += 1.0*(p->second);
  for (_CI p=dmap_tysl_i.begin();p!=dmap_tysl_i.end();++p)
    dmap_tysl_gp[p->first] += 1.0*(p->second);

  // build directional derivative of slave GP tagent (unit)
  GEN::pairedvector<int,double> dmap_txsl_gp_unit(ncol*ndof + linsize);
  GEN::pairedvector<int,double> dmap_tysl_gp_unit(ncol*ndof + linsize);

  const double llv     = tanlength*tanlength;
  const double linv    = 1.0/tanlength;
  const double lllinv  = 1.0/(tanlength*tanlength*tanlength);
  const double sxsxv   = tanv[0]*tanv[0]*llv;
  const double sxsyv   = tanv[0]*tanv[1]*llv;
  const double sysyv   = tanv[1]*tanv[1]*llv;

  for (_CI p=dmap_txsl_gp.begin();p!=dmap_txsl_gp.end();++p)
  {
    dmap_txsl_gp_unit[p->first] += linv*(p->second);
    dmap_txsl_gp_unit[p->first] -= lllinv*sxsxv*(p->second);
    dmap_tysl_gp_unit[p->first] -= lllinv*sxsyv*(p->second);
  }

  for (_CI p=dmap_tysl_gp.begin();p!=dmap_tysl_gp.end();++p)
  {
    dmap_tysl_gp_unit[p->first] += linv*(p->second);
    dmap_tysl_gp_unit[p->first] -= lllinv*sysyv*(p->second);
    dmap_txsl_gp_unit[p->first] -= lllinv*sxsyv*(p->second);
  }

  for (_CI p=dmap_txsl_gp_unit.begin();p!=dmap_txsl_gp_unit.end();++p)
    dslipgp[p->first] += jumpv[0] * (p->second);

  for (_CI p=dmap_tysl_gp_unit.begin();p!=dmap_tysl_gp_unit.end();++p)
    dslipgp[p->first] += jumpv[1] * (p->second);

  //coord lin
  for (int k=0;k<2;++k)
  {
    dslipgp[mynode.Dofs()[k]] += tanv[k];
  }


  for (int z=0;z<ncol;++z)
  {
    CONTACT::CoNode* mnode = dynamic_cast<CONTACT::CoNode*>(mele.Nodes()[z]);
    for (int k=0;k<2;++k)
    {
      dslipgp[mnode->Dofs()[k]] -= mval[z] * tanv[k];

      for (_CI p=dmxi.begin();p!=dmxi.end();++p)
        dslipgp[p->first] -= tanv[k] * mderiv(z,0) * (mcoord(k,z)-(*mcoordold)(k,z)) * (p->second);
    }
  }

  // ***************************
  // Add to node!
   double prod = pwjump;

  // add current Gauss point's contribution to jump
  dynamic_cast<CONTACT::FriNode&>(mynode).AddJumpValue(prod,0);

  // get the corresponding map as a reference
  std::map<int,double>& djumpmap =
      dynamic_cast<CONTACT::FriNode&>(mynode).FriData().GetDerivVarJump()[0];

  double fac = 1.0;
  for (_CI p=dslipgp.begin();p!=dslipgp.end();++p)
    djumpmap[p->first] += fac*(p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwWear2D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   LINALG::SerialDenseMatrix& scoord,
                                   LINALG::SerialDenseMatrix& mcoord,
                                   Teuchos::RCP<LINALG::SerialDenseMatrix> scoordold,
                                   Teuchos::RCP<LINALG::SerialDenseMatrix> mcoordold,
                                   Teuchos::RCP<LINALG::SerialDenseMatrix> lagmult,
                                   int& snodes,
                                   int& linsize,
                                   double& jumpval,
                                   double& area,
                                   double* gpn,
                                   GEN::pairedvector<int,double>& dmxi,
                                   GEN::pairedvector<int,double>& dslipmatrix,
                                   GEN::pairedvector<int,double>& dwear)
{
  const int ncol = mele.NumNode();
  const int ndof = mynode.NumDof();

  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  double gpt[3]     = {0.0, 0.0, 0.0};
  double gplm[3]    = {0.0, 0.0, 0.0};
  double sgpjump[3] = {0.0, 0.0, 0.0};
  double mgpjump[3] = {0.0, 0.0, 0.0};
  double jump[3]    = {0.0, 0.0, 0.0};

  // for linearization
  double lm_lin  = 0.0;
  double lengtht = 0.0;
  double wearval = 0.0;

  //nodal tangent interpolation
  gpt[0]+=mynode.CoData().txi()[0];
  gpt[1]+=mynode.CoData().txi()[1];
  gpt[2]+=mynode.CoData().txi()[2];

  // delta D
  sgpjump[0]+=(scoord(0,snodes)-((*scoordold)(0,snodes)));
  sgpjump[1]+=(scoord(1,snodes)-((*scoordold)(1,snodes)));
  sgpjump[2]+=(scoord(2,snodes)-((*scoordold)(2,snodes)));

  // LM interpolation
  gplm[0]+=((*lagmult)(0,snodes));
  gplm[1]+=((*lagmult)(1,snodes));
  gplm[2]+=((*lagmult)(2,snodes));

  // normalize interpolated GP tangent back to length 1.0 !!!
  lengtht = sqrt(gpt[0]*gpt[0]+gpt[1]*gpt[1]+gpt[2]*gpt[2]);
  if (abs(lengtht)<1.0e-12) dserror("ERROR: IntegrateAndDerivSegment: Divide by zero!");

  for (int i=0;i<3;i++)
    gpt[i]/=lengtht;

  // interpolation of master GP jumps (relative displacement increment)
  for (int i=0;i<ncol;++i)
  {
    mgpjump[0]+=mval[i]*(mcoord(0,i)-(*mcoordold)(0,i));
    mgpjump[1]+=mval[i]*(mcoord(1,i)-(*mcoordold)(1,i));
    mgpjump[2]+=mval[i]*(mcoord(2,i)-(*mcoordold)(2,i));
  }

  // jump
  jump[0] = sgpjump[0] - mgpjump[0];
  jump[1] = sgpjump[1] - mgpjump[1];
  jump[2] = sgpjump[2] - mgpjump[2];

  // evaluate wear
  // normal contact stress -- normal LM value
  for (int i=0;i<2;++i)
  {
    wearval += gpn[i]*gplm[i];
    lm_lin  += gpn[i]*gplm[i];  // required for linearization
  }

  // value of relative tangential jump
  for (int i=0;i<3;++i)
    jumpval+=gpt[i]*jump[i];

  if(sswear_)
    jumpval = ssslip_;

  // no jump --> no wear
  if (abs(jumpval)<1e-12)
    return;

  // product
  // use non-abs value for implicit-wear algorithm
  // just for simple linear. maybe we change this in future
  wearval = abs(wearval)*abs(jumpval);

  double prod = wearval/area;

  // add current node wear to w
  dynamic_cast<CONTACT::FriNode&>(mynode).AddDeltaWearValue(prod);

  //****************************************************************
  //   linearization for implicit algorithms
  //****************************************************************
  if((wearimpl_ || weartype_ == INPAR::CONTACT::wear_discr) and  abs(jumpval)>1e-12)
  {
    // lin. abs(x) = x/abs(x) * lin x.
    double xabsx  = (jumpval/abs(jumpval)) * lm_lin;
    double xabsxT = (jumpval/abs(jumpval));

    // **********************************************************************
    // (1) Lin of normal for LM -- deriv normal maps from weighted gap lin.
    for (_CI p=mynode.CoData().GetDerivN()[0].begin();p!=mynode.CoData().GetDerivN()[0].end();++p)
      dwear[p->first] += abs(jumpval) * gplm[0] * (p->second);

    for (_CI p=mynode.CoData().GetDerivN()[1].begin();p!=mynode.CoData().GetDerivN()[1].end();++p)
      dwear[p->first] += abs(jumpval) * gplm[1] * (p->second);

    // **********************************************************************
    // (3) absolute incremental slip linearization:
    // (a) build directional derivative of slave GP tagent (non-unit)
    GEN::pairedvector<int,double> dmap_txsl_gp(ndof*ncol + linsize);
    GEN::pairedvector<int,double> dmap_tysl_gp(ndof*ncol + linsize);

    GEN::pairedvector<int,double>& dmap_txsl_i = mynode.CoData().GetDerivTxi()[0];
    GEN::pairedvector<int,double>& dmap_tysl_i = mynode.CoData().GetDerivTxi()[1];

    for (_CI p=dmap_txsl_i.begin();p!=dmap_txsl_i.end();++p)
      dmap_txsl_gp[p->first] += (p->second);
    for (_CI p=dmap_tysl_i.begin();p!=dmap_tysl_i.end();++p)
      dmap_tysl_gp[p->first] += (p->second);

    // (b) build directional derivative of slave GP tagent (unit)
    GEN::pairedvector<int,double> dmap_txsl_gp_unit(ndof*ncol + linsize);
    GEN::pairedvector<int,double> dmap_tysl_gp_unit(ndof*ncol + linsize);

    const double ll     = lengtht*lengtht;
    const double linv   = 1.0/lengtht;
    const double lllinv = 1.0/(lengtht*lengtht*lengtht);
    const double sxsx   = gpt[0]*gpt[0]*ll;
    const double sxsy   = gpt[0]*gpt[1]*ll;
    const double sysy   = gpt[1]*gpt[1]*ll;

    for (_CI p=dmap_txsl_gp.begin();p!=dmap_txsl_gp.end();++p)
    {
      dmap_txsl_gp_unit[p->first] += linv*(p->second);
      dmap_txsl_gp_unit[p->first] -= lllinv*sxsx*(p->second);
      dmap_tysl_gp_unit[p->first] -= lllinv*sxsy*(p->second);
    }

    for (_CI p=dmap_tysl_gp.begin();p!=dmap_tysl_gp.end();++p)
    {
      dmap_tysl_gp_unit[p->first] += linv*(p->second);
      dmap_tysl_gp_unit[p->first] -= lllinv*sysy*(p->second);
      dmap_txsl_gp_unit[p->first] -= lllinv*sxsy*(p->second);
    }

    // add tangent lin. to dweargp
    for (_CI p=dmap_txsl_gp_unit.begin();p!=dmap_txsl_gp_unit.end();++p)
      dwear[p->first] += xabsx * jump[0] * (p->second);

    for (_CI p=dmap_tysl_gp_unit.begin();p!=dmap_tysl_gp_unit.end();++p)
      dwear[p->first] += xabsx * jump[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (_CI p=dmap_txsl_gp_unit.begin();p!=dmap_txsl_gp_unit.end();++p)
      dslipmatrix[p->first] += xabsxT * jump[0] * (p->second);

    for (_CI p=dmap_tysl_gp_unit.begin();p!=dmap_tysl_gp_unit.end();++p)
      dslipmatrix[p->first] += xabsxT * jump[1] * (p->second);

    // **********************************************************************
    // (c) build directional derivative of jump
    GEN::pairedvector<int,double> dmap_slcoord_gp_x(ndof*ncol + linsize);
    GEN::pairedvector<int,double> dmap_slcoord_gp_y(ndof*ncol + linsize);

    GEN::pairedvector<int,double> dmap_mcoord_gp_x(ndof*ncol + linsize);
    GEN::pairedvector<int,double> dmap_mcoord_gp_y(ndof*ncol + linsize);

    GEN::pairedvector<int,double> dmap_coord_x(ndof*ncol + linsize);
    GEN::pairedvector<int,double> dmap_coord_y(ndof*ncol + linsize);

    // lin master part -- mxi
    for (int i=0;i<ncol;++i)
    {
      for (_CI p=dmxi.begin();p!=dmxi.end();++p)
      {
        double valx =  mderiv(i,0) * ( mcoord(0,i)-((*mcoordold)(0,i)) );
        dmap_mcoord_gp_x[p->first] += valx*(p->second);
        double valy =  mderiv(i,0) * ( mcoord(1,i)-((*mcoordold)(1,i)) );
        dmap_mcoord_gp_y[p->first] += valy*(p->second);
      }
    }

    // deriv slave x-coords
    dmap_slcoord_gp_x[mynode.Dofs()[0]]+=1.0;
    dmap_slcoord_gp_y[mynode.Dofs()[1]]+=1.0;

    // deriv master x-coords
    for (int i=0;i<ncol;++i)
    {
      MORTAR::MortarNode* mnode = dynamic_cast<MORTAR::MortarNode*> (mele.Nodes()[i]);

      dmap_mcoord_gp_x[mnode->Dofs()[0]]+=mval[i];
      dmap_mcoord_gp_y[mnode->Dofs()[1]]+=mval[i];
    }

    //slave: add to jumplin
    for (_CI p=dmap_slcoord_gp_x.begin();p!=dmap_slcoord_gp_x.end();++p)
      dmap_coord_x[p->first] += (p->second);
    for (_CI p=dmap_slcoord_gp_y.begin();p!=dmap_slcoord_gp_y.end();++p)
      dmap_coord_y[p->first] += (p->second);

    //master: add to jumplin
    for (_CI p=dmap_mcoord_gp_x.begin();p!=dmap_mcoord_gp_x.end();++p)
      dmap_coord_x[p->first] -= (p->second);
    for (_CI p=dmap_mcoord_gp_y.begin();p!=dmap_mcoord_gp_y.end();++p)
      dmap_coord_y[p->first] -= (p->second);

    // add to dweargp
    for (_CI p=dmap_coord_x.begin();p!=dmap_coord_x.end();++p)
      dwear[p->first] += xabsx * gpt[0] * (p->second);

    for (_CI p=dmap_coord_y.begin();p!=dmap_coord_y.end();++p)
      dwear[p->first] += xabsx * gpt[1] * (p->second);

    // add tangent lin. to slip linearization for wear Tmatrix
    for (_CI p=dmap_coord_x.begin();p!=dmap_coord_x.end();++p)
      dslipmatrix[p->first] += xabsxT * gpt[0] * (p->second);

    for (_CI p=dmap_coord_y.begin();p!=dmap_coord_y.end();++p)
      dslipmatrix[p->first] += xabsxT * gpt[1] * (p->second);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwGap2D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   GEN::pairedvector<int,double>& dmxi,
                                   double* gpn)
{
  const int ncol = mele.NumNode();

  double sgpx[3] = {0.0, 0.0, 0.0};
  double mgpx[3] = {0.0, 0.0, 0.0};

  gpn[0]+=mynode.MoData().n()[0];
  gpn[1]+=mynode.MoData().n()[1];
  gpn[2]+=mynode.MoData().n()[2];

  sgpx[0]+=mynode.xspatial()[0];
  sgpx[1]+=mynode.xspatial()[1];
  sgpx[2]+=mynode.xspatial()[2];

  // build interpolation of master GP coordinates
  for (int i=0;i<ncol;++i)
  {
    CONTACT::CoNode* mnode = dynamic_cast<CONTACT::CoNode*>(mele.Nodes()[i]);

    mgpx[0]+=mval[i]*mnode->xspatial()[0];
    mgpx[1]+=mval[i]*mnode->xspatial()[1];
    mgpx[2]+=mval[i]*mnode->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0]*gpn[0]+gpn[1]*gpn[1]+gpn[2]*gpn[2]);
  if (lengthn<1.0e-12) dserror("ERROR: Divide by zero!");

  for (int i=0;i<3;++i)
    gpn[i]/=lengthn;

  // build gap function at current GP
  double gap = 0.0;
  for (int i=0;i<2;++i)
    gap += (mgpx[i]-sgpx[i])*gpn[i];

  // **************************
  // add to node
  // **************************
  mynode.AddgValue(gap);

  // **************************
  // linearization
  // **************************
  typedef GEN::pairedvector<int,double>::const_iterator _CI;
  GEN::pairedvector<int,double> dgapgp(10*ncol);

  //*************************************************************
  for (_CI p=mynode.CoData().GetDerivN()[0].begin();p!=mynode.CoData().GetDerivN()[0].end();++p)
    dgapgp[p->first] += (mgpx[0]-sgpx[0]) * (p->second);

  for (_CI p=mynode.CoData().GetDerivN()[1].begin();p!=mynode.CoData().GetDerivN()[1].end();++p)
    dgapgp[p->first] += (mgpx[1]-sgpx[1]) * (p->second);


  for (int k=0;k<2;++k)
  {
    dgapgp[mynode.Dofs()[k]] -= (gpn[k]);
  }

  for (int z=0;z<ncol;++z)
  {
    MORTAR::MortarNode* mnode = dynamic_cast<MORTAR::MortarNode*> (mele.Nodes()[z]);

    for (int k=0;k<2;++k)
    {
      dgapgp[mnode->Dofs()[k]] += mval[z] * gpn[k];

      for (_CI p=dmxi.begin();p!=dmxi.end();++p)
        dgapgp[p->first] += gpn[k] * mderiv(z,0) * mnode->xspatial()[k] * (p->second);
    }
  }

  std::map<int,double>& dgmap = mynode.CoData().GetDerivG();

  // (1) Lin(g) - gap function
  double fac = 1.0;
  for (_CI p=dgapgp.begin();p!=dgapgp.end();++p)
    dgmap[p->first] += fac*(p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise un-weighted gap                               farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwGap3D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   std::vector<GEN::pairedvector<int,double> >& dmxi,
                                   double* gpn)
{
  const int ncol = mele.NumNode();

  double sgpx[3] = {0.0, 0.0, 0.0};
  double mgpx[3] = {0.0, 0.0, 0.0};

  gpn[0]+=mynode.MoData().n()[0];
  gpn[1]+=mynode.MoData().n()[1];
  gpn[2]+=mynode.MoData().n()[2];

  sgpx[0]+=mynode.xspatial()[0];
  sgpx[1]+=mynode.xspatial()[1];
  sgpx[2]+=mynode.xspatial()[2];

  // build interpolation of master GP coordinates
  for (int i=0;i<ncol;++i)
  {
    CONTACT::CoNode* mnode = dynamic_cast<CONTACT::CoNode*>(mele.Nodes()[i]);

    mgpx[0]+=mval[i]*mnode->xspatial()[0];
    mgpx[1]+=mval[i]*mnode->xspatial()[1];
    mgpx[2]+=mval[i]*mnode->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0]*gpn[0]+gpn[1]*gpn[1]+gpn[2]*gpn[2]);
  if (lengthn<1.0e-12) dserror("ERROR: Divide by zero!");

  for (int i=0;i<3;++i)
    gpn[i]/=lengthn;

  // build gap function at current GP
  double gap = 0.0;
  for (int i=0;i<3;++i)
    gap += (mgpx[i]-sgpx[i])*gpn[i];

  // **************************
  // add to node
  // **************************
  mynode.AddgValue(gap);

  // **************************
  // linearization
  // **************************
  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  //TODO: linsize wor parallel simulations buggy. 100 for safety
  GEN::pairedvector<int,double> dgapgp(3*ncol+3*mynode.GetLinsize()+100);

  //*************************************************************
  for (_CI p=mynode.CoData().GetDerivN()[0].begin();p!=mynode.CoData().GetDerivN()[0].end();++p)
    dgapgp[p->first] += (mgpx[0]-sgpx[0]) * (p->second);

  for (_CI p=mynode.CoData().GetDerivN()[1].begin();p!=mynode.CoData().GetDerivN()[1].end();++p)
    dgapgp[p->first] += (mgpx[1]-sgpx[1]) * (p->second);

  for (_CI p=mynode.CoData().GetDerivN()[2].begin();p!=mynode.CoData().GetDerivN()[2].end();++p)
    dgapgp[p->first] += (mgpx[2]-sgpx[2]) * (p->second);


  for (int k=0;k<3;++k)
  {
    dgapgp[mynode.Dofs()[k]] -= (gpn[k]);
  }

  for (int z=0;z<ncol;++z)
  {
    MORTAR::MortarNode* mnode = dynamic_cast<MORTAR::MortarNode*> (mele.Nodes()[z]);

    for (int k=0;k<3;++k)
    {
      dgapgp[mnode->Dofs()[k]] += mval[z] * gpn[k];

      for (_CI p=dmxi[0].begin();p!=dmxi[0].end();++p)
        dgapgp[p->first] += gpn[k] * mderiv(z,0) * mnode->xspatial()[k] * (p->second);

      for (_CI p=dmxi[1].begin();p!=dmxi[1].end();++p)
        dgapgp[p->first] += gpn[k] * mderiv(z,1) * mnode->xspatial()[k] * (p->second);
    }
  }

  std::map<int,double>& dgmap = mynode.CoData().GetDerivG();

  // (1) Lin(g) - gap function
  double fac = 1.0;
  for (_CI p=dgapgp.begin();p!=dgapgp.end();++p)
    dgmap[p->first] += fac*(p->second);

  return;
}


/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwDM2D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   GEN::pairedvector<int,double>& dmxi)
{
  const int ncol = mele.NumNode();
  const int ndof = mynode.NumDof();

  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  // node-wise M value
  for (int k=0; k<ncol; ++k)
  {
    CONTACT::CoNode* mnode = dynamic_cast<CONTACT::CoNode*>(mele.Nodes()[k]);

    // multiply the two shape functions
    double prod = mval[k];

    //loop over slave dofs
    for (int jdof=0;jdof<ndof;++jdof)
    {
      int col = mnode->Dofs()[jdof];
      if(abs(prod)>MORTARINTTOL) mynode.AddMValue(jdof,col,prod);
      if(abs(prod)>MORTARINTTOL) mynode.AddMNode(mnode->Id());  // only for friction!
    }
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;

  //loop over slave dofs
  for (int jdof=0;jdof<ndof;++jdof)
  {
    int col = mynode.Dofs()[jdof];
    if(abs(prod)>MORTARINTTOL) mynode.AddDValue(jdof,col,prod);
    if(abs(prod)>MORTARINTTOL) mynode.AddSNode(mynode.Id()); // only for friction!
  }

  // integrate LinM
  for (int k=0; k<ncol; ++k)
  {
    // global master node ID
    int mgid = mele.Nodes()[k]->Id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int,double>& dmmap_jk = mynode.CoData().GetDerivM()[mgid];

    // (3) Lin(NMaster) - master GP coordinates
    fac = mderiv(k, 0);
    for (_CI p=dmxi.begin(); p!=dmxi.end(); ++p)
      dmmap_jk[p->first] += fac*(p->second);
  } // loop over master nodes

  return;
}

/*----------------------------------------------------------------------*
 |  node-wise D/M calculation                               farah 09/14 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::nwDM3D(CoNode& mynode,
                                   MORTAR::MortarElement& mele,
                                   LINALG::SerialDenseVector& mval,
                                   LINALG::SerialDenseMatrix& mderiv,
                                   std::vector<GEN::pairedvector<int,double> >& dmxi)
{
  const int ncol = mele.NumNode();
  const int ndof = mynode.NumDof();

  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  // node-wise M value
  for (int k=0; k<ncol; ++k)
  {
    CONTACT::CoNode* mnode = dynamic_cast<CONTACT::CoNode*>(mele.Nodes()[k]);

    // multiply the two shape functions
    double prod = mval[k];

    //loop over slave dofs
    for (int jdof=0;jdof<ndof;++jdof)
    {
      int col = mnode->Dofs()[jdof];
      if(abs(prod)>MORTARINTTOL) mynode.AddMValue(jdof,col,prod);
      if(abs(prod)>MORTARINTTOL) mynode.AddMNode(mnode->Id());  // only for friction!
    }
  }

  // integrate dseg
  // multiply the two shape functions
  double prod = 1.0;

  //loop over slave dofs
  for (int jdof=0;jdof<ndof;++jdof)
  {
    int col = mynode.Dofs()[jdof];
    if(abs(prod)>MORTARINTTOL) mynode.AddDValue(jdof,col,prod);
    if(abs(prod)>MORTARINTTOL) mynode.AddSNode(mynode.Id()); // only for friction!
  }

  // integrate LinM
  for (int k=0; k<ncol; ++k)
  {
    // global master node ID
    int mgid = mele.Nodes()[k]->Id();
    double fac = 0.0;

    // get the correct map as a reference
    std::map<int,double>& dmmap_jk = mynode.CoData().GetDerivM()[mgid];

    fac = mderiv(k, 0);
    for (_CI p=dmxi[0].begin(); p!=dmxi[0].end(); ++p)
      dmmap_jk[p->first] += fac*(p->second);

    fac = mderiv(k, 1);
    for (_CI p=dmxi[1].begin(); p!=dmxi[1].end(); ++p)
      dmmap_jk[p->first] += fac*(p->second);
  } // loop over master nodes

  return;
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP master (2D)       popp 05/08 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::DerivXiGP2D(MORTAR::MortarElement& sele,
                                    MORTAR::MortarElement& mele,
                                    double& sxigp, double& mxigp,
                                    const GEN::pairedvector<int,double>& derivsxi,
                                    GEN::pairedvector<int,double>& derivmxi,
                                    int& linsize)
{
  //check for problem dimension

  // we need the participating slave and master nodes
  DRT::Node** snodes = NULL;
  DRT::Node** mnodes = NULL;
  DRT::Node* hsnodes[4] = {0,0,0,0};
  DRT::Node* hmnodes[4] = {0,0,0,0};
  int numsnode = sele.NumNode();
  int nummnode = mele.NumNode();

  int ndof     = 2;
  if(sele.IsHermite())
  {
    int sfeatures[2] = {0,0};
    sele.AdjEleStatus(sfeatures);
    numsnode = sfeatures[1];
    sele.HermitEleNodes(hsnodes, sfeatures[0]);
    snodes=hsnodes;
  }
  else
    snodes = sele.Nodes();

  if(mele.IsHermite())
  {
    int mfeatures[2] = {0,0};
    mele.AdjEleStatus(mfeatures);
    nummnode = mfeatures[1];
    mele.HermitEleNodes(hmnodes, mfeatures[0]);
    mnodes=hmnodes;
  }
  else
    mnodes=mele.Nodes();

  std::vector<MORTAR::MortarNode*> smrtrnodes(numsnode);
  std::vector<MORTAR::MortarNode*> mmrtrnodes(nummnode);

  for (int i=0;i<numsnode;++i)
  {
    smrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: DerivXiAB2D: Null pointer!");
  }

  for (int i=0;i<nummnode;++i)
  {
    mmrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(mnodes[i]);
    if (!mmrtrnodes[i]) dserror("ERROR: DerivXiAB2D: Null pointer!");
  }

  // we also need shape function derivs in A and B
  double psxigp[2] = {sxigp, 0.0};
  double pmxigp[2] = {mxigp, 0.0};
  LINALG::SerialDenseVector valsxigp(numsnode);
  LINALG::SerialDenseVector valmxigp(nummnode);
  LINALG::SerialDenseMatrix derivsxigp(numsnode,1);
  LINALG::SerialDenseMatrix derivmxigp(nummnode,1);

  sele.EvaluateShape(psxigp,valsxigp,derivsxigp,numsnode,false);
  mele.EvaluateShape(pmxigp,valmxigp,derivmxigp,nummnode,false);

  // we also need the GP slave coordinates + normal
  double sgpn[3] = {0.0,0.0,0.0};
  double sgpx[3] = {0.0,0.0,0.0};
  for (int i=0;i<numsnode;++i)
  {
    sgpn[0]+=valsxigp[i]*smrtrnodes[i]->MoData().n()[0];
    sgpn[1]+=valsxigp[i]*smrtrnodes[i]->MoData().n()[1];
    sgpn[2]+=valsxigp[i]*smrtrnodes[i]->MoData().n()[2];

    sgpx[0]+=valsxigp[i]*smrtrnodes[i]->xspatial()[0];
    sgpx[1]+=valsxigp[i]*smrtrnodes[i]->xspatial()[1];
    sgpx[2]+=valsxigp[i]*smrtrnodes[i]->xspatial()[2];
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  const double length = sqrt(sgpn[0]*sgpn[0]+sgpn[1]*sgpn[1]+sgpn[2]*sgpn[2]);
  if (length<1.0e-12) dserror("ERROR: DerivXiGP2D: Divide by zero!");
  for (int i=0;i<3;++i) sgpn[i]/=length;

  // compute factors and leading constants for master
  double cmxigp      = 0.0;
  double fac_dxm_gp  = 0.0;
  double fac_dym_gp  = 0.0;
  double fac_xmsl_gp = 0.0;
  double fac_ymsl_gp = 0.0;

  for (int i=0;i<nummnode;++i)
  {
    fac_dxm_gp += derivmxigp(i,0)*(mmrtrnodes[i]->xspatial()[0]);
    fac_dym_gp += derivmxigp(i,0)*(mmrtrnodes[i]->xspatial()[1]);

    fac_xmsl_gp += valmxigp[i]*(mmrtrnodes[i]->xspatial()[0]);
    fac_ymsl_gp += valmxigp[i]*(mmrtrnodes[i]->xspatial()[1]);
  }

  cmxigp = -1/(fac_dxm_gp*sgpn[1]-fac_dym_gp*sgpn[0]);
  //std::cout << "cmxigp: " << cmxigp << std::endl;

  fac_xmsl_gp -= sgpx[0];
  fac_ymsl_gp -= sgpx[1];

  // prepare linearization
  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  // build directional derivative of slave GP coordinates
  GEN::pairedvector<int,double> dmap_xsl_gp(linsize+nummnode*ndof);
  GEN::pairedvector<int,double> dmap_ysl_gp(linsize+nummnode*ndof);

  for (int i=0;i<numsnode;++i)
  {
    dmap_xsl_gp[smrtrnodes[i]->Dofs()[0]] += valsxigp[i];
    dmap_ysl_gp[smrtrnodes[i]->Dofs()[1]] += valsxigp[i];

    for (_CI p=derivsxi.begin();p!=derivsxi.end();++p)
    {
      double facx = derivsxigp(i,0)*(smrtrnodes[i]->xspatial()[0]);
      double facy = derivsxigp(i,0)*(smrtrnodes[i]->xspatial()[1]);
      dmap_xsl_gp[p->first] += facx*(p->second);
      dmap_ysl_gp[p->first] += facy*(p->second);
    }
  }

  // build directional derivative of slave GP normal
  GEN::pairedvector<int,double> dmap_nxsl_gp(linsize+nummnode*ndof);
  GEN::pairedvector<int,double> dmap_nysl_gp(linsize+nummnode*ndof);

  double sgpnmod[3] = {0.0,0.0,0.0};
  for (int i=0;i<3;++i) sgpnmod[i]=sgpn[i]*length;

  GEN::pairedvector<int,double> dmap_nxsl_gp_mod(linsize+nummnode*ndof);
  GEN::pairedvector<int,double> dmap_nysl_gp_mod(linsize+nummnode*ndof);

  for (int i=0;i<numsnode;++i)
  {
    GEN::pairedvector<int,double>& dmap_nxsl_i = dynamic_cast<CONTACT::CoNode*>(smrtrnodes[i])->CoData().GetDerivN()[0];
    GEN::pairedvector<int,double>& dmap_nysl_i = dynamic_cast<CONTACT::CoNode*>(smrtrnodes[i])->CoData().GetDerivN()[1];

    for (_CI p=dmap_nxsl_i.begin();p!=dmap_nxsl_i.end();++p)
      dmap_nxsl_gp_mod[p->first] += valsxigp[i]*(p->second);
    for (_CI p=dmap_nysl_i.begin();p!=dmap_nysl_i.end();++p)
      dmap_nysl_gp_mod[p->first] += valsxigp[i]*(p->second);

    for (_CI p=derivsxi.begin();p!=derivsxi.end();++p)
    {
      double valx =  derivsxigp(i,0)*smrtrnodes[i]->MoData().n()[0];
      dmap_nxsl_gp_mod[p->first] += valx*(p->second);
      double valy =  derivsxigp(i,0)*smrtrnodes[i]->MoData().n()[1];
      dmap_nysl_gp_mod[p->first] += valy*(p->second);
    }
  }

  const double sxsx   = sgpnmod[0]*sgpnmod[0];
  const double sxsy   = sgpnmod[0]*sgpnmod[1];
  const double sysy   = sgpnmod[1]*sgpnmod[1];
  const double linv   = 1.0/length;
  const double lllinv = 1.0/(length*length*length);

  for (_CI p=dmap_nxsl_gp_mod.begin();p!=dmap_nxsl_gp_mod.end();++p)
  {
    dmap_nxsl_gp[p->first] += linv*(p->second);
    dmap_nxsl_gp[p->first] -= lllinv*sxsx*(p->second);
    dmap_nysl_gp[p->first] -= lllinv*sxsy*(p->second);
  }

  for (_CI p=dmap_nysl_gp_mod.begin();p!=dmap_nysl_gp_mod.end();++p)
  {
    dmap_nysl_gp[p->first] += linv*(p->second);
    dmap_nysl_gp[p->first] -= lllinv*sysy*(p->second);
    dmap_nxsl_gp[p->first] -= lllinv*sxsy*(p->second);
  }

  // *********************************************************************
  // finally compute Lin(XiGP_master)
  // *********************************************************************

  // add derivative of slave GP coordinates
  for (_CI p=dmap_xsl_gp.begin();p!=dmap_xsl_gp.end();++p)
    derivmxi[p->first] -= sgpn[1]*(p->second);
  for (_CI p=dmap_ysl_gp.begin();p!=dmap_ysl_gp.end();++p)
    derivmxi[p->first] += sgpn[0]*(p->second);

  // add derivatives of master node coordinates
  for (int i=0;i<nummnode;++i)
  {
    derivmxi[mmrtrnodes[i]->Dofs()[0]] += valmxigp[i]*sgpn[1];
    derivmxi[mmrtrnodes[i]->Dofs()[1]] -= valmxigp[i]*sgpn[0];
  }

  // add derivative of slave GP normal
  for (_CI p=dmap_nxsl_gp.begin();p!=dmap_nxsl_gp.end();++p)
    derivmxi[p->first] -= fac_ymsl_gp*(p->second);
  for (_CI p=dmap_nysl_gp.begin();p!=dmap_nysl_gp.end();++p)
    derivmxi[p->first] += fac_xmsl_gp*(p->second);

  // multiply all entries with cmxigp
  for (_CI p=derivmxi.begin();p!=derivmxi.end();++p)
    derivmxi[p->first] = cmxigp*(p->second);

  return;
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of XiGP master (3D)        popp 02/09|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterpolator::DerivXiGP3D(MORTAR::MortarElement& sele,
                                      MORTAR::MortarElement& mele,
                                      double* sxigp, double* mxigp,
                                      const std::vector<GEN::pairedvector<int,double> >& derivsxi,
                                      std::vector<GEN::pairedvector<int,double> >& derivmxi,
                                      double& alpha)
{
  // we need the participating slave and master nodes
  DRT::Node** snodes = sele.Nodes();
  DRT::Node** mnodes = mele.Nodes();
  std::vector<MORTAR::MortarNode*> smrtrnodes(sele.NumNode());
  std::vector<MORTAR::MortarNode*> mmrtrnodes(mele.NumNode());
  const int numsnode = sele.NumNode();
  const int nummnode = mele.NumNode();

  for (int i=0;i<numsnode;++i)
  {
    smrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: DerivXiGP3D: Null pointer!");
  }

  for (int i=0;i<nummnode;++i)
  {
    mmrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(mnodes[i]);
    if (!mmrtrnodes[i]) dserror("ERROR: DerivXiGP3D: Null pointer!");
  }

  // we also need shape function derivs at the GP
  LINALG::SerialDenseVector valsxigp(numsnode);
  LINALG::SerialDenseVector valmxigp(nummnode);
  LINALG::SerialDenseMatrix derivsxigp(numsnode,2,true);
  LINALG::SerialDenseMatrix derivmxigp(nummnode,2,true);

  sele.EvaluateShape(sxigp,valsxigp,derivsxigp,numsnode);
  mele.EvaluateShape(mxigp,valmxigp,derivmxigp,nummnode);

  // we also need the GP slave coordinates + normal
  double sgpn[3] = {0.0,0.0,0.0};
  double sgpx[3] = {0.0,0.0,0.0};
  for (int i=0;i<numsnode;++i)
    for (int k=0;k<3;++k)
    {
      sgpn[k]+=valsxigp[i]*smrtrnodes[i]->MoData().n()[k];
      sgpx[k]+=valsxigp[i]*smrtrnodes[i]->xspatial()[k];
    }

  // build 3x3 factor matrix L
  LINALG::Matrix<3,3> lmatrix(true);
  for (int k=0;k<3;++k) lmatrix(k,2) = -sgpn[k];
  for (int z=0;z<nummnode;++z)
    for (int k=0;k<3;++k)
    {
      lmatrix(k,0) += derivmxigp(z,0) * mmrtrnodes[z]->xspatial()[k];
      lmatrix(k,1) += derivmxigp(z,1) * mmrtrnodes[z]->xspatial()[k];
    }

  // get inverse of the 3x3 matrix L (in place)
  if (abs(lmatrix.Determinant())<1e-12)
    dserror("ERROR: Singular lmatrix for derivgp3d");

  lmatrix.Invert();

  // build directional derivative of slave GP normal
  typedef GEN::pairedvector<int,double>::const_iterator _CI;

  int linsize = 0;
  for (int i=0;i<numsnode;++i)
  {
    CoNode* cnode = dynamic_cast<CoNode*> (snodes[i]);
    linsize += cnode->GetLinsize();
  }

  GEN::pairedvector<int,double> dmap_nxsl_gp(linsize);
  GEN::pairedvector<int,double> dmap_nysl_gp(linsize);
  GEN::pairedvector<int,double> dmap_nzsl_gp(linsize);

  for (int i=0;i<numsnode;++i)
  {
    GEN::pairedvector<int,double>& dmap_nxsl_i = dynamic_cast<CONTACT::CoNode*>(smrtrnodes[i])->CoData().GetDerivN()[0];
    GEN::pairedvector<int,double>& dmap_nysl_i = dynamic_cast<CONTACT::CoNode*>(smrtrnodes[i])->CoData().GetDerivN()[1];
    GEN::pairedvector<int,double>& dmap_nzsl_i = dynamic_cast<CONTACT::CoNode*>(smrtrnodes[i])->CoData().GetDerivN()[2];

    for (_CI p=dmap_nxsl_i.begin();p!=dmap_nxsl_i.end();++p)
      dmap_nxsl_gp[p->first] += valsxigp[i]*(p->second);
    for (_CI p=dmap_nysl_i.begin();p!=dmap_nysl_i.end();++p)
      dmap_nysl_gp[p->first] += valsxigp[i]*(p->second);
    for (_CI p=dmap_nzsl_i.begin();p!=dmap_nzsl_i.end();++p)
      dmap_nzsl_gp[p->first] += valsxigp[i]*(p->second);

    for (_CI p=derivsxi[0].begin();p!=derivsxi[0].end();++p)
    {
      double valx =  derivsxigp(i,0)*smrtrnodes[i]->MoData().n()[0];
      dmap_nxsl_gp[p->first] += valx*(p->second);
      double valy =  derivsxigp(i,0)*smrtrnodes[i]->MoData().n()[1];
      dmap_nysl_gp[p->first] += valy*(p->second);
      double valz =  derivsxigp(i,0)*smrtrnodes[i]->MoData().n()[2];
      dmap_nzsl_gp[p->first] += valz*(p->second);
    }

    for (_CI p=derivsxi[1].begin();p!=derivsxi[1].end();++p)
    {
      double valx =  derivsxigp(i,1)*smrtrnodes[i]->MoData().n()[0];
      dmap_nxsl_gp[p->first] += valx*(p->second);
      double valy =  derivsxigp(i,1)*smrtrnodes[i]->MoData().n()[1];
      dmap_nysl_gp[p->first] += valy*(p->second);
      double valz =  derivsxigp(i,1)*smrtrnodes[i]->MoData().n()[2];
      dmap_nzsl_gp[p->first] += valz*(p->second);
    }
  }

  // start to fill linearization maps for master GP
  // (1) all master nodes coordinates part
  for (int z=0;z<nummnode;++z)
  {
    for (int k=0;k<3;++k)
    {
      derivmxi[0][mmrtrnodes[z]->Dofs()[k]] -= valmxigp[z] * lmatrix(0,k);
      derivmxi[1][mmrtrnodes[z]->Dofs()[k]] -= valmxigp[z] * lmatrix(1,k);
    }
  }

  // (2) slave Gauss point coordinates part
  for (int z=0;z<numsnode;++z)
  {
    for (int k=0;k<3;++k)
    {
      derivmxi[0][smrtrnodes[z]->Dofs()[k]] += valsxigp[z] * lmatrix(0,k);
      derivmxi[1][smrtrnodes[z]->Dofs()[k]] += valsxigp[z] * lmatrix(1,k);

      for (_CI p=derivsxi[0].begin();p!=derivsxi[0].end();++p)
      {
        derivmxi[0][p->first] += derivsxigp(z,0) * smrtrnodes[z]->xspatial()[k] * lmatrix(0,k) * (p->second);
        derivmxi[1][p->first] += derivsxigp(z,0) * smrtrnodes[z]->xspatial()[k] * lmatrix(1,k) * (p->second);
      }

      for (_CI p=derivsxi[1].begin();p!=derivsxi[1].end();++p)
      {
        derivmxi[0][p->first] += derivsxigp(z,1) * smrtrnodes[z]->xspatial()[k] *lmatrix(0,k) * (p->second);
        derivmxi[1][p->first] += derivsxigp(z,1) * smrtrnodes[z]->xspatial()[k] *lmatrix(1,k) * (p->second);
      }
    }
  }

  // (3) slave Gauss point normal part
  for (_CI p=dmap_nxsl_gp.begin();p!=dmap_nxsl_gp.end();++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0,0) *(p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1,0) *(p->second);
  }
  for (_CI p=dmap_nysl_gp.begin();p!=dmap_nysl_gp.end();++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0,1) *(p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1,1) *(p->second);
  }
  for (_CI p=dmap_nzsl_gp.begin();p!=dmap_nzsl_gp.end();++p)
  {
    derivmxi[0][p->first] += alpha * lmatrix(0,2) *(p->second);
    derivmxi[1][p->first] += alpha * lmatrix(1,2) *(p->second);
  }

  /*
  // check linearization
  typedef std::map<int,double>::const_iterator CI;
  std::cout << "\nLinearization of current master GP:" << std::endl;
  std::cout << "-> Coordinate 1:" << std::endl;
  for (CI p=derivmxi[0].begin();p!=derivmxi[0].end();++p)
    std::cout << p->first << " " << p->second << std::endl;
  std::cout << "-> Coordinate 2:" << std::endl;
  for (CI p=derivmxi[1].begin();p!=derivmxi[1].end();++p)
      std::cout << p->first << " " << p->second << std::endl;
  */

  return;
}