/*---------------------------------------------------------------------*/
/*! \file

\brief Implementation of evaluate calls on discretization

\level 0


*/
/*---------------------------------------------------------------------*/

#include "4C_comm_parobjectfactory.hpp"
#include "4C_global_data.hpp"
#include "4C_lib_assemblestrategy.hpp"
#include "4C_lib_discret.hpp"
#include "4C_lib_elements_paramsinterface.hpp"
#include "4C_lib_utils_discret.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate (public)                                        mwgee 12/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::Evaluate(Teuchos::ParameterList& params,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix1,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3)
{
  DRT::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);
  Evaluate(params, strategy);
}


void DRT::Discretization::Evaluate(Teuchos::ParameterList& params, DRT::AssembleStrategy& strategy)
{
  // Call the Evaluate method for the specific element
  Evaluate(params, strategy,
      [&](DRT::Element& ele, Element::LocationArray& la, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2, CORE::LINALG::SerialDenseVector& elevec3)
      {
        const int err =
            ele.Evaluate(params, *this, la, strategy.Elematrix1(), strategy.Elematrix2(),
                strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
        if (err) FOUR_C_THROW("Proc %d: Element %d returned err=%d", Comm().MyPID(), ele.Id(), err);
      });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::Discretization::Evaluate(Teuchos::ParameterList& params, DRT::AssembleStrategy& strategy,
    const std::function<void(DRT::Element&, Element::LocationArray&,
        CORE::LINALG::SerialDenseMatrix&, CORE::LINALG::SerialDenseMatrix&,
        CORE::LINALG::SerialDenseVector&, CORE::LINALG::SerialDenseVector&,
        CORE::LINALG::SerialDenseVector&)>& element_action)
{
  TEUCHOS_FUNC_TIME_MONITOR("DRT::Discretization::Evaluate");

  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() was not called");

  int row = strategy.FirstDofSet();
  int col = strategy.SecondDofSet();

  // call the element's register class preevaluation method
  // for each type of element
  // for most element types, just the base class dummy is called
  // that does nothing
  CORE::COMM::ParObjectFactory::Instance().PreEvaluate(*this, params, strategy.Systemmatrix1(),
      strategy.Systemmatrix2(), strategy.Systemvector1(), strategy.Systemvector2(),
      strategy.Systemvector3());

  Element::LocationArray la(dofsets_.size());

  // loop over column elements
  for (Element* actele : MyColElementRange())
  {
    // get element location vector, dirichlet flags and ownerships
    actele->LocationVector(*this, la, false);

    // get dimension of element matrices and vectors
    // Reshape element matrices and vectors and init to zero
    strategy.ClearElementStorage(la[row].Size(), la[col].Size());

    // call the element evaluate method
    element_action(*actele, la, strategy.Elematrix1(), strategy.Elematrix2(), strategy.Elevector1(),
        strategy.Elevector2(), strategy.Elevector3());

    int eid = actele->Id();
    strategy.AssembleMatrix1(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
    strategy.AssembleMatrix2(eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
    strategy.AssembleVector1(la[row].lm_, la[row].lmowner_);
    strategy.AssembleVector2(la[row].lm_, la[row].lmowner_);
    strategy.AssembleVector3(la[row].lm_, la[row].lmowner_);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate (public)                                        u.kue 01/08|
 *----------------------------------------------------------------------*/
void DRT::Discretization::Evaluate(Teuchos::ParameterList& params,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<Epetra_Vector> systemvector)
{
  Evaluate(params, systemmatrix, Teuchos::null, systemvector, Teuchos::null, Teuchos::null);
}

void DRT::Discretization::Evaluate(const std::function<void(DRT::Element&)>& element_action)
{
  // test only for Filled()!Dof information is not required
  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");

  for (Element* actele : MyColElementRange())
  {
    // call the element evaluate method
    element_action(*actele);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate (public)                                        a.ger 03/09|
 *----------------------------------------------------------------------*/
void DRT::Discretization::Evaluate(Teuchos::ParameterList& params)
{
  // define empty element matrices and vectors
  CORE::LINALG::SerialDenseMatrix elematrix1;
  CORE::LINALG::SerialDenseMatrix elematrix2;
  CORE::LINALG::SerialDenseVector elevector1;
  CORE::LINALG::SerialDenseVector elevector2;
  CORE::LINALG::SerialDenseVector elevector3;

  Element::LocationArray la(dofsets_.size());

  Evaluate(
      [&](DRT::Element& ele)
      {
        const int err = ele.Evaluate(
            params, *this, la, elematrix1, elematrix2, elevector1, elevector2, elevector3);
        if (err) FOUR_C_THROW("Proc %d: Element %d returned err=%d", Comm().MyPID(), ele.Id(), err);
      });
}


/*----------------------------------------------------------------------*
 |  evaluate Neumann conditions (public)                     mwgee 12/06|
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateNeumann(Teuchos::ParameterList& params,
    Epetra_Vector& systemvector, CORE::LINALG::SparseOperator* systemmatrix)
{
  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() was not called");

  bool assemblemat = (systemmatrix != nullptr);

  // get the current time
  double time = params.get("total time", -1.0);

  if (params.isParameter("interface"))
  {
    time = params.get<Teuchos::RCP<DRT::ELEMENTS::ParamsInterface>>("interface")->GetTotalTime();
  }

  //--------------------------------------------------------
  // loop through Point Neumann conditions and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name != (std::string) "PointNeumann") continue;
    if (assemblemat && !systemvector.Comm().MyPID())
    {
      std::cout << "WARNING: System matrix handed in but no linearization of "
                   "PointNeumann conditions implemented. Did you set the LOADLIN-flag "
                   "accidentally?\n";
    }
    const std::vector<int>* nodeids = cond->GetNodes();
    if (!nodeids) FOUR_C_THROW("PointNeumann condition does not have nodal cloud");
    const auto* tmp_funct = cond->GetIf<std::vector<int>>("funct");
    const auto& onoff = cond->Get<std::vector<int>>("onoff");
    const auto& val = cond->Get<std::vector<double>>("val");

    for (const int nodeid : *nodeids)
    {
      // do only nodes in my row map
      if (!NodeRowMap()->MyGID(nodeid)) continue;
      DRT::Node* actnode = gNode(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node %d", nodeid);
      // call explicitly the main dofset, nodeid.e. the first column
      std::vector<int> dofs = Dof(0, actnode);
      const unsigned numdf = dofs.size();
      for (unsigned j = 0; j < numdf; ++j)
      {
        if (onoff[j] == 0) continue;
        const int gid = dofs[j];
        double value = val[j];

        const double functfac = std::invoke(
            [&]()
            {
              if (tmp_funct && (*tmp_funct)[j] > 0)
              {
                return GLOBAL::Problem::Instance()
                    ->FunctionById<CORE::UTILS::FunctionOfTime>((*tmp_funct)[j] - 1)
                    .Evaluate(time);
              }
              else
                return 1.0;
            });

        value *= functfac;
        const int lid = systemvector.Map().LID(gid);
        if (lid < 0) FOUR_C_THROW("Global id %d not on this proc in system vector", gid);
        systemvector[lid] += value;
      }
    }
  }

  //--------------------------------------------------------
  // loop through line/surface/volume Neumann BCs and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name == (std::string) "LineNeumann" || name == (std::string) "SurfaceNeumann" ||
        name == (std::string) "VolumeNeumann")
    {
      std::map<int, Teuchos::RCP<DRT::Element>>& geom = cond->Geometry();
      CORE::LINALG::SerialDenseVector elevector;
      CORE::LINALG::SerialDenseMatrix elematrix;
      for (const auto& [_, ele] : geom)
      {
        // get element location vector, dirichlet flags and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;
        ele->LocationVector(*this, lm, lmowner, lmstride);
        elevector.size((int)lm.size());
        if (!assemblemat)
        {
          ele->EvaluateNeumann(params, *this, *cond, lm, elevector);
          CORE::LINALG::Assemble(systemvector, elevector, lm, lmowner);
        }
        else
        {
          const int size = lm.size();
          if (elematrix.numRows() != size)
            elematrix.shape(size, size);
          else
            elematrix.putScalar(0.0);
          ele->EvaluateNeumann(params, *this, *cond, lm, elevector, &elematrix);
          CORE::LINALG::Assemble(systemvector, elevector, lm, lmowner);
          systemmatrix->Assemble(ele->Id(), lmstride, elematrix, lm, lmowner);
        }
      }
    }
  }

  //--------------------------------------------------------
  // loop through Point Moment EB conditions and evaluate them
  //--------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name != (std::string) "PointNeumannEB") continue;
    const std::vector<int>* nodeids = cond->GetNodes();
    if (!nodeids) FOUR_C_THROW("Point Moment condition does not have nodal cloud");

    for (const int nodeid : *nodeids)
    {
      // create matrices for fext and fextlin
      CORE::LINALG::SerialDenseVector elevector;
      CORE::LINALG::SerialDenseMatrix elematrix;

      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;

      // do only nodes in my row map
      if (!NodeRowMap()->MyGID(nodeid)) continue;

      // get global node
      DRT::Node* actnode = gNode(nodeid);
      if (!actnode) FOUR_C_THROW("Cannot find global node %d", nodeid);

      // get elements attached to global node
      DRT::Element** curreleptr = actnode->Elements();

      // find element from pointer
      // please note, that external force will be applied to the first element [0] attached to a
      // node this needs to be done, otherwise it will be applied several times on several elements.
      DRT::Element* currele = curreleptr[0];

      // get information from location
      currele->LocationVector(*this, lm, lmowner, lmstride);
      const int size = (int)lm.size();
      elevector.size(size);

      // evaluate linearized point moment conditions and assemble f_ext and f_ext_lin into global
      // matrix
      //-----if the stiffness matrix was given in-------
      if (assemblemat)
      {
        // resize f_ext_lin matrix
        if (elematrix.numRows() != size)
          elematrix.shape(size, size);
        else
          elematrix.putScalar(0.0);
        // evaluate linearized point moment conditions and assemble matrices
        currele->EvaluateNeumann(params, *this, *cond, lm, elevector, &elematrix);
        systemmatrix->Assemble(currele->Id(), lmstride, elematrix, lm, lmowner);
      }
      //-----if no stiffness matrix was given in-------
      else
        currele->EvaluateNeumann(params, *this, *cond, lm, elevector);
      CORE::LINALG::Assemble(systemvector, elevector, lm, lmowner);
    }
  }
}


/*----------------------------------------------------------------------*
 |  evaluate Dirichlet conditions (public)                  rauch 06/16 |
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateDirichlet(Teuchos::ParameterList& params,
    Teuchos::RCP<Epetra_Vector> systemvector, Teuchos::RCP<Epetra_Vector> systemvectord,
    Teuchos::RCP<Epetra_Vector> systemvectordd, Teuchos::RCP<Epetra_IntVector> toggle,
    Teuchos::RCP<CORE::LINALG::MapExtractor> dbcmapextractor) const
{
  DRT::UTILS::EvaluateDirichlet(
      *this, params, systemvector, systemvectord, systemvectordd, toggle, dbcmapextractor);
}


/*----------------------------------------------------------------------*
 |  evaluate a condition (public)                               tk 02/08|
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateCondition(Teuchos::ParameterList& params,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix1,
    Teuchos::RCP<CORE::LINALG::SparseOperator> systemmatrix2,
    Teuchos::RCP<Epetra_Vector> systemvector1, Teuchos::RCP<Epetra_Vector> systemvector2,
    Teuchos::RCP<Epetra_Vector> systemvector3, const std::string& condstring, const int condid)
{
  DRT::AssembleStrategy strategy(
      0, 0, systemmatrix1, systemmatrix2, systemvector1, systemvector2, systemvector3);
  EvaluateCondition(params, strategy, condstring, condid);
}  // end of DRT::Discretization::EvaluateCondition


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateCondition(Teuchos::ParameterList& params,
    DRT::AssembleStrategy& strategy, const std::string& condstring, const int condid)
{
  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() was not called");

  int row = strategy.FirstDofSet();
  int col = strategy.SecondDofSet();

  // get the current time
  const double time = params.get("total time", -1.0);

  Element::LocationArray la(dofsets_.size());

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (const auto& [name, cond] : condition_)
  {
    if (name == condstring)
    {
      if (condid == -1 || condid == cond->Get<int>("ConditionID"))
      {
        std::map<int, Teuchos::RCP<DRT::Element>>& geom = cond->Geometry();
        // if (geom.empty()) FOUR_C_THROW("evaluation of condition with empty geometry");
        // no check for empty geometry here since in parallel computations
        // can exist processors which do not own a portion of the elements belonging
        // to the condition geometry

        // Evaluate Loadcurve if defined. Put current load factor in parameter list
        const auto* curve = cond->GetIf<int>("curve");
        int curvenum = -1;
        if (curve) curvenum = *curve;
        double curvefac = 1.0;
        if (curvenum >= 0)
        {
          curvefac = GLOBAL::Problem::Instance()
                         ->FunctionById<CORE::UTILS::FunctionOfTime>(curvenum)
                         .Evaluate(time);
        }

        // Get ConditionID of current condition if defined and write value in parameter list
        const auto* condID = cond->GetIf<int>("ConditionID");
        if (condID)
        {
          params.set("ConditionID", *condID);
          params.set("LoadCurveFactor " + std::to_string(*condID), curvefac);
        }
        else
        {
          params.set("LoadCurveFactor", curvefac);
        }
        params.set<Teuchos::RCP<DRT::Condition>>("condition", cond);

        for (const auto& [_, ele] : geom)
        {
          // get element location vector and ownerships
          // the LocationVector method will return the location vector
          // of the dofs this condition is meant to assemble into.
          // These dofs do not need to be the same as the dofs of the element
          // (this is the standard case, though). Special boundary conditions,
          // like weak Dirichlet conditions, assemble into the dofs of the parent element.
          ele->LocationVector(*this, la, false, condstring, params);

          // get dimension of element matrices and vectors
          // Reshape element matrices and vectors and initialize to zero
          strategy.ClearElementStorage(la[row].Size(), la[col].Size());

          // call the element specific evaluate method
          int err = ele->Evaluate(params, *this, la, strategy.Elematrix1(), strategy.Elematrix2(),
              strategy.Elevector1(), strategy.Elevector2(), strategy.Elevector3());
          if (err) FOUR_C_THROW("error while evaluating elements");

          // assembly
          /* If BlockMatrixes are used, the decision which assemble strategy is
           * used, is based on the element id. As this id is compared to a list
           * of conditioned volume elements, always the volume element id should
           * be given to the Assembling! (comment: eid is not used by
           * sysmat.assemble(...,eid,...))*/
          int eid;
          if (auto* faceele = dynamic_cast<DRT::FaceElement*>(ele.get()))
            eid = faceele->ParentElement()->Id();
          else
            eid = ele->Id();

          strategy.AssembleMatrix1(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.AssembleMatrix2(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.AssembleVector1(la[row].lm_, la[row].lmowner_);
          strategy.AssembleVector2(la[row].lm_, la[row].lmowner_);
          strategy.AssembleVector3(la[row].lm_, la[row].lmowner_);
        }
      }
    }
  }
}  // end of DRT::Discretization::EvaluateCondition


/*----------------------------------------------------------------------*
 |  evaluate/assemble scalars across elements (public)       bborn 08/08|
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateScalars(
    Teuchos::ParameterList& params, Teuchos::RCP<CORE::LINALG::SerialDenseVector> scalars)
{
  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() was not called");

  // number of scalars
  const int numscalars = scalars->length();
  if (numscalars <= 0) FOUR_C_THROW("scalars vector of interest has size <=0");
  // intermediate sum of each scalar on each processor
  CORE::LINALG::SerialDenseVector cpuscalars(numscalars);

  // define element matrices and vectors
  // -- which are empty and unused, just to satisfy element Evaluate()
  CORE::LINALG::SerialDenseMatrix elematrix1;
  CORE::LINALG::SerialDenseMatrix elematrix2;
  CORE::LINALG::SerialDenseVector elevector2;
  CORE::LINALG::SerialDenseVector elevector3;

  // loop over _row_ elements
  for (const auto& actele : MyRowElementRange())
  {
    // get element location vector
    Element::LocationArray la(dofsets_.size());
    actele->LocationVector(*this, la, false);

    // define element vector
    CORE::LINALG::SerialDenseVector elescalars(numscalars);

    // call the element evaluate method
    {
      int err = actele->Evaluate(
          params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc %d: Element %d returned err=%d", Comm().MyPID(), actele->Id(), err);
    }

    // sum up (on each processor)
    cpuscalars += elescalars;
  }

  // reduce
  for (int i = 0; i < numscalars; ++i) (*scalars)(i) = 0.0;
  Comm().SumAll(cpuscalars.values(), scalars->values(), numscalars);
}  // DRT::Discretization::EvaluateScalars


/*-----------------------------------------------------------------------------*
 | evaluate/assemble scalars across conditioned elements (public)   fang 02/15 |
 *-----------------------------------------------------------------------------*/
void DRT::Discretization::EvaluateScalars(Teuchos::ParameterList& params,  //! (in) parameter list
    Teuchos::RCP<CORE::LINALG::SerialDenseVector>
        scalars,                    //! (out) result vector for scalar quantities to be computed
    const std::string& condstring,  //! (in) name of condition to be evaluated
    const int condid                //! (in) condition ID (optional)
)
{
  // safety checks
  if (!Filled()) FOUR_C_THROW("FillComplete() has not been called on discretization!");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() has not been called on discretization!");

  // determine number of scalar quantities to be computed
  const int numscalars = scalars->length();

  // safety check
  if (numscalars <= 0)
    FOUR_C_THROW("Result vector for EvaluateScalars routine must have positive length!");

  // initialize vector for intermediate results of scalar quantities on single processor
  CORE::LINALG::SerialDenseVector cpuscalars(numscalars);

  // define empty dummy element matrices and residuals
  CORE::LINALG::SerialDenseMatrix elematrix1;
  CORE::LINALG::SerialDenseMatrix elematrix2;
  CORE::LINALG::SerialDenseVector elevector2;
  CORE::LINALG::SerialDenseVector elevector3;

  // loop over all conditions on discretization
  for (const auto& [name, condition] : condition_)
  {
    // consider only conditions with specified label
    if (name == condstring)
    {
      // additional filtering by condition ID if explicitly provided
      if (condid == -1 or condid == condition->Get<int>("ConditionID"))
      {
        // extract geometry map of current condition
        std::map<int, Teuchos::RCP<DRT::Element>>& geometry = condition->Geometry();

        // add condition to parameter list for elements
        params.set<Teuchos::RCP<DRT::Condition>>("condition", condition);

        // loop over all elements associated with current condition
        for (auto& [_, element] : geometry)
        {
          // consider only unghosted elements for evaluation
          if (element->Owner() == Comm().MyPID())
          {
            // construct location vector for current element
            Element::LocationArray la(dofsets_.size());
            element->LocationVector(*this, la, false);

            // initialize result vector for current element
            CORE::LINALG::SerialDenseVector elescalars(numscalars);

            // call element evaluation routine
            int error = element->Evaluate(
                params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);

            // safety check
            if (error)
            {
              FOUR_C_THROW(
                  "Element evaluation failed for element %d on processor %d with error code %d!",
                  element->Id(), Comm().MyPID(), error);
            }

            // update result vector on single processor
            cpuscalars += elescalars;
          }  // if(element.Owner() == Comm().MyPID())
        }    // loop over elements
      }      // if(condid == -1 or condid == condition.Get<int>("ConditionID"))
    }        // if(conditionpair->first == condstring)
  }          // loop over conditions

  // communicate results across all processors
  Comm().SumAll(cpuscalars.values(), scalars->values(), numscalars);
}  // DRT::Discretization::EvaluateScalars


/*----------------------------------------------------------------------*
 |  evaluate/assemble scalars across elements (public)         gee 05/11|
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateScalars(
    Teuchos::ParameterList& params, Teuchos::RCP<Epetra_MultiVector> scalars)
{
  if (!Filled()) FOUR_C_THROW("FillComplete() was not called");
  if (!HaveDofs()) FOUR_C_THROW("AssignDegreesOfFreedom() was not called");

  Epetra_MultiVector& sca = *(scalars.get());

  // number of scalars
  const int numscalars = scalars->NumVectors();
  if (numscalars <= 0) FOUR_C_THROW("scalars vector of interest has size <=0");

  // define element matrices and vectors
  // -- which are empty and unused, just to satisfy element Evaluate()
  CORE::LINALG::SerialDenseMatrix elematrix1;
  CORE::LINALG::SerialDenseMatrix elematrix2;
  CORE::LINALG::SerialDenseVector elevector2;
  CORE::LINALG::SerialDenseVector elevector3;

  // loop over _row_ elements
  const int numrowele = NumMyRowElements();
  for (int i = 0; i < numrowele; ++i)
  {
    // pointer to current element
    DRT::Element* actele = lRowElement(i);

    if (!scalars->Map().MyGID(actele->Id()))
      FOUR_C_THROW("Proc does not have global element %d", actele->Id());

    // get element location vector
    Element::LocationArray la(dofsets_.size());
    actele->LocationVector(*this, la, false);

    // define element vector
    CORE::LINALG::SerialDenseVector elescalars(numscalars);

    // call the element evaluate method
    {
      int err = actele->Evaluate(
          params, *this, la, elematrix1, elematrix2, elescalars, elevector2, elevector3);
      if (err)
        FOUR_C_THROW("Proc %d: Element %d returned err=%d", Comm().MyPID(), actele->Id(), err);
    }

    for (int j = 0; j < numscalars; ++j)
    {
      (*sca(j))[i] = elescalars(j);
    }

  }  // for (int i=0; i<numrowele; ++i)
}  // DRT::Discretization::EvaluateScalars


/*----------------------------------------------------------------------*
 |  evaluate an initial scalar or vector field (public)       popp 06/11|
 *----------------------------------------------------------------------*/
void DRT::Discretization::EvaluateInitialField(const std::string& fieldstring,
    Teuchos::RCP<Epetra_Vector> fieldvector, const std::vector<int>& locids) const
{
  DRT::UTILS::EvaluateInitialField(*this, fieldstring, fieldvector, locids);
}  // DRT::Discretization::EvaluateIntialField

FOUR_C_NAMESPACE_CLOSE