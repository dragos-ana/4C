/*----------------------------------------------------------------------------*/
/*! \file

\brief ALE element for 2D case


\level 1
*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#ifndef FOUR_C_ALE_ALE2_HPP
#define FOUR_C_ALE_ALE2_HPP

/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_config.hpp"

#include "4C_discretization_fem_general_utils_integration.hpp"
#include "4C_lib_elementtype.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* forward declarations */
namespace DRT
{
  class Discretization;

  namespace ELEMENTS
  {
    class Ale2Line;

    /*----------------------------------------------------------------------------*/
    /* definition of classes */
    class Ale2Type : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "Ale2Type"; }

      static Ale2Type& Instance();

      CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

      Teuchos::RCP<DRT::Element> Create(const std::string eletype, const std::string eledistype,
          const int id, const int owner) override;

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(
          DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void SetupElementDefinition(
          std::map<std::string, std::map<std::string, INPUT::LineDefinition>>& definitions)
          override;

     private:
      static Ale2Type instance_;
    };

    /*----------------------------------------------------------------------------*/
    /*!
    \brief A C++ wrapper for the ale2 element
    */
    class Ale2 : public DRT::Element
    {
     public:
      //! @name Friends
      friend class Ale2Surface;

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      */
      Ale2(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale2(const Ale2& old);

      /*!
      \brief Deep copy this instance of Ale2 and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int NumLine() const override
      {
        if (NumNode() == 9 || NumNode() == 8 || NumNode() == 4)
          return 4;
        else if (NumNode() == 3 || NumNode() == 6)
          return 3;
        else
        {
          FOUR_C_THROW("Could not determine number of lines");
          return -1;
        }
      }

      /*!
      \brief Return number of surfaces of this element
      */
      int NumSurface() const override { return 1; }

      /*!
      \brief Return number of volumes of this element (always 1)
      */
      int NumVolume() const override { return -1; }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override { return Ale2Type::Instance().UniqueParObjectId(); }

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Unpack(const std::vector<char>& data) override;


      //@}

      //! @name Acess methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override { return 2; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return Ale2Type::Instance(); }

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& distype,
          INPUT::LineDefinition* linedef) override;

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate ale2 element stiffness, mass, internal forces etc

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          std::vector<int>& lm, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2,
          CORE::LINALG::SerialDenseVector& elevec3) override;


      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a surfaces Neumann condition on the shell element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;


      //@}

      //! @name Other

      /// set number of gauss points to element shape default
      CORE::FE::GaussRule2D getOptimalGaussrule(const CORE::FE::CellType& distype);

      //@}


     private:
      /*! \brief ALE mesh motion via Laplacian smoohting
       *
       *  Solve Laplace equation
       *  \f[
       *    \nabla_{config}\cdot\nabla_{config} d = 0, \quad d = \hat d \text{ on } \Gamma
       *  \f]
       *  with \f$\nabla_{config}\f$ denoting the material or spatial gradient
       *  operator and the displacement field \f$d\f$ and satisfying prescribed
       *  displacement on the entire boundary \f$\Gamma\f$.
       *
       *  \note For spatial configuration, displacement vector equals current
       *  displacements. For material configuration, displacement vector is zero.
       */
      void static_ke_laplace(DRT::Discretization& dis,  ///< discretization
          std::vector<int>& lm,                         ///< node owning procs
          CORE::LINALG::SerialDenseMatrix* sys_mat,     ///< element stiffness matrix (to be filled)
          CORE::LINALG::SerialDenseVector& residual,    ///< element residual vector (to be filled)
          std::vector<double>& displacements,           ///< nodal discplacements
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
      );

      void static_ke_spring(
          CORE::LINALG::SerialDenseMatrix* sys_mat,   ///< element stiffness matrix (to be filled)
          CORE::LINALG::SerialDenseVector& residual,  ///< element residual vector (to be filled)
          std::vector<double>& displacements,         ///, nodal displacements
          const bool spatialconfiguration  ///< use spatial configuration (true), material
                                           ///< configuration (false)
      );

      /*! \brief evaluate element quantities for nonlinear case
       *
       *  We can handle St.-Venant-Kirchhoff material and many materials
       *  from the elast hyper toolbox.
       *
       *  \note this routine was copied from DRT::ELEMENTS::Wall1
       */
      void static_ke_nonlinear(const std::vector<int>& lm,  ///< node owning procs
          const std::vector<double>& disp,                  ///< nodal displacements
          CORE::LINALG::SerialDenseMatrix*
              stiffmatrix,                         ///< element stiffness matrix (to be filled)
          CORE::LINALG::SerialDenseVector* force,  ///< element residual vector (to be filled)
          Teuchos::ParameterList& params,          ///< parameter list
          const bool spatialconfiguration,         ///< use spatial configuration (true), material
                                                   ///< configuration (false)
          const bool pseudolinear  ///< compute residual as stiffness * displacements (pseudo-linear
                                   ///< approach)
      );

      /*! \brief Calculate determinant of Jacobian mapping and check for validity
       *
       *  Use current displacement state, i.e. current/spatial configuration.
       *  Check for invalid mappings (detJ <= 0)
       *
       *  \author mayr.mt \date 01/2016
       */
      void compute_det_jac(
          CORE::LINALG::SerialDenseVector& elevec1,  ///< vector with element result data
          const std::vector<int>& lm,                ///< node owning procs
          const std::vector<double>& disp            ///< nodal displacements
      );

      /*! \brief Element quality measure according to [Oddy et al. 1988a]
       *
       *  Distortion metric for quadrilaterals and hexahedrals. Value is zero for
       *  squares/cubes and increases to large values for distorted elements.
       *
       *  Reference: Oddy A, Goldak J, McDill M, Bibby M (1988): A distortion metric
       *  for isoparametric finite elements, Trans. Can. Soc. Mech. Engrg.,
       *  Vol. 12 (4), pp. 213-217
       */
      void EvaluateOddy(const CORE::LINALG::SerialDenseMatrix& xjm, double det, double& qm);

      void CallMatGeoNonl(
          const CORE::LINALG::SerialDenseVector& strain,     ///< Green-Lagrange strain vector
          CORE::LINALG::SerialDenseMatrix& stress,           ///< stress vector
          CORE::LINALG::SerialDenseMatrix& C,                ///< elasticity matrix
          const int numeps,                                  ///< number of strains
          Teuchos::RCP<const CORE::MAT::Material> material,  ///< the material data
          Teuchos::ParameterList& params,                    ///< element parameter list
          int gp                                             ///< Integration point
      );

      void MaterialResponse3dPlane(
          CORE::LINALG::SerialDenseMatrix& stress,        ///< stress state (output)
          CORE::LINALG::SerialDenseMatrix& C,             ///< material tensor (output)
          const CORE::LINALG::SerialDenseVector& strain,  ///< strain state (input)
          Teuchos::ParameterList& params,                 ///< parameter list
          int gp                                          ///< Integration point
      );

      void MaterialResponse3d(CORE::LINALG::Matrix<6, 1>* stress,  ///< stress state (output)
          CORE::LINALG::Matrix<6, 6>* cmat,                        ///< material tensor (output)
          const CORE::LINALG::Matrix<6, 1>* glstrain,              ///< strain state (input)
          Teuchos::ParameterList& params,                          ///< parameter list
          int gp                                                   ///< Integration point
      );

      //! Transform Green-Lagrange notation from 2D to 3D
      void GreenLagrangePlane3d(const CORE::LINALG::SerialDenseVector&
                                    glplane,  ///< Green-Lagrange strains in 2D notation
          CORE::LINALG::Matrix<6, 1>& gl3d);  ///< Green-Lagrange strains in 2D notation

      void edge_geometry(int i, int j, const CORE::LINALG::SerialDenseMatrix& xyze, double* length,
          double* sin_alpha, double* cos_alpha);
      double ale2_area_tria(const CORE::LINALG::SerialDenseMatrix& xyze, int i, int j, int k);
      void ale2_tors_spring_tri3(
          CORE::LINALG::SerialDenseMatrix* sys_mat, const CORE::LINALG::SerialDenseMatrix& xyze);
      void ale2_tors_spring_quad4(
          CORE::LINALG::SerialDenseMatrix* sys_mat, const CORE::LINALG::SerialDenseMatrix& xyze);
      void ale2_torsional(int i, int j, int k, const CORE::LINALG::SerialDenseMatrix& xyze,
          CORE::LINALG::SerialDenseMatrix* k_torsion);

      void CalcBOpLin(CORE::LINALG::SerialDenseMatrix& boplin,
          CORE::LINALG::SerialDenseMatrix& deriv, CORE::LINALG::SerialDenseMatrix& xjm, double& det,
          const int iel);

      void BOpLinCure(CORE::LINALG::SerialDenseMatrix& b_cure,
          const CORE::LINALG::SerialDenseMatrix& boplin, const CORE::LINALG::SerialDenseVector& F,
          const int numeps, const int nd);

      void JacobianMatrix(const CORE::LINALG::SerialDenseMatrix& xrefe,
          const CORE::LINALG::SerialDenseMatrix& deriv, CORE::LINALG::SerialDenseMatrix& xjm,
          double* det, const int iel);

      ///! Compute deformation gradient
      void DefGrad(CORE::LINALG::SerialDenseVector& F,  ///< deformation gradient (to be filled)
          CORE::LINALG::SerialDenseVector& strain,      ///< strain tensor (to be filled)
          const CORE::LINALG::SerialDenseMatrix&
              xrefe,  ///< nodal positions of reference configuration
          const CORE::LINALG::SerialDenseMatrix&
              xcure,                                ///< nodal positions of current configuration
          CORE::LINALG::SerialDenseMatrix& boplin,  ///< B-operator
          const int iel);

      //! Compute geometric part of stiffness matrix
      void Kg(CORE::LINALG::SerialDenseMatrix& estif,  ///< element stiffness matrix (to be filled)
          const CORE::LINALG::SerialDenseMatrix& boplin,  ///< B-operator
          const CORE::LINALG::SerialDenseMatrix& stress,  ///< 2. Piola-Kirchhoff stress tensor
          const double fac,                               ///< factor for Gaussian quadrature
          const int nd,                                   ///< number of DOFs in this element
          const int numeps);

      //! Compute elastic part of stiffness matrix
      void Keu(CORE::LINALG::SerialDenseMatrix& estif,  ///< element stiffness matrix (to be filled)
          const CORE::LINALG::SerialDenseMatrix&
              b_cure,  ///< B-operator for current configuration (input)
          const CORE::LINALG::SerialDenseMatrix& C,  ///< material tensor (input)
          const double fac,                          ///< factor for Gaussian quadrature
          const int nd,                              ///< number of DOFs in this element
          const int numeps);

      //! Compute internal forces for solid approach
      void Fint(
          const CORE::LINALG::SerialDenseMatrix& stress,  ///< 2. Piola-Kirchhoff stress (input)
          const CORE::LINALG::SerialDenseMatrix&
              b_cure,  ///< B-operator for current configuration (input)
          CORE::LINALG::SerialDenseVector& intforce,  ///< force vector (to be filled)
          const double fac,                           ///< factor for Gaussian quadrature
          const int nd                                ///< number of DOFs in this element
      );

      //! action parameters recognized by ale2
      enum ActionType
      {
        none,
        calc_ale_solid,         ///< compute stiffness based on fully nonlinear elastic solid with
                                ///< hyperelastic material law
        calc_ale_solid_linear,  ///< compute stiffness based on linear elastic solid with
                                ///< hyperelastic material law
        calc_ale_springs_material,  ///< compute stiffness based on springs algorithm in material
                                    ///< configuration
        calc_ale_springs_spatial,   ///< compute stiffness based on springs algorithm in spatial
                                    ///< configuration
        calc_ale_laplace_material,  ///< compute stiffness based on laplacian smoothing based on
                                    ///< material configuration
        calc_ale_laplace_spatial,   ///< compute stiffness based on laplacian smoothing based on
                                    ///< spatial configuration
        setup_material,             ///< Setup material in case of ElastHyper Tool Box
        calc_det_jac  ///< calculate Jacobian determinant and mesh quality measure according to
                      ///< [Oddy et al. 1988]
      };

      //! don't want = operator
      Ale2& operator=(const Ale2& old);
    };


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================


    class Ale2LineType : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "Ale2LineType"; }

      static Ale2LineType& Instance();

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(
          DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        CORE::LINALG::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static Ale2LineType instance_;
    };


    /*!
    \brief An element representing a line of a ale2 element
    */
    class Ale2Line : public DRT::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this line
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent ale element of this line
      \param lline: the local line number of this line w.r.t. the parent element
      */
      Ale2Line(int id, int owner, int nnode, const int* nodeids, DRT::Node** nodes,
          DRT::ELEMENTS::Ale2* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Ale2Line(const Ale2Line& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int UniqueParObjectId() const override
      {
        return Ale2LineType::Instance().UniqueParObjectId();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Unpack(const std::vector<char>& data) override;


      //@}

      //! @name Access methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override { return 2; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return Ale2LineType::Instance(); }

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a line Neumann condition on the ale2 element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      // don't want = operator
      Ale2Line& operator=(const Ale2Line& old);

    };  // class Ale2Line



  }  // namespace ELEMENTS
}  // namespace DRT

FOUR_C_NAMESPACE_CLOSE

#endif