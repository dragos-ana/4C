/*-----------------------------------------------------------------------*/
/*! \file
\brief A class for mortar coupling of ONE slave element and ONE master
       element of a mortar interface in 3D (definition of sub-classes).

\level 1

*/
/*---------------------------------------------------------------------*/
#ifndef FOUR_C_MORTAR_COUPLING3D_CLASSES_HPP
#define FOUR_C_MORTAR_COUPLING3D_CLASSES_HPP

#include "4C_config.hpp"

#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MORTAR
{
  // forward declarations
  // namespace UTILS
  //{
  // template<typename _Key, typename _Tp> class fixedsizemap;
  //}
  /*!
  \brief A class representing a special mortar element for 3D quadratic
         mortar coupling with the use auf auxiliary planes. This approach
         is based on "Puso, M.A., Laursen, T.A., Solberg, J., A segment-to-
         segment mortar contact method for quadratic elements and large
         deformations, CMAME, 197, 2008, pp. 555-566". For this type of
         quadratic formulation, a quadratic MORTAR::Element is split into several
         linear IntElements, on which the geometrical coupling is performed.
         Note that this is a derived class from MORTAR::Element, with the
         only difference that we explicitly set the node pointers here!

  */

  class IntElement : public MORTAR::Element
  {
   public:
    /*!
    \brief Standard constructor

    \param lid   (in):    A locally unique element id
    \param id    (in):    Parent element global id
    \param owner (in):    owner processor of the element
    \param parshape (in): shape of parent element
    \param shape (in):    shape of this element
    \param numnode (in):  Number of nodes to this element
    \param nodeids (in):  ids of nodes adjacent to this element
    \param nodes (in):    pointers to nodes adjacent to this element
    \param isslave (in):  flag indicating whether element is slave or master side
    */
    IntElement(int lid, int id, int owner, MORTAR::Element* parele, const CORE::FE::CellType& shape,
        const int numnode, const int* nodeids, std::vector<DRT::Node*> nodes, const bool isslave,
        const bool rewind);


    /*!
    \brief Return local id of this IntElement

    */
    int Lid() const { return lid_; }

    /*!
    \brief Get shape type of parent element

    */
    virtual CORE::FE::CellType ParShape() const { return parele_->Shape(); }

    /*!
    \brief Get shape type of parent element

    */
    virtual MORTAR::Element* ParEle() const { return parele_; }

    /*!
    \brief Affine map of IntElement coordinates to parent element

    */
    virtual bool MapToParent(const double* xi, double* parxi);

    /*!
    \brief Affine map of IntElement coordinate derivatives to parent element

    */
    virtual bool MapToParent(const std::vector<CORE::GEN::Pairedvector<int, double>>& dxi,
        std::vector<CORE::GEN::Pairedvector<int, double>>& dparxi);

    DRT::Node** Nodes() override
    {
      if (parele_->Shape() != CORE::FE::CellType::nurbs9)
        return DRT::Element::Nodes();
      else
        return nodes_ptr_.data();
    }

    /*!
    \brief Get the linearization of the spatial position of the Nodes for this IntEle.
           For Lagrange finite elements, this is trivial, since the nodes are interpolatory.
           For NURBS elements, we generated pseudo-nodes as the corners of the parameter
           space onto the actual surface. Those nodes now depend on all CPs and shape functions
           of the parent NURBS element.

           Returns a vector of vector of maps. Outer vector for the (pseudo-)nodes,
           inner vector for the spatial dimensions, map for the derivatives.
    */
    void NodeLinearization(
        std::vector<std::vector<CORE::GEN::Pairedvector<int, double>>>& nodelin) override;

   protected:
    // don't want = operator and cctor
    IntElement operator=(const IntElement& old);
    IntElement(const IntElement& old);

    int lid_;  // local IntElement id

    // the integration element has its own vector of nodes
    // for Lagrange elements, those are copies of the original mortar nodes
    // for NURBS elements, those are newly created pseudo-nodes, which are
    // located on the physical surface. The "nodes" in IGA are already used
    // with control points
    // Be careful here: the pseudo-nodes need an Id!
    // They are given the Id of the closest control point. However, be aware
    // that this Id is now used twice: once for the control point (as a part
    // of the discretization) and once for the pseudo-node, which is only
    // stored here temporarily.
    std::vector<MORTAR::Node> nodes_;
    std::vector<DRT::Node*> nodes_ptr_;
    const bool rewind_;  // if the parameter space of the int element has been rewinded

    MORTAR::Element* parele_;

  };  // class IntElement


  /*!
  \brief A class representing one Integration Cell after triangulation
         of the clip polygon of slave and master element from the Coupling
         class. This class provides some basic funcitonality a MORTAR::Element would
         also provide (coords, shape functions and derivatives, Jacobian, ...).
         Note that an IntCell can EITHER live in physical space (this is the
         case when an auxiliary plane is used for 3D coupling) or in the slave
         element parameter space (this is the case when 3D coupling is performed
         on the slave surface without any auxiliary plane). Of course, in the
         2nd case the third coordinate of all intcell points is zero!

  */

  class IntCell
  {
   public:
    /*!
    \brief Standard constructor

    Constructs an instance of this class.<br>
    Note that this is \b not a collective call as coupling is
    performed in parallel by individual processes.

    */
    IntCell(int id, int nvertices, CORE::LINALG::Matrix<3, 3>& coords, double* auxn,
        const CORE::FE::CellType& shape, std::vector<CORE::GEN::Pairedvector<int, double>>& linv1,
        std::vector<CORE::GEN::Pairedvector<int, double>>& linv2,
        std::vector<CORE::GEN::Pairedvector<int, double>>& linv3,
        std::vector<CORE::GEN::Pairedvector<int, double>>& linauxn);

    /*!
    \brief Destructor

    */
    virtual ~IntCell() = default;
    //! @name Access methods

    /*!
    \brief Return ID of this intcell

    */
    const int& Id() const { return id_; }

    /*!
    \brief Set slave element ID of this intcell

    */
    void SetSlaveId(const int slaveid)
    {
      slaveId_ = slaveid;
      return;
    }

    /*!
    \brief Return slave element ID of this intcell

    */
    const int& GetSlaveId() const
    {
      if (slaveId_ < 0) FOUR_C_THROW("Invalid slave element ID for this integration cell!");

      return slaveId_;
    }

    /*!
    \brief Set master element ID of this intcell

    */
    void SetMasterId(const int masterid)
    {
      masterId_ = masterid;
      return;
    }
    /*!
    \brief Return master element ID of this intcell

    */
    const int& GetMasterId() const
    {
      if (masterId_ < 0) FOUR_C_THROW("Invalid master element ID for this integration cell!");

      return masterId_;
    }

    /*!
    \brief Return number of vertices of this intcell

    */
    int NumVertices() const { return nvertices_; }

    /*!
    \brief Return current area

    */
    virtual double& Area() { return area_; }

    /*!
    \brief Return coordinates of intcell vertices

    */
    const CORE::LINALG::Matrix<3, 3>& Coords() { return coords_; }

    /*!
    \brief Return normal of auxiliary plane of this intcell

    */
    virtual double* Auxn() { return auxn_; }

    /*!
    \brief Get shape type of element

    */
    virtual CORE::FE::CellType Shape() const { return shape_; }

    /*!
    \brief Return one of the three 'DerivVertex' maps (vectors) of this node

    These maps contain the directional derivatives of the intcell's vertex
    coordinates with respect to the slave and master displacements. A vector
    is used because the coordinates themselves are a vector (2 components for
    the 'coupling in slave parameter space' case). The respective vertex (1,2,3)
    is addressed by an int-variable and checked internally.

    */
    virtual std::vector<CORE::GEN::Pairedvector<int, double>>& GetDerivVertex(int i)
    {
      if (shape_ == CORE::FE::CellType::line2)
      {
        if (i < 0 || i > 1) FOUR_C_THROW("IntLine has 2 vertex linearizations only!");
        return linvertex_[i];
      }
      else
      {
        if (i < 0 || i > 2) FOUR_C_THROW("IntCell has 3 vertex linearizations only!");
        return linvertex_[i];
      }
    }

    /*!
    \brief Return the 'DerivAuxn' map (vector) of this intcell

    */
    virtual std::vector<CORE::GEN::Pairedvector<int, double>>& GetDerivAuxn() { return linauxn_; }

    //@}

    //! @name Evaluation methods

    /*!
    \brief Interpolate global coordinates for given local intcell coordinates

    This method interpolates global coordinates for a given local intcell
    coordinate variable using the intcell vertex coordinates. For interpolation
    one can choose between shape functions or shape function derivatives!

    \param xi (in)        : local intcell coordinates
    \param inttype (in)   : set to 0 for shape function usage,
                            set to 1 for derivative xi usage
                            set to 2 for derivative eta usage
    \param globccord (out): interpolated global coordinates
    */
    virtual bool LocalToGlobal(const double* xi, double* globcoord, int inttype);

    // output functionality
    virtual void Print();

    /*!
    \brief Evaluate shape functions and derivatives

    */
    virtual bool EvaluateShape(
        const double* xi, CORE::LINALG::Matrix<3, 1>& val, CORE::LINALG::Matrix<3, 2>& deriv);

    /*!
    \brief Evaluate Jacobian determinant for parameter space integration

    */
    virtual double Jacobian();

    /*!
    \brief Compute Jacobian determinant derivative
           Note that this is a linearization with respect to the intcell
           vertices, which themselves have to be linearized later, of course!
    */
    virtual void DerivJacobian(CORE::GEN::Pairedvector<int, double>& derivjac);

    //@}

   protected:
    // don't want = operator and cctor
    IntCell operator=(const IntCell& old);
    IntCell(const IntCell& old);

    int id_;                             // local ID of this intcell
    int slaveId_;                        // id of slave element
    int masterId_;                       // id of master element
    int nvertices_;                      // number of vertices (always 3)
    double area_;                        // integration cell area
    CORE::LINALG::Matrix<3, 3> coords_;  // coords of cell vertices
    double auxn_[3];                     // normal of auxiliary plane (3D)

    bool auxplane_;             // flag indicating coupling strategy (true = auxplane)
    CORE::FE::CellType shape_;  // shape of this element (always tri3)
    std::vector<std::vector<CORE::GEN::Pairedvector<int, double>>>
        linvertex_;  // derivatives of the 3 vertices
    std::vector<CORE::GEN::Pairedvector<int, double>>
        linauxn_;  // derivatives of auxiliary plane normal

  };  // class IntCell

  /*!
  \brief A class representing one Vertex during the polygon clipping of
         slave and master element from the Coupling class. Besides the
         vertex coordinates this class provides different pointers to
         build up doubly-linked list structures.
         Note that a Vertex can EITHER live in physical space (this is the
         case when an auxiliary plane is used for 3D coupling) or in the slave
         element parameter space (this is the case when 3D coupling is performed
         on the slave surface without any auxiliary plane). Of course, in the
         2nd case the third coordinate of all vertices is zero!

  */

  class Vertex
  {
   public:
    //! @name Enums and Friends
    enum VType  // vertex types recognized by Vertex
    {
      slave,       // slave node
      projmaster,  // projected master node
      lineclip,    // clipping point of two lines
      master,      // master node (LTS)
      projslave    // projected slave node (LTS)
    };

    //@}

    /*!
    \brief Standard constructor

    Constructs an instance of this class.<br>
    Note that this is \b not a collective call as coupling is
    performed in parallel by individual processes.

    */
    Vertex(std::vector<double> coord, Vertex::VType type, std::vector<int> nodeids, Vertex* next,
        Vertex* prev, bool intersect, bool entryexit, Vertex* neighbor, double alpha);

    /*!
    \brief Copy Constructor

    Makes a deep copy of a Vertex

    */
    Vertex(const Vertex& old);

    /*!
    \brief Destructor

    */
    virtual ~Vertex() = default;
    //! @name Access methods

    /*!
    \brief Return vector of vertex coordinates (length 3)

    */
    virtual std::vector<double>& Coord() { return coord_; }

    /*!
    \brief Return vertex type (slave, projmaster or lineclip)

    */
    virtual Vertex::VType& v_type() { return type_; }

    /*!
    \brief Return pointer to next vertex on polygon

    */
    virtual Vertex* Next() { return next_; }

    /*!
    \brief Assign pointer to next vertex on polygon

    */
    virtual void AssignNext(Vertex* assign)
    {
      next_ = assign;
      return;
    }

    /*!
    \brief Return pointer to previous vertex on polygon

    */
    virtual Vertex* Prev() { return prev_; }

    /*!
    \brief Assign pointer to previous vertex on polygon

    */
    virtual void AssignPrev(Vertex* assign)
    {
      prev_ = assign;
      return;
    }

    /*!
    \brief Return intersection status of this vertex
    True if vertex is an intersection point of the polygons.

    */
    virtual bool& Intersect() { return intersect_; }

    /*!
    \brief Return entry / exit status of this vertex
    True if vertex is an entry intersection point, false if
    vertex is an exit intesection point with respect to the
    respective other polygon. Irrelevant if intersect_==false.

    */
    virtual bool& EntryExit()
    {
      if (!intersect_) FOUR_C_THROW("EntryExit only for intersections");
      return entryexit_;
    }

    /*!
    \brief Return pointer to neighbor on other polygon
    This pointer can only be set for an intersrection vertex,
    i.e. if intersect_==true. It then points to the identical
    vertex on the other polygon.

    */
    virtual Vertex* Neighbor()
    {
      if (!intersect_) FOUR_C_THROW("Neighbor only for intersections");
      return neighbor_;
    }

    /*!
    \brief Assign pointer to neighbor on other polygon

    */
    virtual void AssignNeighbor(Vertex* assign)
    {
      if (!intersect_) FOUR_C_THROW("Neighbor only for intersections");
      neighbor_ = assign;
      return;
    }

    /*!
    \brief Return intersection parameter alpha. Note that
    valid intersections yield an alpha in the range [0,1].

    */
    virtual double& Alpha() { return alpha_; }

    /*!
    \brief Return vector of relevant node ids (length 1 or 4)
    Note that for a slave or projmaster type vertex only the
    respective node id is relevant, therefore length 1. For a
    lineclip type vertex we need the 4 node ids of both the
    corresponding slave and master lines which intersect!

    */
    virtual std::vector<int>& Nodeids()
    {  // if(type_==Vertex::slave && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::projmaster && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::lineclip && nodeids_.size()!=4) FOUR_C_THROW("Error: Vertex Ids");
      return nodeids_;
    }

    //@}


   protected:
    std::vector<double> coord_;  // vertex coordinates (length 3)
    Vertex::VType type_;         // vertex type (slave,projmaster,lineclip)
    std::vector<int> nodeids_;   // relevant ids (1 if slave or master, 4 if lineclip)
    Vertex* next_;               // pointer to next vertex on polygon
    Vertex* prev_;               // pointer to previous vertex on polygon
    bool intersect_;             // if true, this is an intersection vertex
    bool entryexit_;             // if true, this is an entry vertex
    Vertex* neighbor_;           // pointer to neighbor vertex on other polygon
    double alpha_;               // intersection parameter

  };  // class Vertex

}  // namespace MORTAR

FOUR_C_NAMESPACE_CLOSE

#endif