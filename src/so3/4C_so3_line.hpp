// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_LINE_HPP
#define FOUR_C_SO3_LINE_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN



namespace Discret
{
  namespace Elements
  {
    class StructuralLineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "StructuralLineType"; }

      static StructuralLineType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
      }

     private:
      static StructuralLineType instance_;
    };

    /*!
    \brief An element representing a line edge of any 3D structural element
           Note that this element should not be used in 2D cases!

    */
    class StructuralLine : public Core::Elements::FaceElement
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
      \param parent: The parent shell element of this line
      \param lline: the local line number of this line w.r.t. the parent element
      */
      StructuralLine(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Core::Elements::Element* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      StructuralLine(const StructuralLine& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      inline int unique_par_object_id() const override
      {
        return StructuralLineType::instance().unique_par_object_id();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element

      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      //@}

      //! @name Access methods

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      inline int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual Core::Elements::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      inline int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return StructuralLineType::instance();
      }

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a line Neumann condition on the shell element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      // don't want = operator
      StructuralLine& operator=(const StructuralLine& old);

      //! gaussian integration to be used
      Core::FE::GaussRule1D gaussrule_;

      //! line integration
      void line_integration(double& dL, const Core::LinAlg::SerialDenseMatrix& x,
          const Core::LinAlg::SerialDenseMatrix& deriv);

      /*!
      \brief Create matrix with material configuration

      \param x  (out)  : nodal coords in material frame
      */
      inline void material_configuration(Core::LinAlg::SerialDenseMatrix& x) const
      {
        const int numnode = num_node();
        for (int i = 0; i < numnode; ++i)
        {
          x(i, 0) = nodes()[i]->x()[0];
          x(i, 1) = nodes()[i]->x()[1];
          x(i, 2) = nodes()[i]->x()[2];
        }
        return;
      }

    };  // class StructuralLine


  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
