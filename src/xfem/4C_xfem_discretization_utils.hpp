// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_DISCRETIZATION_UTILS_HPP
#define FOUR_C_XFEM_DISCRETIZATION_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace XFEM
{
  namespace Utils
  {
    void print_discretization_to_stream(std::shared_ptr<Core::FE::Discretization> dis,
        const std::string& disname, bool elements, bool elecol, bool nodes, bool nodecol,
        bool faces, bool facecol, std::ostream& s,
        std::map<int, Core::LinAlg::Matrix<3, 1>>* curr_pos = nullptr);

    class XFEMDiscretizationBuilder
    {
     public:
      /// constructor
      XFEMDiscretizationBuilder() { /* should stay empty! */ };

      void setup_xfem_discretization(const Teuchos::ParameterList& xgen_params,
          std::shared_ptr<Core::FE::Discretization> dis, int numdof = 4) const;

      //! setup xfem discretization and embedded discretization
      void setup_xfem_discretization(const Teuchos::ParameterList& xgen_params,
          std::shared_ptr<Core::FE::Discretization> dis, Core::FE::Discretization& embedded_dis,
          const std::string& embedded_cond_name, int numdof = 4) const;

     private:
      //! split a discretization into two by removing conditioned nodes
      //! in source and adding to target
      void split_discretization_by_condition(
          Core::FE::Discretization& sourcedis,  //< initially contains all
          Core::FE::Discretization& targetdis,  //< initially empty
          std::vector<const Core::Conditions::Condition*>&
              conditions,  //< conditioned nodes to be shifted to target
          const std::vector<std::string>& conditions_to_copy  //< conditions to copy to target
      ) const;

      /*! split the discretization by removing the given elements and nodes in
       *  the source discretization and adding them to the target discretization */
      void split_discretization(Core::FE::Discretization& sourcedis,
          Core::FE::Discretization& targetdis, const std::map<int, Core::Nodes::Node*>& sourcenodes,
          const std::map<int, Core::Nodes::Node*>& sourcegnodes,
          const std::map<int, std::shared_ptr<Core::Elements::Element>>& sourceelements,
          const std::vector<std::string>& conditions_to_copy) const;

      //! re-partitioning of newly created discretizations (e.g. split by condition)
      void redistribute(Core::FE::Discretization& dis, std::vector<int>& noderowvec,
          std::vector<int>& nodecolvec) const;

      /** \brief remove conditions which are no longer part of the split
       *         partial discretizations, respectively
       *
       *  */
      std::shared_ptr<Core::Conditions::Condition> split_condition(
          const Core::Conditions::Condition* src_cond, const std::vector<int>& nodecolvec,
          MPI_Comm comm) const;
    };  // class XFEMDiscretizationBuilder
  }  // namespace Utils

  class DiscretizationXWall : public Core::FE::DiscretizationFaces
  {
   public:
    /*!
    \brief Standard Constructor

    \param name: name of this discretization
    \param comm: comm object associated with this discretization
    \param n_dim: number of space dimensions of this discretization
    */
    DiscretizationXWall(const std::string name, MPI_Comm comm, unsigned int n_dim);



    /*!
    \brief Get the gid of all dofs of a node.

    Ask the current DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))

    Additional input nodal dof set: If the node contains more than one set of dofs, which can be
    evaluated, the number of the set needs to be given. Currently only the case for XFEM.

    \param dof (in)         : vector of dof gids (to be filled)
    \param nds (in)         : number of dofset
    \param nodaldofset (in) : number of nodal dofset
    \param node (in)        : the node
    \param element (in)     : the element (optionally)
    */
    void dof(std::vector<int>& dof, const Core::Nodes::Node* node, unsigned nds,
        unsigned nodaldofset, const Core::Elements::Element* element = nullptr) const override
    {
      if (nds > 1) FOUR_C_THROW("xwall discretization can only handle one dofset at the moment");

      FOUR_C_ASSERT(nds < dofsets_.size(), "undefined dof set");
      FOUR_C_ASSERT(havedof_, "no dofs assigned");

      std::vector<int> totaldof;
      dofsets_[nds]->dof(totaldof, node, nodaldofset);

      FOUR_C_ASSERT_ALWAYS(element, "element required for location vector of hex8 element");

      const int size = std::min((int)totaldof.size(), element->num_dof_per_node(*node));
      // only take the first dofs that have a meaning for all elements at this node
      for (int i = 0; i < size; i++) dof.push_back(totaldof.at(i));


      return;
    }
  };  // class DiscretizationXWall
}  // namespace XFEM


FOUR_C_NAMESPACE_CLOSE

#endif
