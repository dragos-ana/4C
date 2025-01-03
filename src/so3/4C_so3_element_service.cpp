// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_element_service.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_tet10.hpp"

FOUR_C_NAMESPACE_OPEN

void Discret::Elements::assemble_gauss_point_values(
    std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& global_data,
    const Core::LinAlg::SerialDenseMatrix& gp_data, const Core::Elements::Element& ele)
{
  for (int gp = 0; gp < gp_data.numRows(); ++gp)
  {
    const Epetra_BlockMap& elemap = global_data[gp]->Map();
    int lid = elemap.LID(ele.id());
    if (lid != -1)
    {
      for (int i = 0; i < gp_data.numCols(); ++i)
      {
        ((*global_data[gp])(i))[lid] += gp_data(gp, i);
      }
    }
  }
}

void Discret::Elements::assemble_nodal_element_count(
    Core::LinAlg::Vector<int>& global_count, const Core::Elements::Element& ele)
{
  for (int n = 0; n < ele.num_node(); ++n)
  {
    const int lid = global_count.Map().LID(ele.node_ids()[n]);

    if (lid != -1)
    {
      global_count[lid] += 1;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
std::vector<double> Discret::Elements::project_nodal_quantity_to_xi(
    const Core::LinAlg::Matrix<3, 1>& xi, const std::vector<double>& nodal_quantity)
{
  const int numNodesPerElement = Core::FE::num_nodes<distype>;

  Core::LinAlg::Matrix<numNodesPerElement, 1> shapefunct(true);
  Core::FE::shape_function<distype>(xi, shapefunct);

  const int num_dof_per_node = static_cast<int>(nodal_quantity.size()) / numNodesPerElement;
  std::vector<double> projected_quantities(num_dof_per_node, 0.0);

  for (int dof = 0; dof < num_dof_per_node; ++dof)
  {
    for (int i = 0; i < numNodesPerElement; ++i)
    {
      projected_quantities[dof] += nodal_quantity[i * num_dof_per_node + dof] * shapefunct(i);
    }
  }

  return projected_quantities;
}

// explicit template instantiations
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::hex8>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::hex27>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::tet4>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::tet10>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::wedge6>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);
template std::vector<double>
Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::nurbs27>(
    const Core::LinAlg::Matrix<3, 1>&, const std::vector<double>&);

FOUR_C_NAMESPACE_CLOSE
