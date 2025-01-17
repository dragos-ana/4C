// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_immersed_problem_fsi_partitioned_immersed.hpp"

#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fsi_debugwriter.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN


FSI::PartitionedImmersed::PartitionedImmersed(MPI_Comm comm) : Partitioned(comm)
{
  // empty constructor
}


void FSI::PartitionedImmersed::setup()
{
  // call setup of base class
  FSI::Partitioned::setup();
}


void FSI::PartitionedImmersed::setup_coupling(const Teuchos::ParameterList& fsidyn, MPI_Comm comm)
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n setup_coupling in FSI::PartitionedImmersed ..." << std::endl;

  // for immersed fsi
  coupsfm_ = nullptr;
  matchingnodes_ = false;

  // enable debugging
  if (fsidyn.get<bool>("DEBUGOUTPUT"))
    debugwriter_ = std::make_shared<Utils::DebugWriter>(structure_field()->discretization());
}


void FSI::PartitionedImmersed::extract_previous_interface_solution()
{
  // not necessary in immersed fsi.
  // overrides version in fsi_paritioned with "do nothing".
}

FOUR_C_NAMESPACE_CLOSE
