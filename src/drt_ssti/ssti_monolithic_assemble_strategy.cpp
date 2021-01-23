/*----------------------------------------------------------------------*/
/*! \file
\brief Assemble strategy for monolithic SSTI

\level 2

*----------------------------------------------------------------------*/
#include "ssti_monolithic.H"
#include "ssti_monolithic_assemble_strategy.H"
#include "ssti_utils.H"

#include "../drt_adapter/ad_str_ssiwrapper.H"

#include "../drt_io/io_control.H"

#include "../drt_lib/drt_locsys.H"

#include "../drt_scatra/scatra_timint_meshtying_strategy_s2i.H"

#include "../linalg/linalg_matrixtransform.H"

/*----------------------------------------------------------------------*
 |                                                         constructors |
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBase::AssembleStrategyBase(Teuchos::RCP<const SSTI::SSTIMono> ssti_mono)
    : ssti_mono_(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlock::AssembleStrategyBlock(Teuchos::RCP<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBase(ssti_mono),
      block_position_scatra_(ssti_mono->GetBlockPositions(SSTI::Subproblem::scalar_transport)),
      block_position_thermo_(ssti_mono->GetBlockPositions(SSTI::Subproblem::thermo)),
      position_structure_(ssti_mono->GetBlockPositions(SSTI::Subproblem::structure)->at(0))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlockBlock::AssembleStrategyBlockBlock(
    Teuchos::RCP<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBlock(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategyBlockSparse::AssembleStrategyBlockSparse(
    Teuchos::RCP<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBlock(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::AssembleStrategySparse::AssembleStrategySparse(Teuchos::RCP<const SSTI::SSTIMono> ssti_mono)
    : AssembleStrategyBase(std::move(ssti_mono))
{
}

/*----------------------------------------------------------------------*
 |                            assemble scatra domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatradomain_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(scatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionScaTra()->size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(BlockPositionScaTra()->size()); ++jblock)
    {
      systemmatrix_block->Assign(BlockPositionScaTra()->at(iblock),
          BlockPositionScaTra()->at(jblock), LINALG::View,
          scatradomain_block->Matrix(iblock, jblock));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatradomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(scatradomain);

  // add scalar transport system matrix to global system matrix
  systemmatrix_block->Assign(BlockPositionScaTra()->at(0), BlockPositionScaTra()->at(0),
      LINALG::View, *scatradomain_sparse);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatradomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto scatradomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(scatradomain);

  // add scalar transport system matrix to global system matrix
  systemmatrix_sparse->Add(*scatradomain_sparse, false, 1.0, 0.0);
}

/*----------------------------------------------------------------------*
 |                         assemble structure domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseMatrix> structuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
  {
    systemmatrix_block->Assign(
        PositionStructure(), PositionStructure(), LINALG::View, *structuredomain);
  }
  else
  {
    AssembleStructureDomainMeshtying(
        systemmatrix_block->Matrix(PositionStructure(), PositionStructure()), structuredomain,
        false);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseMatrix> structuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
  {
    systemmatrix_block->Assign(
        PositionStructure(), PositionStructure(), LINALG::View, *structuredomain);
  }
  else
  {
    AssembleStructureDomainMeshtying(
        systemmatrix_block->Matrix(PositionStructure(), PositionStructure()), structuredomain,
        false);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseMatrix> structuredomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
    systemmatrix_sparse->Add(*structuredomain, false, 1.0, 1.0);
  else
    AssembleStructureDomainMeshtying(*systemmatrix_sparse, structuredomain, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleStructureDomainMeshtying(
    LINALG::SparseMatrix& systemmatrix_structure,
    Teuchos::RCP<LINALG::SparseMatrix> structuredomain, bool zero)
{
  /* Transform and assemble the structural matrix in the global system matrix block by block:
   * S_m: structural interior and master side dofs
   * S_ss: structural slave surface dofs
   * S_sl: structural slave line dofs
   *
   *       S_m  S_ss  S_sl
   *       --------------
   * S_m  |  a |  b |  c |
   * S_ss |  d |  e |  f |
   * S_sl |  g |  h |  i |
   *       --------------
   */

  // assemble derivs. of interior & master dofs w.r.t. interior & master dofs (block a)
  LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureCondensed(),
      *MapStructureCondensed(), 1.0, nullptr, nullptr, systemmatrix_structure, true, zero);

  // assemble  derivs. of surface slave dofs w.r.t. master & interior dofs (block d)
  LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureSlave(),
      *MapStructureCondensed(), 1.0, &(*StructureSlaveConverter()), nullptr, systemmatrix_structure,
      true, true);

  // assemble derivs. of master & interior w.r.t. surface slave dofs (block b)
  LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureCondensed(),
      *MapStructureSlave(), 1.0, nullptr, &(*StructureSlaveConverter()), systemmatrix_structure,
      true, true);

  // assemble derivs. of surface slave dofs w.r.t. surface slave dofs (block e)
  LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureSlave(),
      *MapStructureSlave(), 1.0, &(*StructureSlaveConverter()), &(*StructureSlaveConverter()),
      systemmatrix_structure, true, true);

  if (Meshtying3DomainIntersection())
  {
    // assemble derivs. of line slave dofs w.r.t. master & interior (block g)
    LINALG::MatrixLogicalSplitAndTransform()(*structuredomain,
        *MapStructureSlave3DomainIntersection(), *MapStructureCondensed(), 1.0,
        &(*StructureSlaveConverter3DomainIntersection()), nullptr, systemmatrix_structure, true,
        true);

    // assemble derivs. of master & interior w.r.t. line slave dofs (block c)
    LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureCondensed(),
        *MapStructureSlave3DomainIntersection(), 1.0, nullptr,
        &(*StructureSlaveConverter3DomainIntersection()), systemmatrix_structure, true, true);

    // assemble derivs. of line slave dof w.r.t. line slave dofs (block i)
    LINALG::MatrixLogicalSplitAndTransform()(*structuredomain,
        *MapStructureSlave3DomainIntersection(), *MapStructureSlave3DomainIntersection(), 1.0,
        &(*StructureSlaveConverter3DomainIntersection()),
        &(*StructureSlaveConverter3DomainIntersection()), systemmatrix_structure, true, true);

    // assemble derivs. of surface slave dofs w.r.t. line slave dofs (block f)
    LINALG::MatrixLogicalSplitAndTransform()(*structuredomain, *MapStructureSlave(),
        *MapStructureSlave3DomainIntersection(), 1.0, &(*StructureSlaveConverter()),
        &(*StructureSlaveConverter3DomainIntersection()), systemmatrix_structure, true, true);

    // assemble derivs. of line slave dofs w.r.t. surface slave dofs (block h)
    LINALG::MatrixLogicalSplitAndTransform()(*structuredomain,
        *MapStructureSlave3DomainIntersection(), *MapStructureSlave(), 1.0,
        &(*StructureSlaveConverter3DomainIntersection()), &(*StructureSlaveConverter()),
        systemmatrix_structure, true, true);
  }
}

/*----------------------------------------------------------------------*
 |                            assemble thermo domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermodomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermodomain_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(thermodomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionThermo()->size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(BlockPositionThermo()->size()); ++jblock)
    {
      systemmatrix_block->Assign(BlockPositionThermo()->at(iblock),
          BlockPositionThermo()->at(jblock), LINALG::View,
          thermodomain_block->Matrix(iblock, jblock));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermodomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermodomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(thermodomain);

  // add scalar transport system matrix to global system matrix
  systemmatrix_block->Assign(BlockPositionThermo()->at(0), BlockPositionThermo()->at(0),
      LINALG::View, *thermodomain_sparse);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermodomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto thermodomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(thermodomain);

  // add scalar transport system matrix to global system matrix
  systemmatrix_sparse->Add(*thermodomain_sparse, false, 1.0, 1.0);
}

/*----------------------------------------------------------------------*
 |                  assemble scatra-structure domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleScatraStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrastructuredomain_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(scatrastructuredomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionScaTra()->size()); ++iblock)
  {
    const auto scatrastructuredomain_subblock = scatrastructuredomain_block->Matrix(iblock, 0);

    // add entire block or assemble slave side to master side
    if (!InterfaceMeshtying())
    {
      systemmatrix_block->Assign(BlockPositionScaTra()->at(iblock), PositionStructure(),
          LINALG::View, scatrastructuredomain_subblock);
    }
    else
    {
      AssembleScatraStructureDomainMeshtying(
          systemmatrix_block->Matrix(BlockPositionScaTra()->at(iblock), PositionStructure()),
          scatrastructuredomain_subblock, false);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleScatraStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrastructuredomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrastructuredomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
  {
    systemmatrix_block->Assign(BlockPositionScaTra()->at(0), PositionStructure(), LINALG::View,
        *scatrastructuredomain_sparse);
  }
  else
  {
    AssembleScatraStructureDomainMeshtying(
        systemmatrix_block->Matrix(BlockPositionScaTra()->at(0), PositionStructure()),
        *scatrastructuredomain_sparse, false);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleScatraStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructuredomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto scatrastructuredomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrastructuredomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
    systemmatrix_sparse->Add(*scatrastructuredomain_sparse, false, 1.0, 1.0);
  else
    AssembleScatraStructureDomainMeshtying(
        *systemmatrix_sparse, *scatrastructuredomain_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleScatraStructureDomainMeshtying(
    LINALG::SparseMatrix& systemmatrix_scatra_structure,
    const LINALG::SparseMatrix& scatrastructuredomain, bool zero)
{
  // assemble interior and master-side columns of scatra-structure block into global
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(scatrastructuredomain, scatrastructuredomain.RangeMap(),
      *MapStructureCondensed(), 1.0, nullptr, nullptr, systemmatrix_scatra_structure, true, zero);

  // transform and assemble slave-side columns of scatra-structure block into global
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(scatrastructuredomain, scatrastructuredomain.RangeMap(),
      *MapStructureSlave(), 1.0, nullptr, &(*StructureSlaveConverter()),
      systemmatrix_scatra_structure, true, true);

  if (Meshtying3DomainIntersection())
  {
    // assemble derivs. of scatra w.r.t. structural line slave dofs
    LINALG::MatrixLogicalSplitAndTransform()(scatrastructuredomain,
        scatrastructuredomain.RangeMap(), *MapStructureSlave3DomainIntersection(), 1.0, nullptr,
        &(*StructureSlaveConverter3DomainIntersection()), systemmatrix_scatra_structure, true,
        true);
  }
}

/*----------------------------------------------------------------------*
 |               assemble scatra-structure interface into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleScatraStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrastructureinterface_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(scatrastructureinterface);

  // derive linearizations of master-side scatra fluxes w.r.t. master-side structural dofs
  // and assemble into auxiliary system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionScaTra()->size()); ++iblock)
  {
    const auto scatrastructureinterface_subblock =
        scatrastructureinterface_block->Matrix(iblock, 0);

    // assemble scatrastructureinterface_sparse into system matrix
    AssembleScatraStructureDomainMeshtying(
        systemmatrix_block->Matrix(BlockPositionScaTra()->at(iblock), PositionStructure()),
        scatrastructureinterface_subblock, true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleScatraStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrastructureinterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrastructureinterface);

  // assemble scatrastructureinterface_sparse into system matrix
  AssembleScatraStructureDomainMeshtying(
      systemmatrix_block->Matrix(BlockPositionScaTra()->at(0), PositionStructure()),
      *scatrastructureinterface_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleScatraStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrastructureinterface)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto scatrastructureinterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrastructureinterface);

  // assemble scatrastructureinterface_sparse into system matrix
  AssembleScatraStructureDomainMeshtying(
      *systemmatrix_sparse, *scatrastructureinterface_sparse, true);
}

/*----------------------------------------------------------------------*
 |                  assemble scatra-thermo interface into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleScatraThermoInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrathermointerface_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(scatrathermointerface);

  const auto scatrainterface = AllMaps()->MapInterface(MeshtyingScatra());

  LINALG::SparseMatrix masterderiv(*scatrainterface, 27, false, true);

  for (int i = 0; i < static_cast<int>(BlockPositionScaTra()->size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(BlockPositionThermo()->size()); ++j)
    {
      const auto scatrathermointerface_subblock = scatrathermointerface_block->Matrix(i, j);

      // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures
      // into system matrix
      LINALG::MatrixLogicalSplitAndTransform()(scatrathermointerface_subblock,
          scatrathermointerface_subblock.RangeMap(), scatrathermointerface_subblock.DomainMap(),
          1.0, nullptr, nullptr,
          systemmatrix_block->Matrix(BlockPositionScaTra()->at(i), BlockPositionThermo()->at(j)),
          true, true);

      // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures
      // into system matrix
      ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

      LINALG::MatrixLogicalSplitAndTransform()(scatrathermointerface_subblock,
          scatrathermointerface_subblock.RangeMap(),
          *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(), 1.0, nullptr, &thermo_converter,
          masterderiv, true, true);
    }
  }

  masterderiv.Complete(*MeshtyingThermo()->CouplingAdapter()->MasterDofMap(), *scatrainterface);

  const auto blockmasterderiv = masterderiv.Split<LINALG::DefaultBlockMatrixStrategy>(
      *AllMaps()->MapsThermo(), *AllMaps()->MapsScatra());

  blockmasterderiv->Complete();

  for (int i = 0; i < static_cast<int>(BlockPositionScaTra()->size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(BlockPositionThermo()->size()); ++j)
    {
      const auto masterderiv_subblock = blockmasterderiv->Matrix(i, j);

      // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      LINALG::MatrixLogicalSplitAndTransform()(masterderiv_subblock,
          masterderiv_subblock.RangeMap(), masterderiv_subblock.DomainMap(), 1.0, nullptr, nullptr,
          systemmatrix_block->Matrix(BlockPositionScaTra()->at(i), BlockPositionThermo()->at(j)),
          true, true);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleScatraThermoInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto scatrathermointerface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrathermointerface);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures into
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->RangeMap(), scatrathermointerface_sparse->DomainMap(), 1.0,
      nullptr, nullptr,
      systemmatrix_block->Matrix(BlockPositionScaTra()->at(0), BlockPositionThermo()->at(0)), true,
      true);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures into
  // system matrix
  ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

  LINALG::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->RangeMap(),
      *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(), 1.0, nullptr, &thermo_converter,
      systemmatrix_block->Matrix(BlockPositionScaTra()->at(0), BlockPositionThermo()->at(0)), true,
      true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleScatraThermoInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> scatrathermointerface)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto scatrathermointerface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(scatrathermointerface);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. slave temperatures into
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->RangeMap(), scatrathermointerface_sparse->DomainMap(), 1.0,
      nullptr, nullptr, *systemmatrix_sparse, true, true);

  // assemble linearizations of slave- and master side scatra fluxes w.r.t. master temperatures into
  // system matrix
  ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

  LINALG::MatrixLogicalSplitAndTransform()(*scatrathermointerface_sparse,
      scatrathermointerface_sparse->RangeMap(),
      *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(), 1.0, nullptr, &thermo_converter,
      *systemmatrix_sparse, true, true);
}

/*----------------------------------------------------------------------*
 |                  assemble structure-scatra domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleStructureScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurescatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto structurescatradomain_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(structurescatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionScaTra()->size()); ++iblock)
  {
    const auto structurescatradomain_subblock = structurescatradomain_block->Matrix(0, iblock);

    // add entire block or assemble slave side to master side
    if (!InterfaceMeshtying())
    {
      systemmatrix_block->Assign(PositionStructure(), BlockPositionScaTra()->at(iblock),
          LINALG::View, structurescatradomain_subblock);
    }
    else
    {
      AssembleStructureScatraDomainMeshtying(
          systemmatrix_block->Matrix(PositionStructure(), BlockPositionScaTra()->at(iblock)),
          structurescatradomain_subblock, false);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleStructureScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurescatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto structurescatradomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(structurescatradomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
  {
    systemmatrix_block->Assign(PositionStructure(), BlockPositionScaTra()->at(0), LINALG::View,
        *structurescatradomain_sparse);
  }
  else
  {
    AssembleStructureScatraDomainMeshtying(
        systemmatrix_block->Matrix(PositionStructure(), BlockPositionScaTra()->at(0)),
        *structurescatradomain_sparse, false);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleStructureScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurescatradomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto structurescatradomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(structurescatradomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
    systemmatrix_sparse->Add(*structurescatradomain_sparse, false, 1.0, 1.0);
  else
    AssembleStructureScatraDomainMeshtying(
        *systemmatrix_sparse, *structurescatradomain_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleStructureScatraDomainMeshtying(
    LINALG::SparseMatrix& systemmatrix_structure_scatra,
    const LINALG::SparseMatrix& structurescatradomain, bool zero)
{
  // assemble interior and master - side rows of structure - scatra block into global system
  // matrix
  LINALG::MatrixLogicalSplitAndTransform()(structurescatradomain, *MapStructureCondensed(),
      structurescatradomain.DomainMap(), 1.0, nullptr, nullptr, systemmatrix_structure_scatra, true,
      zero);

  // transform and assemble slave-side rows of structure-scatra block into global system
  // matrix
  LINALG::MatrixLogicalSplitAndTransform()(structurescatradomain, *MapStructureSlave(),
      structurescatradomain.DomainMap(), 1.0, &(*StructureSlaveConverter()), nullptr,
      systemmatrix_structure_scatra, true, true);

  if (Meshtying3DomainIntersection())
  {
    // assemble derivs. of structural surface line dofs & interior dofs w.r.t. scatra dofs
    LINALG::MatrixLogicalSplitAndTransform()(structurescatradomain,
        *MapStructureSlave3DomainIntersection(), structurescatradomain.DomainMap(), 1.0,
        &(*StructureSlaveConverter3DomainIntersection()), nullptr, systemmatrix_structure_scatra,
        true, true);
  }
}
/*----------------------------------------------------------------------*
 |                     assemble thermo-scatra domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleThermoScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermoscatradomain_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(thermoscatradomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionThermo()->size()); ++iblock)
  {
    for (int jblock = 0; jblock < static_cast<int>(BlockPositionScaTra()->size()); ++jblock)
    {
      const auto thermoscatradomain_subblock = thermoscatradomain_block->Matrix(iblock, jblock);
      LINALG::MatrixLogicalSplitAndTransform()(thermoscatradomain_subblock,
          thermoscatradomain_subblock.RangeMap(), thermoscatradomain_subblock.DomainMap(), 1.0,
          nullptr, nullptr,
          systemmatrix_block->Matrix(
              BlockPositionThermo()->at(iblock), BlockPositionScaTra()->at(jblock)),
          true, true);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleThermoScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatradomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermoscatradomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(thermoscatradomain);

  // add scalar transport system matrix to global system matrix
  LINALG::MatrixLogicalSplitAndTransform()(*thermoscatradomain_sparse,
      thermoscatradomain_sparse->RangeMap(), thermoscatradomain_sparse->DomainMap(), 1.0, nullptr,
      nullptr,
      systemmatrix_block->Matrix(BlockPositionThermo()->at(0), BlockPositionScaTra()->at(0)), true,
      true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleThermoScatraDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatradomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto thermoscatradomain_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(thermoscatradomain);

  // add scalar transport system matrix to global system matrix
  systemmatrix_sparse->Add(*thermoscatradomain_sparse, false, 1.0, 1.0);
}

/*----------------------------------------------------------------------*
 |                     assemble thermo-scatra domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleThermoScatraInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermoscatrainterface_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(thermoscatrainterface);

  LINALG::SparseMatrix masterflux(
      *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(), 27, false, true);

  for (int i = 0; i < static_cast<int>(BlockPositionThermo()->size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(BlockPositionScaTra()->size()); ++j)
    {
      const auto scatrathermointerface_subblock = thermoscatrainterface_block->Matrix(i, j);

      // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      LINALG::MatrixLogicalSplitAndTransform()(scatrathermointerface_subblock,
          scatrathermointerface_subblock.RangeMap(), scatrathermointerface_subblock.DomainMap(),
          1.0, nullptr, nullptr,
          systemmatrix_block->Matrix(BlockPositionThermo()->at(i), BlockPositionScaTra()->at(j)),
          true, true);

      // assemble linearizations of master side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

      LINALG::SparseMatrix slaveflux(*MeshtyingThermo()->BlockMapsSlave().Map(i), 27, false, true);

      SCATRA::MeshtyingStrategyS2I::ExtractMatrixRows(
          scatrathermointerface_subblock, slaveflux, *MeshtyingThermo()->BlockMapsSlave().Map(i));

      slaveflux.Complete(
          *AllMaps()->MapsScatra()->FullMap(), *MeshtyingThermo()->BlockMapsSlave().Map(i));

      LINALG::MatrixLogicalSplitAndTransform()(scatrathermointerface_subblock,
          *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(),
          scatrathermointerface_subblock.DomainMap(), -1.0, &thermo_converter, nullptr, masterflux,
          true, true);
    }
  }

  masterflux.Complete(
      *AllMaps()->MapsScatra()->FullMap(), *MeshtyingThermo()->CouplingAdapter()->MasterDofMap());

  const auto blockmasterflux = masterflux.Split<LINALG::DefaultBlockMatrixStrategy>(
      *AllMaps()->MapsScatra(), *AllMaps()->MapsThermo());

  blockmasterflux->Complete();

  for (int i = 0; i < static_cast<int>(BlockPositionThermo()->size()); ++i)
  {
    for (int j = 0; j < static_cast<int>(BlockPositionScaTra()->size()); ++j)
    {
      auto blockmasterflux_subblock = blockmasterflux->Matrix(i, j);

      // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
      // into system matrix
      LINALG::MatrixLogicalSplitAndTransform()(blockmasterflux_subblock,
          blockmasterflux_subblock.RangeMap(), blockmasterflux_subblock.DomainMap(), 1.0, nullptr,
          nullptr,
          systemmatrix_block->Matrix(BlockPositionThermo()->at(i), BlockPositionScaTra()->at(j)),
          true, true);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleThermoScatraInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermoscatrainterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermoscatrainterface);

  // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  LINALG::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      thermoscatrainterface_sparse->RangeMap(), thermoscatrainterface_sparse->DomainMap(), 1.0,
      nullptr, nullptr,
      systemmatrix_block->Matrix(BlockPositionThermo()->at(0), BlockPositionScaTra()->at(0)), true,
      true);

  // assemble linearizations of master side thermo fluxes w.r.t. slave and master side elch
  // into system matrix
  ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

  LINALG::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(),
      thermoscatrainterface_sparse->DomainMap(), -1.0, &thermo_converter, nullptr,
      systemmatrix_block->Matrix(BlockPositionThermo()->at(0), BlockPositionScaTra()->at(0)), true,
      true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleThermoScatraInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermoscatrainterface)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto thermoscatrainterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermoscatrainterface);

  // assemble linearizations of slave side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  LINALG::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      thermoscatrainterface_sparse->RangeMap(), thermoscatrainterface_sparse->DomainMap(), 1.0,
      nullptr, nullptr, *systemmatrix_sparse, true, true);

  // assemble linearizations of master side scatra fluxes w.r.t. slave and master side elch
  // into system matrix
  ADAPTER::CouplingSlaveConverter thermo_converter(*MeshtyingThermo()->CouplingAdapter());

  LINALG::MatrixLogicalSplitAndTransform()(*thermoscatrainterface_sparse,
      *MeshtyingThermo()->CouplingAdapter()->MasterDofMap(),
      thermoscatrainterface_sparse->DomainMap(), -1.0, &thermo_converter, nullptr,
      *systemmatrix_sparse, true, true);
}

/*----------------------------------------------------------------------*
 |                  assemble thermo-structure domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleThermoStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermostructuredomain_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(thermostructuredomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionThermo()->size()); ++iblock)
  {
    const auto thermostructuredomain_subblock = thermostructuredomain_block->Matrix(iblock, 0);

    // add entire block or assemble slave side to master side
    if (!InterfaceMeshtying())
    {
      systemmatrix_block->Assign(BlockPositionThermo()->at(iblock), PositionStructure(),
          LINALG::View, thermostructuredomain_subblock);
    }
    else
    {
      AssembleThermoStructureDomainMeshtying(
          systemmatrix_block->Matrix(BlockPositionThermo()->at(iblock), PositionStructure()),
          thermostructuredomain_subblock, false);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleThermoStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructuredomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermostructuredomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermostructuredomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
  {
    systemmatrix_block->Assign(BlockPositionThermo()->at(0), PositionStructure(), LINALG::View,
        *thermostructuredomain_sparse);
  }
  else
  {
    AssembleThermoStructureDomainMeshtying(
        systemmatrix_block->Matrix(BlockPositionThermo()->at(0), PositionStructure()),
        *thermostructuredomain_sparse, false);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleThermoStructureDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructuredomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto thermostructuredomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermostructuredomain);

  if (!InterfaceMeshtying())
    systemmatrix_sparse->Add(*thermostructuredomain_sparse, false, 1.0, 1.0);
  else
    AssembleThermoStructureDomainMeshtying(
        *systemmatrix_sparse, *thermostructuredomain_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleThermoStructureDomainMeshtying(
    LINALG::SparseMatrix& systemmatrix_thermo_structure,
    const LINALG::SparseMatrix& thermostructuredomain, bool zero)
{
  // assemble interior and master-side columns of scatra-structure block into global
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(thermostructuredomain, thermostructuredomain.RangeMap(),
      *MapStructureCondensed(), 1.0, nullptr, nullptr, systemmatrix_thermo_structure, true, zero);

  // transform and assemble slave-side columns of scatra-structure block into global
  // system matrix
  LINALG::MatrixLogicalSplitAndTransform()(thermostructuredomain, thermostructuredomain.RangeMap(),
      *MapStructureSlave(), 1.0, nullptr, &(*StructureSlaveConverter()),
      systemmatrix_thermo_structure, true, true);

  if (Meshtying3DomainIntersection())
  {
    // assemble derivs. of scatra w.r.t. structural line slave dofs
    LINALG::MatrixLogicalSplitAndTransform()(thermostructuredomain,
        thermostructuredomain.RangeMap(), *MapStructureSlave3DomainIntersection(), 1.0, nullptr,
        &(*StructureSlaveConverter3DomainIntersection()), systemmatrix_thermo_structure, true,
        true);
  }
}

/*----------------------------------------------------------------------*
 |               assemble thermo-structure interface into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleThermoStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermostructureinterface_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(thermostructureinterface);

  // derive linearizations of master-side scatra fluxes w.r.t. master-side structural dofs
  // and assemble into auxiliary system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionThermo()->size()); ++iblock)
  {
    const auto thermostructureinterface_subblock =
        thermostructureinterface_block->Matrix(iblock, 0);
    // assemble scatrastructureinterface_sparse into system matrix
    AssembleThermoStructureDomainMeshtying(
        systemmatrix_block->Matrix(BlockPositionThermo()->at(iblock), PositionStructure()),
        thermostructureinterface_subblock, true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleThermoStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto thermostructureinterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermostructureinterface);

  // assemble scatrastructureinterface_sparse into system matrix
  AssembleThermoStructureDomainMeshtying(
      systemmatrix_block->Matrix(BlockPositionThermo()->at(0), PositionStructure()),
      *thermostructureinterface_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleThermoStructureInterface(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> thermostructureinterface)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto thermostructureinterface_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(thermostructureinterface);

  // assemble scatrastructureinterface_sparse into system matrix
  AssembleThermoStructureDomainMeshtying(
      *systemmatrix_sparse, *thermostructureinterface_sparse, true);
}

/*----------------------------------------------------------------------*
 |                  assemble structure-thermo domain into system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::AssembleStructureThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurethermodomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto structurethermodomain_block =
      LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(structurethermodomain);

  // assemble blocks of scalar transport system matrix into global system matrix
  for (int iblock = 0; iblock < static_cast<int>(BlockPositionThermo()->size()); ++iblock)
  {
    const auto structurethermodomain_subblock = structurethermodomain_block->Matrix(0, iblock);

    // add entire block or assemble slave side to master side
    if (!InterfaceMeshtying())
    {
      systemmatrix_block->Assign(PositionStructure(), BlockPositionThermo()->at(iblock),
          LINALG::View, structurethermodomain_subblock);
    }
    else
    {
      AssembleStructureThermoDomainMeshtying(
          systemmatrix_block->Matrix(PositionStructure(), BlockPositionThermo()->at(iblock)),
          structurethermodomain_subblock, false);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::AssembleStructureThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurethermodomain)
{
  auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

  auto structurethermodomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(structurethermodomain);

  // add entire block or assemble slave side to master side
  if (InterfaceMeshtying())
  {
    AssembleStructureThermoDomainMeshtying(
        systemmatrix_block->Matrix(PositionStructure(), BlockPositionThermo()->at(0)),
        *structurethermodomain_sparse, false);
  }
  else
  {
    systemmatrix_block->Assign(PositionStructure(), BlockPositionThermo()->at(0), LINALG::View,
        *structurethermodomain_sparse);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::AssembleStructureThermoDomain(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix,
    Teuchos::RCP<LINALG::SparseOperator> structurethermodomain)
{
  auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

  auto structurethermodomain_sparse =
      LINALG::CastToSparseMatrixAndCheckSuccess(structurethermodomain);

  // add entire block or assemble slave side to master side
  if (!InterfaceMeshtying())
    systemmatrix_sparse->Add(*structurethermodomain_sparse, false, 1.0, 1.0);
  else
    AssembleStructureScatraDomainMeshtying(
        *systemmatrix_sparse, *structurethermodomain_sparse, true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleStructureThermoDomainMeshtying(
    LINALG::SparseMatrix& systemmatrix_structure_thermo,
    const LINALG::SparseMatrix& structurethermodomain, bool zero)
{
  // assemble interior and master - side rows of structure - scatra block into global system
  // matrix
  LINALG::MatrixLogicalSplitAndTransform()(structurethermodomain, *MapStructureCondensed(),
      structurethermodomain.DomainMap(), 1.0, nullptr, nullptr, systemmatrix_structure_thermo, true,
      zero);

  // transform and assemble slave-side rows of structure-scatra block into global system
  // matrix
  LINALG::MatrixLogicalSplitAndTransform()(structurethermodomain, *MapStructureSlave(),
      structurethermodomain.DomainMap(), 1.0, &(*StructureSlaveConverter()), nullptr,
      systemmatrix_structure_thermo, true, true);

  if (Meshtying3DomainIntersection())
  {
    // assemble derivs. of structural surface line dofs & interior dofs w.r.t. scatra dofs
    LINALG::MatrixLogicalSplitAndTransform()(structurethermodomain,
        *MapStructureSlave3DomainIntersection(), structurethermodomain.DomainMap(), 1.0,
        &(*StructureSlaveConverter3DomainIntersection()), nullptr, systemmatrix_structure_thermo,
        true, true);
  }
}

/*----------------------------------------------------------------------*
 |                       apply meshtying to the assembled system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockBlock::ApplyMeshtyingSystemMatrix(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix)
{
  if (InterfaceMeshtying())
  {
    auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

    ApplyMeshtyingSysMat(systemmatrix_block->Matrix(PositionStructure(), PositionStructure()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlockSparse::ApplyMeshtyingSystemMatrix(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix)
{
  if (InterfaceMeshtying())
  {
    // cast systemmatrix
    auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

    ApplyMeshtyingSysMat(systemmatrix_block->Matrix(PositionStructure(), PositionStructure()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::ApplyMeshtyingSystemMatrix(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix)
{
  if (InterfaceMeshtying())
  {
    // cast systemmatrix
    auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

    ApplyMeshtyingSysMat(*systemmatrix_sparse);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::ApplyMeshtyingSysMat(LINALG::SparseMatrix& systemmatrix_structure)
{
  // map for slave side structural degrees of freedom
  Teuchos::RCP<const Epetra_Map> slavemaps;
  if (Meshtying3DomainIntersection())
  {
    slavemaps = LINALG::MultiMapExtractor::MergeMaps(
        {MapStructureSlave(), MapStructureSlave3DomainIntersection()});
  }
  else
    slavemaps = MapStructureSlave();

  // subject slave-side rows of structural system matrix to pseudo Dirichlet conditions to
  // finalize structural meshtying
  const double one(1.0);
  for (int doflid_slave = 0; doflid_slave < slavemaps->NumMyElements(); ++doflid_slave)
  {
    // extract global ID of current slave-side row
    const int dofgid_slave = slavemaps->GID(doflid_slave);
    if (dofgid_slave < 0) dserror("Local ID not found!");

    // apply pseudo Dirichlet conditions to filled matrix, i.e., to local row and column
    // indices
    if (systemmatrix_structure.Filled())
    {
      const int rowlid_slave = systemmatrix_structure.RowMap().LID(dofgid_slave);
      if (rowlid_slave < 0) dserror("Global ID not found!");
      if (systemmatrix_structure.EpetraMatrix()->ReplaceMyValues(
              rowlid_slave, 1, &one, &rowlid_slave))
        dserror("ReplaceMyValues failed!");
    }

    // apply pseudo Dirichlet conditions to unfilled matrix, i.e., to global row and
    // column indices
    else if (systemmatrix_structure.EpetraMatrix()->InsertGlobalValues(
                 dofgid_slave, 1, &one, &dofgid_slave))
      dserror("InsertGlobalValues failed!");
  }
}

/*----------------------------------------------------------------------*
 |                                apply structural DBC to system matrix |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBlock::ApplyStructuralDBCSystemMatrix(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix)
{
  // locsys manager of strucutre
  const auto& locsysmanager_structure = StructureField()->LocsysManager();

  // map of strucutral Dirichlet BCs
  const auto dbcmap_structure = StructureField()->GetDBCMapExtractor()->CondMap();

  if (locsysmanager_structure == Teuchos::null)
    systemmatrix->ApplyDirichlet(*dbcmap_structure);
  else
  {
    auto systemmatrix_block = LINALG::CastToBlockSparseMatrixBaseAndCheckSuccess(systemmatrix);

    // apply structural Dirichlet conditions
    for (int iblock = 0; iblock < systemmatrix_block->Cols(); ++iblock)
    {
      locsysmanager_structure->RotateGlobalToLocal(
          Teuchos::rcp(&systemmatrix_block->Matrix(PositionStructure(), iblock), false));
      systemmatrix_block->Matrix(PositionStructure(), iblock)
          .ApplyDirichletWithTrafo(
              locsysmanager_structure->Trafo(), *dbcmap_structure, iblock == PositionStructure());
      locsysmanager_structure->RotateLocalToGlobal(
          Teuchos::rcp(&systemmatrix_block->Matrix(PositionStructure(), iblock), false));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategySparse::ApplyStructuralDBCSystemMatrix(
    Teuchos::RCP<LINALG::SparseOperator> systemmatrix)
{
  // locsys manager of strucutre
  const auto& locsysmanager_structure = StructureField()->LocsysManager();

  // map of strucutral Dirichlet BCs
  const auto& dbcmap_structure = StructureField()->GetDBCMapExtractor()->CondMap();

  // structural dof row map
  const auto& dofrowmap_structure = StructureField()->DofRowMap();

  if (locsysmanager_structure == Teuchos::null)
    systemmatrix->ApplyDirichlet(*dbcmap_structure);
  else
  {
    auto systemmatrix_sparse = LINALG::CastToSparseMatrixAndCheckSuccess(systemmatrix);

    // extract structural rows of global system matrix
    auto systemmatrix_structure =
        Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_structure, 27, false, true));
    LINALG::MatrixLogicalSplitAndTransform()(*systemmatrix_sparse, *dofrowmap_structure,
        systemmatrix->DomainMap(), 1.0, nullptr, nullptr, *systemmatrix_structure);
    systemmatrix_structure->Complete(systemmatrix->DomainMap(), *dofrowmap_structure);

    // apply structural Dirichlet conditions
    locsysmanager_structure->RotateGlobalToLocal(systemmatrix_structure);
    systemmatrix_structure->ApplyDirichletWithTrafo(
        locsysmanager_structure->Trafo(), *dbcmap_structure);
    locsysmanager_structure->RotateLocalToGlobal(systemmatrix_structure);

    // assemble structural rows of global system matrix back into global system matrix
    systemmatrix_sparse->Put(*systemmatrix_structure, 1.0, dofrowmap_structure);
  }
}

/*----------------------------------------------------------------------*
 |                                             assemble right-hand-side |
 *----------------------------------------------------------------------*/
void SSTI::AssembleStrategyBase::AssembleRHS(Teuchos::RCP<Epetra_Vector> RHS,
    Teuchos::RCP<Epetra_Vector> RHSscatra, Teuchos::RCP<const Epetra_Vector> RHSstructure,
    Teuchos::RCP<const Epetra_Vector> RHSthermo)
{
  // zero out RHS
  RHS->PutScalar(0.0);

  // assemble scalar transport right-hand side vector into monolithic right-hand side vector
  AllMaps()->MapsSubproblems()->InsertVector(
      RHSscatra, ssti_mono_->GetProblemPosition(Subproblem::scalar_transport), RHS);
  AllMaps()->MapsSubproblems()->InsertVector(
      RHSthermo, ssti_mono_->GetProblemPosition(Subproblem::thermo), RHS);

  if (!InterfaceMeshtying())
  {
    AllMaps()->MapsSubproblems()->AddVector(
        RHSstructure, ssti_mono_->GetProblemPosition(Subproblem::structure), RHS, -1.0);
  }
  else
  {
    // perform structural meshtying before assembling structural right-hand side vector into
    // monolithic right-hand side vector

    // make copy of structural right-hand side vector
    Epetra_Vector residual_structure(*RHSstructure);

    // transform slave-side part of structural right-hand side vector to master side
    Teuchos::RCP<Epetra_Vector> slavetomaster =
        StructuralMeshtying()->MapsInterfaceStructure()->InsertVector(
            StructuralMeshtying()->InterfaceCouplingAdapterStructure()->SlaveToMaster(
                StructuralMeshtying()->MapsInterfaceStructure()->ExtractVector(
                    residual_structure, 0)),
            1);

    if (Meshtying3DomainIntersection())
    {
      slavetomaster->Update(1.0,
          *(StructuralMeshtying()->MapsInterfaceStructure3DomainIntersection()->InsertVector(
              StructuralMeshtying()
                  ->InterfaceCouplingAdapterStructure3DomainIntersection()
                  ->SlaveToMaster(StructuralMeshtying()
                                      ->MapsInterfaceStructure3DomainIntersection()
                                      ->ExtractVector(residual_structure, 0)),
              1)),
          1.0);
    }

    // locsys manager of strucutre
    const auto& locsysmanager_structure = ssti_mono_->StructureField()->LocsysManager();

    // apply pseudo Dirichlet conditions to transformed slave-side part of structural right-hand
    // side vector
    const Teuchos::RCP<const Epetra_Vector> zeros =
        Teuchos::rcp(new Epetra_Vector(slavetomaster->Map()));
    if (locsysmanager_structure != Teuchos::null)
      locsysmanager_structure->RotateGlobalToLocal(slavetomaster);
    LINALG::ApplyDirichlettoSystem(
        slavetomaster, zeros, *ssti_mono_->StructureField()->GetDBCMapExtractor()->CondMap());
    if (locsysmanager_structure != Teuchos::null)
      locsysmanager_structure->RotateLocalToGlobal(slavetomaster);

    // assemble transformed slave-side part of structural right-hand side vector
    residual_structure.Update(1.0, *slavetomaster, 1.0);

    // zero out slave-side part of structural right-hand side vector
    StructuralMeshtying()->MapsInterfaceStructure()->PutScalar(residual_structure, 0, 0.0);
    if (Meshtying3DomainIntersection())
    {
      StructuralMeshtying()->MapsInterfaceStructure3DomainIntersection()->PutScalar(
          residual_structure, 0, 0.0);
    }

    // assemble final structural right-hand side vector into monolithic right-hand side vector
    AllMaps()->MapsSubproblems()->AddVector(
        residual_structure, ssti_mono_->GetProblemPosition(Subproblem::structure), *RHS, -1.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<SSTI::AssembleStrategyBase> SSTI::BuildAssembleStrategy(
    Teuchos::RCP<const SSTI::SSTIMono> ssti_mono, LINALG::MatrixType matrixtype_ssti,
    LINALG::MatrixType matrixtype_scatra)
{
  Teuchos::RCP<SSTI::AssembleStrategyBase> assemblestrategy = Teuchos::null;

  switch (matrixtype_ssti)
  {
    case LINALG::MatrixType::block_field:
    {
      switch (matrixtype_scatra)
      {
        case LINALG::MatrixType::block_condition:
        {
          assemblestrategy = Teuchos::rcp(new SSTI::AssembleStrategyBlockBlock(ssti_mono));
          break;
        }
        case LINALG::MatrixType::sparse:
        {
          assemblestrategy = Teuchos::rcp(new SSTI::AssembleStrategyBlockSparse(ssti_mono));
          break;
        }
        default:
        {
          dserror("unknown matrix type");
          break;
        }
      }
      break;
    }
    case LINALG::MatrixType::sparse:
    {
      assemblestrategy = Teuchos::rcp(new SSTI::AssembleStrategySparse(ssti_mono));
      break;
    }
    default:
    {
      dserror("unknown matrix type");
      break;
    }
  }

  return assemblestrategy;
}