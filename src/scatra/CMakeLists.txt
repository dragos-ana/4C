# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

four_c_auto_define_module()

set(_dependencies
    # cmake-format: sortable
    adapter
    config
    core
    elch
    fluid
    fluid_turbulence
    global_data
    inpar
    mat
    mortar
    porofluid_pressure_based_elast_scatra
    scatra_ele
    )

four_c_add_dependency(${AUTO_DEFINED_MODULE_NAME} ${_dependencies})
