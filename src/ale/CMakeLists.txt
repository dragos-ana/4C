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
    contact
    core
    fluid_ele
    global_data
    inpar
    mat
    mortar
    )

four_c_add_dependency(${AUTO_DEFINED_MODULE_NAME} ${_dependencies})
