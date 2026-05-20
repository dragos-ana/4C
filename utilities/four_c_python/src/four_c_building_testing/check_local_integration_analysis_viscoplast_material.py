# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
This script verifies the csv files written by the analysis framework of the local time integration of InelasticDefgradTransvIsotropElastViscoplast for consistency, incorporating material parameters from the simulation input file.
The Adaptive Estimate Interpolation algorithm may or may not be enabled (Ana, Schmidt, Wall: Accelerating Local Newton--Raphson Schemes in
    Computational Plasticity / Viscoplasticity).
The following verifications are performed:
    - Headers of timestep output files (Gauss point files and summary), Local Newton output files (Gauss point files) and eventually Adaptive Estimate Interpolation output files (Gauss point files).
    - Consistency of the values stored as Eval. (accumulated over the current timestep), and Total (accumulated over all timesteps), and consistency with the set maximum thresholds (e.g., maximum number of local iterations): for timestep output files (Gauss point files and summary).
    - Consistency between the summary timestep output file and all associated Gauss point output files -> Are the summary values the sums of all Gauss point values?.
    - Consistency of Gauss point output files for Local Newton data and eventually for Adaptive Estimate Interpolation data.
"""

import argparse
from pathlib import Path

import yaml
import csv
import pathlib
import re
import numpy as np


# global variables
TIMINT_OUTPUT_HEADERS = {
    "general": [
        "step",
        "time",
        "Eval. iterations (global)",
        "Total iterations (global)",
        "Eval. iterations (LNL)",
        "Total iterations (LNL)",
        "Eval. time (CU)",
        "Eval. time (CU)",
        "Total time (CU)",
        "Total time (CU)",
    ],
    # next headers only included if the simulation uses the adaptive estimate interpolation algorithm
    "aei_specific_counters_and_times": [
        "Eval. PPC iters (AEI)",
        "Eval. interp. iters (AEI)",
        "Eval. re-estimations (AEI)",
        "Total PPC iters (AEI)",
        "Total interp. iters (AEI)",
        "Total re-estimations (AEI)",
        "Eval. start. point det. time (AEI)",
        "Total start. point det. time (AEI)",
    ],
    "aei_specific_interp_points": [
        "Starting point (AEI)",
        "Interp. point opt. equiv. stress (AEI)",
    ],  # these are not contained within the summary file, only in the Gauss point output files
}
LNL_OUTPUT_HEADERS = {
    "general": [
        "step",
        "time",
        "Global iteration",
        "Local iteration",
        "Equivalent stress",
        "Plastic strain",
        "Error status",
        "Increment norm",
        "Is converged?",
        "Plastic strain increment",
        "Residual norm",
    ],
    "aei_specific": ["Current interpolation point"],
}
AEI_OUTPUT_HEADERS = [
    "step",
    "time",
    "Global iteration",
    "Local iteration",
    "Current interpolation point",
    "Lower interpolation bound",
    "Upper interpolation bound",
]
NUM_TOL = 1.0e-12  # tolerance used for checking whether two numbers are equal, or for checking whether a number is 0.0


def read_csv_as_dict(csv_file: Path, drop_columns=None) -> dict:
    """Reads csv file as dict.

    Args:
        csv_file: File to be read.
        drop_columns: Columns from the csv file to drop for the output dict.

    Returns:
        Data from the csv file excluding the specified columns
    """
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        data = {key: [] for key in reader.fieldnames if key not in (drop_columns or [])}
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data


def get_matching_timint_output_row_index(timint_output_file, step: int) -> int:
    """Gets the row index within a given timestep output file, given the timestep index.

    Args:
        timint_output_file: Timestep output file (either for specific Gauss point or summary).
        step: Timestep index.
    Returns:
        Row index within the timestep output file.
    """

    timint_dict = read_csv_as_dict(csv_file=timint_output_file)

    for i, val in enumerate(timint_dict["step"]):
        if int(val) == step:
            return int(i)

    raise Exception(
        f"No matching row index found for step {step} within {timint_output_file}"
    )


def get_num_of_local_iters_for_step_from_lnl_gp_output_file(
    lnl_output_gp_file: Path, step: int
) -> int:
    """Retrieves the total number of local iterations evaluated within a given timestep from a Local Newton output file.

    Args:
        lnl_output_gp_file: Gauss point output file for Local Newton data.
        step: Index of the considered timestep.

    Returns:
        Total number of evaluated Local Newton iterations within the specified timestep.
    """

    lnl_output_data = read_csv_as_dict(csv_file=lnl_output_gp_file)

    all_timestep_row_indices = []
    for row in range(len(lnl_output_data["step"])):
        if int(lnl_output_data["step"][row]) == int(step):
            all_timestep_row_indices.append(row)

    num_local_iters = 0
    for i in range(len(all_timestep_row_indices)):
        # track changes of global iterations
        if (
            i > 0
            and int(lnl_output_data["Local iteration"][all_timestep_row_indices[i]])
            == 0
        ):
            num_local_iters += int(
                lnl_output_data["Local iteration"][all_timestep_row_indices[i] - 1]
            )

    # also track the last global iteration within the considered timestep
    num_local_iters += int(
        lnl_output_data["Local iteration"][all_timestep_row_indices[-1]]
    )

    return num_local_iters


def get_current_interpolation_point_from_aei_output_gp_file(
    aei_output_gp_file: Path, step: int, global_iter: int, local_iter: int
) -> float:
    """Retrieves the current interpolation from an Adaptive Estimate Interpolation output file, given the timestep and both iteration indices.

    Args:
        aei_output_gp_file: Gauss point file for Adaptive Estimate Interpolation data.
        step: Index of the considered timestep.
        global_iter: Index of the considered global iteration.
        local_iter: Index of the considered local iteration.

    Returns:
        Current interpolation point after the adaptive estimate interpolation with the provided specification if this exists; else None.
    """

    aei_output_data = read_csv_as_dict(csv_file=aei_output_gp_file)

    all_matching_rows = []
    for row in range(len(aei_output_data["step"])):
        if (
            int(aei_output_data["step"][row]) == step
            and int(aei_output_data["Global iteration"][row]) == global_iter
            and int(aei_output_data["Local iteration"][row]) == local_iter
        ):
            all_matching_rows.append(row)

    if all_matching_rows:
        return aei_output_data["Current interpolation point"][
            all_matching_rows[-1]
        ]  # last interpolation iteration!
    else:
        return None


def are_equal_numbers(num_1: float, num_2: float) -> bool:
    """Verifies whether two numbers are equal using the specified global tolerance.

    Args:
        num_1: First number
        num_2: Second number

    Returns:
        Boolean: are the numbers equal (True)?
    """
    return abs(num_1 - num_2) < NUM_TOL


def retrieve_all_csv_files(
    csv_dir: Path, sim_name: str, use_adaptive_estimate_interpolation: bool
) -> dict:
    """Retrieves the csv files written by the analysis framework for the given directory and simulation name.

    Args:
        csv_dir: Directory of the output csv files.
        sim_name: Simulation name / output prefix.
        use_adaptive_estimate_interpolation: Is the adaptive estimate interpolation algorithm used?

    Returns:
        Dictionary containing the summary and Gauss point csv files; adaptive estimate interpolation files are only contained if the option is enabled.
    """

    # attempt to retrieve the csv files
    timint_output_summary_file_pattern = f"{sim_name}-timint_output.csv"
    timint_output_gp_files_pattern = rf"{sim_name}-timint_output_gp_\d+\.csv"
    aei_output_gp_files_pattern = rf"{sim_name}-aei_output_gp_\d+\.csv"
    lnl_output_gp_files_pattern = rf"{sim_name}-lnl_output_gp_\d+\.csv"

    timint_output_summary_file = None
    timint_output_gp_files = []
    aei_output_gp_files = []
    lnl_output_gp_files = []
    for f in csv_dir.glob("*.csv"):
        if f.name == timint_output_summary_file_pattern:
            timint_output_summary_file = f
        elif re.compile(timint_output_gp_files_pattern).fullmatch(f.name):
            timint_output_gp_files.append(f)
        elif re.compile(aei_output_gp_files_pattern).fullmatch(f.name):
            aei_output_gp_files.append(f)
        elif re.compile(lnl_output_gp_files_pattern).fullmatch(f.name):
            lnl_output_gp_files.append(f)

    if not timint_output_summary_file:
        raise FileNotFoundError(
            f"Summary file for time integration output {timint_output_summary_file_pattern} could not be found!"
        )
    if not timint_output_gp_files:
        raise FileNotFoundError(
            f"No Gauss point output files {timint_output_gp_files_pattern} could be found!"
        )
    if use_adaptive_estimate_interpolation and not aei_output_gp_files:
        raise FileNotFoundError(
            f"No Gauss point output files {aei_output_gp_files_pattern} could be found!"
        )
    if not lnl_output_gp_files:
        raise FileNotFoundError(
            f"No Gauss point output files {lnl_output_gp_files_pattern} could be found!"
        )

    # return result
    result = {
        "timint_output_summary_file": timint_output_summary_file,
        "timint_output_gp_files": sorted(timint_output_gp_files),
        "lnl_output_gp_files": sorted(lnl_output_gp_files),
    }
    if use_adaptive_estimate_interpolation:
        result["aei_output_gp_files"] = sorted(aei_output_gp_files)

    return result


def retrieve_material_params(sim_input_yaml: Path) -> dict:
    """Retrieves the material parameters related to the Local Newton, and to the Adaptive Estimate Interpolation if enabled.
    Note that the tested file should contain these sections and several other parameters, which will be verified whenever the parameters are required in the other routines of this script.

    Args:
        sim_input_yaml: Input yaml for the simulation.

    Returns:
        Dict containing Local Newton params, and possibly the Adaptive Estimate Interpolation params if this is enabled within the input file.
    """

    # read content of the yaml file
    yaml_content = None
    with open(sim_input_yaml) as stream:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc("Contents of the yaml file {sim_input_yaml} could not be read!")

    # retrieve the material parameters
    if "MATERIALS" not in yaml_content:
        raise ValueError(
            "The yaml file {sim_input_yaml} does not contain a MATERIALS section!"
        )
    materials = yaml_content["MATERIALS"]
    viscoplast_materials = []
    mat_name = "MAT_InelasticDefgradTransvIsotropElastViscoplast"
    for mat_block in materials:
        if mat_name in mat_block.keys():
            viscoplast_materials.append({mat_name: mat_block[mat_name]})
    if not viscoplast_materials:
        raise ValueError(
            f"Did not find material {mat_name} in materials block of {sim_input_yaml}!"
        )
    if len(viscoplast_materials) > 1:
        raise ValueError(
            f"Ambiguous choice for the viscoplastic material! There are two or more {mat_name} materials within the provided input yaml {sim_input_yaml}!"
        )
    viscoplast_material_params = viscoplast_materials[0]
    viscoplast_material_params = viscoplast_material_params[
        next(iter(viscoplast_material_params))
    ]
    if "LOCAL_NEWTON" not in viscoplast_material_params:
        raise ValueError(
            "Could not find LOCAL_NEWTON section within the viscoplastic material parameters!"
        )
    if "ADAPTIVE_ESTIMATE_INTERPOLATION" not in viscoplast_material_params:
        raise ValueError(
            "Could not find ADAPTIVE_ESTIMATE_INTERPOLATION section within the viscoplastic material parameters!"
        )
    if (
        "USE_ADAPTIVE_ESTIMATE_INTERPOLATION"
        not in viscoplast_material_params["ADAPTIVE_ESTIMATE_INTERPOLATION"]
    ):
        raise ValueError(
            "The parameter USE_ADAPTIVE_ESTIMATE_INTERPOLATION must be clearly specified within ADAPTIVE_ESTIMATE_INTERPOLATION!"
        )

    output_dict = {"LOCAL_NEWTON": viscoplast_material_params["LOCAL_NEWTON"]}
    if viscoplast_material_params["ADAPTIVE_ESTIMATE_INTERPOLATION"][
        "USE_ADAPTIVE_ESTIMATE_INTERPOLATION"
    ]:
        output_dict["ADAPTIVE_ESTIMATE_INTERPOLATION"] = viscoplast_material_params[
            "ADAPTIVE_ESTIMATE_INTERPOLATION"
        ]
    return output_dict


def check_timint_output_headers(
    csv_file: Path, is_summary_file: bool, use_adaptive_estimate_interpolation: bool
):
    """Verifies the headers of the timestep output csv files against the reference global variable.

    Args:
        csv_file: Csv timestep output file.
        is_summary_file: Is this a summary file over all GP?
        use_adaptive_estimate_interpolation: Is the adaptive estimate interpolation algorithm used?
    """

    with open(csv_file) as f:
        headers = next(csv.reader(f))
    reference_headers = TIMINT_OUTPUT_HEADERS["general"]
    if use_adaptive_estimate_interpolation:
        reference_headers = (
            reference_headers + TIMINT_OUTPUT_HEADERS["aei_specific_counters_and_times"]
        )
        if not is_summary_file:
            reference_headers = (
                reference_headers + TIMINT_OUTPUT_HEADERS["aei_specific_interp_points"]
            )

    extra = set(headers) - set(reference_headers)
    missing = set(reference_headers) - set(headers)
    if extra or missing:
        raise ValueError(
            f"Header mismatch in {csv_file}. "
            f"Unexpected columns: {extra}. "
            f"Missing columns: {missing}."
        )


def check_aei_output_headers(csv_file: Path):
    """Verifies the headers of the output csv files for adaptive estimate interpolation against the reference global variable.

    Args:
        csv_file: Csv output file for adaptive estimate interpolation.
    """
    with open(csv_file) as f:
        headers = next(csv.reader(f))
    reference_headers = AEI_OUTPUT_HEADERS

    extra = set(headers) - set(reference_headers)
    missing = set(reference_headers) - set(headers)
    if extra or missing:
        raise ValueError(
            f"Header mismatch in {csv_file}. "
            f"Unexpected columns: {extra}. "
            f"Missing columns: {missing}."
        )


def check_lnl_output_headers(csv_file: Path, use_adaptive_estimate_interpolation: bool):
    """Verifies the headers of the output csv files for Local Newton data.

    Args:
        csv_file: Csv output file for Local Newton data.
        use_adaptive_estimate_interpolation: Is the adaptive estimate interpolation used?
    """
    with open(csv_file) as f:
        headers = next(csv.reader(f))
    reference_headers = LNL_OUTPUT_HEADERS["general"]
    if use_adaptive_estimate_interpolation:
        reference_headers += LNL_OUTPUT_HEADERS["aei_specific"]

    extra = set(headers) - set(reference_headers)
    missing = set(reference_headers) - set(headers)
    if extra or missing:
        raise ValueError(
            f"Header mismatch in {csv_file}. "
            f"Unexpected columns: {extra}. "
            f"Missing columns: {missing}."
        )


def check_eval_total_values_in_timint_files(
    csv_file: Path, is_summary_file: bool, material_params: dict
):
    """Verifies whether the total values accumulated over all timesteps are consistent with the eval. values accumulated over the current timestep in the given timestep output file.
    Also verifies the eval. values: iterations are verified against their specified maxima, and we verify whether the computation times are non-zero.
    For the Gauss point files, the starting points and the interpolation points leading to optimal equivalent stresses are also verified for consistency.
    Note that the headers have to be verified prior to running this function!

    Args:
        csv_file: Csv output file for timestep output
        is_summary_file: Is this a summary file over all GP?
        material_params: Material parameters read from the input yaml.
    """

    def check_eval_total_consistent(eval_values: list, total_values: list):
        assert len(eval_values) > 0, "No evaluated values given for {csv_file}"
        assert len(eval_values) == len(total_values), (
            f"The number of evaluated values is {len(eval_values)} and for total values is {len(eval_values)} within {csv_file}"
        )
        for i in range(1, len(eval_values)):
            assert are_equal_numbers(
                total_values[i], total_values[i - 1] + eval_values[i]
            ), (
                f"Inconsistent total value update for evaluated value with index {i} within {csv_file}"
            )

        return True

    # read csv file
    data_dict = read_csv_as_dict(csv_file=csv_file)

    # verify general columns for constitutive update
    assert data_dict["step"] == list(range(1, len(data_dict["step"]) + 1)), (
        f"Step: {data_dict['step']} is not made up of consecutive integers starting at 1 for {csv_file}"
    )

    assert np.allclose(
        np.array(data_dict["time"]),
        np.array(range(1, len(data_dict["step"]) + 1)) * data_dict["time"][0],
    ), (
        f"Time: {np.array(data_dict['time'])}, Step * Timestep: {np.array(range(1, len(data_dict['step']) + 1)) * data_dict['time'][0]} for {csv_file} "
    )  # we only check single runs, no restarts with different timesteps

    # verify constitutive update counters; all must be smaller than the set maximum number of Local Newton iterations for the Gauss point files
    check_eval_total_consistent(
        eval_values=[int(x) for x in data_dict["Eval. iterations (global)"]],
        total_values=[int(x) for x in data_dict["Total iterations (global)"]],
    )
    eval_num_of_global_iters = data_dict["Eval. iterations (global)"]

    if not is_summary_file:
        if "MAX_ITER" not in material_params["LOCAL_NEWTON"]:
            raise ValueError(
                "MAX_ITER must be specified within the LOCAL_NEWTON section of the simulation input file!"
            )
        local_newton_max_iter = material_params["LOCAL_NEWTON"]["MAX_ITER"]
        for x_ind, x in enumerate(data_dict["Eval. iterations (LNL)"]):
            assert int(x) <= local_newton_max_iter * eval_num_of_global_iters[x_ind], (
                f"Invalid number of evaluated local iterations for file row {x_ind + 2}: number of iterations: {x}, maximum: {local_newton_max_iter * eval_num_of_global_iters[x_ind]} for {csv_file}"
            )

    check_eval_total_consistent(
        eval_values=[int(x) for x in data_dict["Eval. iterations (LNL)"]],
        total_values=[int(x) for x in data_dict["Total iterations (LNL)"]],
    )

    # verify consistent constitutive update times -> all must be higher than 0 and eval. and total values must be consistent
    assert not np.allclose(
        data_dict["Eval. time (CU)"],
        np.zeros(len(data_dict["Eval. time (CU)"])),
        atol=NUM_TOL,
    ), "All CU times are essentially 0.0 for {csv_file}"
    check_eval_total_consistent(
        eval_values=data_dict["Eval. time (CU)"],
        total_values=data_dict["Total time (CU)"],
    )

    # other routines for adaptive estimate interpolation
    if "ADAPTIVE_ESTIMATE_INTERPOLATION" in material_params:
        aei_params = material_params["ADAPTIVE_ESTIMATE_INTERPOLATION"]
        # verify counters -> all must be smaller than their set tolerances (checked for the Gauss point files), and eval. and total values must be consistent
        if not is_summary_file:
            if "MAX_NUM_PLASTIC_PRED_CONSTRUCT_ITERS" not in aei_params:
                raise ValueError(
                    "MAX_NUM_PLASTIC_PRED_CONSTRUCT_ITERS must be specified within the ADAPTIVE_ESTIMATE_INTERPOLATION section of the input file!"
                )
            max_num_ppc_iters = aei_params["MAX_NUM_PLASTIC_PRED_CONSTRUCT_ITERS"]
            for x_ind, x in enumerate(data_dict["Eval. PPC iters (AEI)"]):
                assert int(x) < max_num_ppc_iters * eval_num_of_global_iters[x_ind], (
                    "Inconsistent number of PPC iters (AEI) for file row {x_ind + 2}: {int(x)}, maximum: {max_num_ppc_iters * eval_num_of_global_iters[x_ind]} for {csv_file}"
                )

            if "MAX_NUM_ESTIMATE_INTERP_ITERS" not in aei_params:
                raise ValueError(
                    "MAX_NUM_ESTIMATE_INTERP_ITERS must be specified within the ADAPTIVE_ESTIMATE_INTERPOLATION section of the input file!"
                )
            max_num_estimate_interp_iters = aei_params["MAX_NUM_ESTIMATE_INTERP_ITERS"]
            for x_ind, x in enumerate(data_dict["Eval. interp. iters (AEI)"]):
                assert (
                    int(x)
                    < max_num_estimate_interp_iters * eval_num_of_global_iters[x_ind]
                ), (
                    f"Inconsistent number of estimate interpolation iters (AEI) for file row {x_ind + 2}: {int(x)}, maximum: {max_num_estimate_interp_iters * eval_num_of_global_iters[x_ind]} for {csv_file}"
                )

            if "MAX_NUM_REESTIMATIONS" not in aei_params:
                raise ValueError(
                    "MAX_NUM_REESTIMATIONS must be specified within the ADAPTIVE_ESTIMATE_INTERPOLATION section of the input file!"
                )
            max_num_reeestimations = aei_params["MAX_NUM_REESTIMATIONS"]
            for x_ind, x in enumerate(data_dict["Eval. re-estimations (AEI)"]):
                assert (
                    int(x) < max_num_reeestimations * eval_num_of_global_iters[x_ind]
                ), (
                    f"Inconsistent number of re-estimations for file row {x_ind + 2}: {int(x)}, maximum: {max_num_reeestimations * eval_num_of_global_iters[x_ind]} for {csv_file}"
                )

        check_eval_total_consistent(
            eval_values=[int(x) for x in data_dict["Eval. PPC iters (AEI)"]],
            total_values=[int(x) for x in data_dict["Total PPC iters (AEI)"]],
        )
        check_eval_total_consistent(
            eval_values=[int(x) for x in data_dict["Eval. interp. iters (AEI)"]],
            total_values=[int(x) for x in data_dict["Total interp. iters (AEI)"]],
        )
        check_eval_total_consistent(
            eval_values=[int(x) for x in data_dict["Eval. re-estimations (AEI)"]],
            total_values=[int(x) for x in data_dict["Total re-estimations (AEI)"]],
        )
        # verify consistent computation times -> all must be higher than 0 and eval. and total values must be consistent
        assert not np.allclose(
            data_dict["Eval. start. point det. time (AEI)"],
            np.zeros(len(data_dict["Eval. start. point det. time (AEI)"])),
            atol=NUM_TOL,
        ), (
            "All AEI starting point determination times are essentially 0.0 for {csv_file}"
        )
        check_eval_total_consistent(
            eval_values=data_dict["Eval. start. point det. time (AEI)"],
            total_values=data_dict["Total start. point det. time (AEI)"],
        )

        # verify consistent starting points for the adaptive estimate interpolation
        if not is_summary_file:
            if "STARTING_POINT_TYPE" not in aei_params:
                raise ValueError(
                    "STARTING_POINT_TYPE must be specified within ADAPTIVE_ESTIMATE_INTERPOLATION!"
                )
            match aei_params["STARTING_POINT_TYPE"]:
                # for user-set starting points, we know exactly which values to expect
                case "user_set":
                    if "USER_SET_STARTING_POINT" not in aei_params:
                        raise ValueError(
                            "USER_SET_STARTING_POINT must be specified within ADAPTIVE_ESTIMATE_INTERPOLATION!"
                        )
                    aei_user_set_starting_point = aei_params["USER_SET_STARTING_POINT"]
                    assert np.allclose(
                        np.array(data_dict["Starting point (AEI)"]),
                        aei_user_set_starting_point
                        * np.ones(len(data_dict["Starting point (AEI)"])),
                        atol=NUM_TOL,
                    ), (
                        f"Not all starting points match the set starting point {aei_user_set_starting_point} for {csv_file}"
                    )
                # for the optimal interpolation point based on the equivalent stress, we have a one time step delay -> the starting point for this time step, is the optimal point from the previous timestep
                case "optimal_equiv_stress":
                    assert np.allclose(
                        np.array(data_dict["Starting point (AEI)"][1:]),
                        np.array(
                            data_dict["Interp. point opt. equiv. stress (AEI)"][:-1]
                        ),
                        atol=NUM_TOL,
                    ), (
                        f"Not all starting points match the interpolation points associated with optimal equivalent stress for {csv_file}"
                    )
                case _:
                    raise ValueError(
                        f"STARTING_POINT_TYPE {aei_params['STARTING_POINT_TYPE']} not yet enabled for testing!"
                    )
            # verify that both starting points and optimal interpolation points are always between 0.0 and 1.0
            assert all(
                x >= 0.0 and x <= 1.0 for x in data_dict["Starting point (AEI)"]
            ), f"Not all starting points lie within [0.0, 1.0] for for {csv_file}"
            assert all(
                x >= 0.0 and x <= 1.0
                for x in data_dict["Interp. point opt. equiv. stress (AEI)"]
            ), (
                f"Not all interpolation points (optimal equivalent stress) lie within [0.0, 1.0] for {csv_file}"
            )


def check_timestep_output_summary_file_based_on_gp_files(
    timint_output_summary_file: Path, timint_output_gp_files: list[Path]
):
    """Verifies whether the summary timestep output file is obtained as a sum of the columns from the respective Gauss point files (except for the general columns such as step, time, and global iterations).

    Args:
        timint_output_summary_file: Timestep output file (summary).
        timint_output_gp_files: Timestep output files (for all Gauss points).
    """

    # define columns to not be summed up in the verification
    no_sum_columns = [
        "step",
        "time",
        "Eval. iterations (global)",
        "Total iterations (global)",
    ]  # these columns should not be summed up when reducing all Gauss point files!

    # reduce all Gauss point file outputs (i.e., sum up all relevant columns)
    reduced_gp_file_output_dict = None
    for f in timint_output_gp_files:
        data_dict = read_csv_as_dict(
            csv_file=f, drop_columns=TIMINT_OUTPUT_HEADERS["aei_specific_interp_points"]
        )
        if reduced_gp_file_output_dict is None:
            reduced_gp_file_output_dict = data_dict
        else:
            for key in data_dict:
                if key not in no_sum_columns:
                    reduced_gp_file_output_dict[key] = [
                        a + b
                        for a, b in zip(
                            reduced_gp_file_output_dict[key], data_dict[key]
                        )
                    ]
    # read summary file
    timint_output_summary_file_dict = read_csv_as_dict(
        csv_file=timint_output_summary_file
    )

    for key in timint_output_summary_file_dict:
        # verify equality
        reduced_values = np.array(reduced_gp_file_output_dict[key])
        summary_values = np.array(timint_output_summary_file_dict[key])

        assert np.allclose(summary_values, reduced_values), (
            f"Inconsistent values for key {key}: summary file {timint_output_summary_file}, timint_output_gp_files {timint_output_gp_files}"
        )


def check_aei_output_gp_file(
    aei_output_gp_file: Path, timint_output_gp_file: Path, material_params: dict
):
    """Goes through the detailed output for the adaptive estimate interpolation, and determines its consistency, e.g.,
    are the interpolation intervals adapted as expected?
    Cross-checking with the timestep output file is also performed.

    Args:
        aei_output_gp_file: Gauss point output file for adaptive estimate interpolation.
        timint_output_gp_file: Corresponding timestep output file.
        material_params: Material parameters read from the input yaml.
    """

    # read output Gauss point files
    aei_dict = read_csv_as_dict(csv_file=aei_output_gp_file)
    timint_dict = read_csv_as_dict(csv_file=timint_output_gp_file)

    # helper function, to assert equality of interpolation points / bounds
    def check_interpolation_points(
        lower_interpolation_bound: float,
        lower_interpolation_bound_ref: float,
        upper_interpolation_bound: float,
        upper_interpolation_bound_ref: float,
        current_interpolation_point: float,
        current_interpolation_point_ref: float,
    ):
        """Verifies the provided interpolation points / bounds against reference values."""
        assert are_equal_numbers(
            lower_interpolation_bound, lower_interpolation_bound_ref
        ), (
            f"Lower interpolation bound: {lower_interpolation_bound}, Reference value: {lower_interpolation_bound_ref} for {aei_output_gp_file}"
        )
        assert are_equal_numbers(
            upper_interpolation_bound, upper_interpolation_bound_ref
        ), (
            f"Upper interpolation bound: {upper_interpolation_bound}, Reference value: {upper_interpolation_bound_ref} for {aei_output_gp_file}"
        )
        assert are_equal_numbers(
            current_interpolation_point, current_interpolation_point_ref
        ), (
            f"current_interpolation_point: {current_interpolation_point}, Reference value: {current_interpolation_point_ref} for {aei_output_gp_file}"
        )

    # retrieve required AEI parameters
    aei_params = material_params["ADAPTIVE_ESTIMATE_INTERPOLATION"]
    if "INTERVAL_SCANNING_PARAM" not in aei_params:
        raise ValueError(
            "INTERVAL_SCANNING_PARAM must be provided within ADAPTIVE_ESTIMATE_INTERPOLATION in the simulation input file!"
        )
    interval_scanning_param = aei_params["INTERVAL_SCANNING_PARAM"]

    # verify consistency of timesteps using the corresponding timestep output file
    assert set(aei_dict["step"]).issubset(set(timint_dict["step"])), (
        f"The AEI file {aei_output_gp_file} contains different steps than the corresponding TIMINT file {timint_output_gp_file}"
    )

    # loop through the rows of the AEI file
    for i in range(len(aei_dict["step"])):
        # determine whether this is a new timestep and / or a new global iteration
        new_timestep = True
        if i > 0 and aei_dict["step"][i] == aei_dict["step"][i - 1]:
            new_timestep = False
        new_global_iter = True
        if i > 0 and int(aei_dict["Global iteration"][i]) == int(
            aei_dict["Global iteration"][i - 1]
        ):
            new_global_iter = False
        if new_timestep:
            assert new_global_iter, (
                f"Inconsistency in file row {i + 2} of {aei_output_gp_file}! A new timestep also means a new global iteration"
            )

        # verifications in case of new timestep
        if new_timestep:
            if i > 0:
                assert int(aei_dict["step"][i]) == int(aei_dict["step"][i - 1]) + 1, (
                    f"Inconsistent steps for file rows {i + 1} and {i + 2} in {aei_output_gp_file}"
                )
                assert are_equal_numbers(
                    aei_dict["time"][i],
                    aei_dict["time"][i - 1] + timint_dict["time"][0],
                ), (
                    f"Inconsistent times for file rows {i + 1} and {i + 2} in {aei_output_gp_file}"
                )

            assert int(aei_dict["Global iteration"][i]) == 0, (
                f"Inconsistent global iteration for file row {i + 2} -> must be 0 because this is a new timestep for {aei_output_gp_file}"
            )
            assert int(aei_dict["Local iteration"][i]) == 0, (
                f"Inconsistent local iteration for file row {i + 2} -> must be 0 because this is a new timestep for {aei_output_gp_file}"
            )
            check_interpolation_points(
                lower_interpolation_bound=aei_dict["Lower interpolation bound"][i],
                lower_interpolation_bound_ref=0.0,
                upper_interpolation_bound=aei_dict["Upper interpolation bound"][i],
                upper_interpolation_bound_ref=1.0,
                current_interpolation_point=aei_dict["Current interpolation point"][i],
                current_interpolation_point_ref=timint_dict["Starting point (AEI)"][
                    get_matching_timint_output_row_index(
                        step=aei_dict["step"][i],
                        timint_output_file=timint_output_gp_file,
                    )
                ],
            )
            if i > 0:
                assert int(aei_dict["Global iteration"][i - 1]) == int(
                    timint_dict["Eval. iterations (global)"][
                        get_matching_timint_output_row_index(
                            step=aei_dict["step"][i - 1],
                            timint_output_file=timint_output_gp_file,
                        )
                    ]
                ), (
                    f"Number of global iterations within {aei_output_gp_file} and {timint_output_gp_file} do not match for step {aei_dict['step'][i - 1]}"
                )

        # verifications in case of new global iteration
        if new_global_iter:
            if i > 0 and not new_timestep:
                assert (
                    int(aei_dict["Global iteration"][i])
                    == int(aei_dict["Global iteration"][i - 1]) + 1
                ), (
                    f"Global iterations within {aei_output_gp_file} should be consecutive, but they are not for file rows {i + 1} and {i + 2} for {aei_output_gp_file}"
                )
            assert int(aei_dict["Local iteration"][i]) == 0, (
                f"Local iteration must be 0 for file row {i + 2} because this is a new global iteration for {aei_output_gp_file}"
            )

            check_interpolation_points(
                lower_interpolation_bound=aei_dict["Lower interpolation bound"][i],
                lower_interpolation_bound_ref=0.0,
                upper_interpolation_bound=aei_dict["Upper interpolation bound"][i],
                upper_interpolation_bound_ref=1.0,
                current_interpolation_point=aei_dict["Current interpolation point"][i],
                current_interpolation_point_ref=timint_dict["Starting point (AEI)"][
                    get_matching_timint_output_row_index(
                        step=aei_dict["step"][i],
                        timint_output_file=timint_output_gp_file,
                    )
                ],
            )

        # verifications if this is not a new timestep and not a new global iteration
        else:
            assert (
                aei_dict["Local iteration"][i]
                <= timint_dict["Eval. iterations (LNL)"][
                    get_matching_timint_output_row_index(
                        step=aei_dict["step"][i],
                        timint_output_file=timint_output_gp_file,
                    )
                ]
            ), (
                "Inconsistent local iteration for file row {i + 2} for {aei_output_gp_file} and {timint_output_gp_file}"
            )

            # determine if this is a new local iteration: -> re-estimation triggered
            if int(aei_dict["Local iteration"][i]) != int(
                aei_dict["Local iteration"][i - 1]
            ):
                intermediate_point = 0.5 * (
                    aei_dict["Lower interpolation bound"][i - 1]
                    + aei_dict["Current interpolation point"][i - 1]
                )
                # --> three cases are now possible for re-estimation based on the intermediate interpolation point
                # CASE 1:The current interpolation point is updated to the intermediate point and the Local Newton proceeds; then the interpolation interval remains unchanged.
                # CASE 2: The intermediate point is unsuitable as an updated estimate and the lower bound is updated to the intermediate point; subsequently, the current interpolation point is reset based on the updated interval.
                # CASE 3: The intermediate point is unsuitable as an updated estimate and the upper bound is updated to the intermediate point; subsequently, the current interpolation point is reset based on the updated interval.

                current_interp_point_eq_intermediate = (
                    aei_dict["Current interpolation point"][i] == intermediate_point
                )
                lower_interp_bound_eq_intermediate = (
                    aei_dict["Lower interpolation bound"][i] == intermediate_point
                )
                upper_interp_bound_eq_intermediate = (
                    aei_dict["Upper interpolation bound"][i] == intermediate_point
                )
                assert (
                    len(
                        {
                            aei_dict["Current interpolation point"][i],
                            aei_dict["Lower interpolation bound"][i],
                            aei_dict["Upper interpolation bound"][i],
                        }
                    )
                    == 3
                ), (
                    "Expected all three interpolation values to be distinct: "
                    f"current={aei_dict['Current interpolation point'][i]}, "
                    f"lower={aei_dict['Lower interpolation bound'][i]}, "
                    f"upper={aei_dict['Upper interpolation bound'][i]}"
                )
                assert (
                    current_interp_point_eq_intermediate
                    or lower_interp_bound_eq_intermediate
                    or upper_interp_bound_eq_intermediate
                ), f"Either the current interpolation point {aei_dict['Current interpolation point'][i - 1]} or the lower bound {aei_dict['Lower interpolation bound'][i - 1]} have to be set to the intermediate point {intermediate_point} in the re-estimation, for file row {i + 2} of {aei_output_gp_file}"

                if current_interp_point_eq_intermediate:
                    check_interpolation_points(
                        lower_interpolation_bound=aei_dict["Lower interpolation bound"][
                            i
                        ],
                        lower_interpolation_bound_ref=aei_dict[
                            "Lower interpolation bound"
                        ][i - 1],
                        upper_interpolation_bound=aei_dict["Upper interpolation bound"][
                            i
                        ],
                        upper_interpolation_bound_ref=aei_dict[
                            "Upper interpolation bound"
                        ][i - 1],
                        current_interpolation_point=aei_dict[
                            "Current interpolation point"
                        ][i],
                        current_interpolation_point_ref=intermediate_point,
                    )

                elif lower_interp_bound_eq_intermediate:
                    updated_current_interp_point = (
                        intermediate_point
                        + interval_scanning_param
                        * (
                            aei_dict["Upper interpolation bound"][i - 1]
                            - intermediate_point
                        )
                    )
                    check_interpolation_points(
                        lower_interpolation_bound=aei_dict["Lower interpolation bound"][
                            i
                        ],
                        lower_interpolation_bound_ref=intermediate_point,
                        upper_interpolation_bound=aei_dict["Upper interpolation bound"][
                            i
                        ],
                        upper_interpolation_bound_ref=aei_dict[
                            "Upper interpolation bound"
                        ][i - 1],
                        current_interpolation_point=aei_dict[
                            "Current interpolation point"
                        ][i],
                        current_interpolation_point_ref=updated_current_interp_point,
                    )

                elif upper_interp_bound_eq_intermediate:
                    updated_current_interp_point = aei_dict[
                        "Lower interpolation bound"
                    ][i - 1] + interval_scanning_param * (
                        intermediate_point
                        - aei_dict["Lower interpolation bound"][i - 1]
                    )
                    check_interpolation_points(
                        lower_interpolation_bound=aei_dict["Lower interpolation bound"][
                            i
                        ],
                        lower_interpolation_bound_ref=aei_dict[
                            "Lower interpolation bound"
                        ][i - 1],
                        upper_interpolation_bound=aei_dict["Upper interpolation bound"][
                            i
                        ],
                        upper_interpolation_bound_ref=intermediate_point,
                        current_interpolation_point=aei_dict[
                            "Current interpolation point"
                        ][i],
                        current_interpolation_point_ref=updated_current_interp_point,
                    )

            else:  # we are still within the same estimate interpolation, so that means that the interpolation interval was adapted
                previous_interpolation_point = aei_dict["Current interpolation point"][
                    i - 1
                ]
                lower_interp_bound_updated = are_equal_numbers(
                    aei_dict["Lower interpolation bound"][i],
                    previous_interpolation_point,
                )
                upper_interp_bound_updated = are_equal_numbers(
                    aei_dict["Upper interpolation bound"][i],
                    previous_interpolation_point,
                )
                assert lower_interp_bound_updated or updated_current_interp_point, (
                    f"Inconsistent update in file row {i + 2} of {aei_output_gp_file}"
                )
                assert not (
                    lower_interp_bound_updated and upper_interp_bound_updated
                ), (
                    f"Inconsistency for file row {i + 2} of {aei_output_gp_file}! It cannot be that both bounds are updated to the same value!"
                )

                if lower_interp_bound_updated:
                    updated_current_interp_point = (
                        previous_interpolation_point
                        + interval_scanning_param
                        * (
                            aei_dict["Upper interpolation bound"][i - 1]
                            - previous_interpolation_point
                        )
                    )
                    check_interpolation_points(
                        lower_interpolation_bound=aei_dict["Lower interpolation bound"][
                            i
                        ],
                        lower_interpolation_bound_ref=previous_interpolation_point,
                        upper_interpolation_bound=aei_dict["Upper interpolation bound"][
                            i
                        ],
                        upper_interpolation_bound_ref=aei_dict[
                            "Upper interpolation bound"
                        ][i - 1],
                        current_interpolation_point=aei_dict[
                            "Current interpolation point"
                        ][i],
                        current_interpolation_point_ref=updated_current_interp_point,
                    )

                elif upper_interp_bound_updated:
                    updated_current_interp_point = aei_dict[
                        "Lower interpolation bound"
                    ][i - 1] + interval_scanning_param * (
                        previous_interpolation_point
                        - aei_dict["Lower interpolation bound"][i - 1]
                    )
                    check_interpolation_points(
                        lower_interpolation_bound=aei_dict["Lower interpolation bound"][
                            i
                        ],
                        lower_interpolation_bound_ref=aei_dict[
                            "Lower interpolation bound"
                        ][i - 1],
                        upper_interpolation_bound=aei_dict["Upper interpolation bound"][
                            i
                        ],
                        upper_interpolation_bound_ref=previous_interpolation_point,
                        current_interpolation_point=aei_dict[
                            "Current interpolation point"
                        ][i],
                        current_interpolation_point_ref=updated_current_interp_point,
                    )


def check_lnl_output_gp_file(
    lnl_output_gp_file: Path,
    timint_output_gp_file: Path,
    material_params: dict,
    aei_output_gp_file: Path = None,
):
    """Goes through the detailed output for the Local Newton Loop, and determines its consistency (internal consistency over local iterations, and cross-checking with the corresponding timestep output file and eventually with the adaptive estimate interpolation file).

    Args:
        lnl_output_gp_file: Gauss point output file for Local Newton.
        timint_output_gp_file: Corresponding Gauss point timestep output file.
        material_params: material parameters read from the input yaml.
        aei_output_gp_file: Gauss point output file for adaptive estimate interpolation.
    """

    # read output gp files
    lnl_dict = read_csv_as_dict(csv_file=lnl_output_gp_file)
    timint_dict = read_csv_as_dict(csv_file=timint_output_gp_file)

    # verify consistency of timesteps with the corresponding timestep output file
    assert set(lnl_dict["step"]).issubset(set(timint_dict["step"])), (
        f"The LNL file {lnl_output_gp_file} contains different steps than the corresponding TIMINT file {timint_output_gp_file}"
    )

    # loop through the rows of the Local Newton file
    for i in range(len(lnl_dict["step"])):
        # verify consistency with the adaptive estimate interpolation file
        if aei_output_gp_file is not None:
            current_interpolation_point_ref = (
                get_current_interpolation_point_from_aei_output_gp_file(
                    aei_output_gp_file=aei_output_gp_file,
                    global_iter=lnl_dict["Global iteration"][i],
                    local_iter=lnl_dict["Local iteration"][i],
                    step=lnl_dict["step"][i],
                )
            )
            if current_interpolation_point_ref:
                if i < len(lnl_dict["step"]) - 1:
                    assert are_equal_numbers(
                        lnl_dict["Current interpolation point"][i + 1],
                        current_interpolation_point_ref,
                    ), f"Current interpolation point for file row {i + 2} from {lnl_output_gp_file} does not match the one from {aei_output_gp_file}"

        # we cannot have convergence and errors are at the same time
        if int(lnl_dict["Is converged?"][i]):
            assert (
                int(lnl_dict["Error status"][i]) == 0
            ), f"Inconsistency in file row {i + 2} of {lnl_output_gp_file}: we cannot have both convergence and evaluation errors within the same iteration!"

        # determine whether this is a new timestep and / or a new global iteration
        new_timestep = True
        if i > 0 and int(lnl_dict["step"][i]) == int(lnl_dict["step"][i - 1]):
            new_timestep = False
        new_global_iter = True
        if i > 0 and int(lnl_dict["Global iteration"][i]) == int(
            lnl_dict["Global iteration"][i - 1]
        ):
            new_global_iter = False
        if new_timestep:
            assert new_global_iter, (
                f"Inconsistency in file row {i + 2} of {aei_output_gp_file}! A new timestep also means a new global iteration"
            )

        # verifications in case of new timestep
        if new_timestep:
            if i > 0:
                assert int(lnl_dict["step"][i]) == int(lnl_dict["step"][i - 1]) + 1, (
                    f"Inconsistent steps for file rows {i + 1} and {i + 2} in {lnl_output_gp_file}"
                )
                assert are_equal_numbers(
                    lnl_dict["time"][i],
                    lnl_dict["time"][i - 1] + timint_dict["time"][0],
                ), (
                    f"Inconsistent times for file rows {i + 1} and {i + 2} in {lnl_output_gp_file}"
                )

            assert int(lnl_dict["Global iteration"][i]) == 0, (
                f"Inconsistent global iteration for file row {i + 2} -> must be 0 because this is a new timestep for {lnl_output_gp_file}"
            )
            assert int(lnl_dict["Local iteration"][i]) == 0, (
                f"Inconsistent local iteration for file row {i + 2} -> must be 0 because this is a new timestep for {lnl_output_gp_file}"
            )
            if i > 0:
                assert int(lnl_dict["Global iteration"][i - 1]) == int(
                    timint_dict["Eval. iterations (global)"][
                        get_matching_timint_output_row_index(
                            step=lnl_dict["step"][i - 1],
                            timint_output_file=timint_output_gp_file,
                        )
                    ]
                ), (
                    f"Number of global iterations within {lnl_output_gp_file} and {timint_output_gp_file} do not match for file row {i + 1} of {lnl_output_gp_file}"
                )

                assert int(
                    get_num_of_local_iters_for_step_from_lnl_gp_output_file(
                        lnl_output_gp_file=lnl_output_gp_file,
                        step=lnl_dict["step"][i - 1],
                    )
                ) == int(
                    timint_dict["Eval. iterations (LNL)"][
                        get_matching_timint_output_row_index(
                            step=lnl_dict["step"][i - 1],
                            timint_output_file=timint_output_gp_file,
                        )
                    ]
                ), (
                    f"Mismatch between the number of evaluated local iterations for step {lnl_dict['step'][i - 1]} between {lnl_output_gp_file} and {timint_output_gp_file}"
                )

        # verifications in case of new global iteration
        if new_global_iter:
            if i > 0 and not new_timestep:
                assert (
                    int(lnl_dict["Global iteration"][i])
                    == int(lnl_dict["Global iteration"][i - 1]) + 1
                ), (
                    f"Global iterations within {lnl_output_gp_file} should be consecutive, but they are not for file rows {i + 1} and {i + 2} for {lnl_output_gp_file}"
                )
            assert int(lnl_dict["Local iteration"][i]) == 0, (
                f"Local iteration must be 0 for file row {i + 2} because this is a new global iteration for {lnl_output_gp_file}"
            )
            assert (
                int(lnl_dict["Is converged?"][i - 1]) == 1
                and int(lnl_dict["Error status"][i - 1]) == 0
            ), f"Inconsistency in file row {i + 2} of {lnl_output_gp_file}! The previous iteration should have converged and have no errors!"

        # verifications in the case that this is not a new timestep or a new global iteration
        else:
            if i > 0:
                assert int(lnl_dict["Local iteration"][i]) == int(
                    lnl_dict["Local iteration"][i - 1] + 1
                ), (
                    "Inconsistent local iteration update for file rows {i + 1} and {i + 2} for {lnl_output_gp_file} "
                )
                if aei_output_gp_file is not None:
                    if not are_equal_numbers(
                        lnl_dict["Current interpolation point"][i],
                        lnl_dict["Current interpolation point"][i - 1],
                    ):
                        assert (
                            int(lnl_dict["Error status"][i - 1]) > 0
                        ), f"A re-estimation took place in file row {i + 2} of {lnl_output_gp_file}, but there was no error in file row {i + 1}"


def cli():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="Verify the csv files written out by the local integration analysis framework for InelasticDefgradTransvIsotropElastViscoplast."
    )
    parser.add_argument(
        "sim_input_yaml", type=str, help="Input yaml file for the simulation"
    )
    parser.add_argument("sim_name", type=str, help="Output name for the simulation")
    parser.add_argument(
        "csv_dir",
        type=str,
        help="Directory for the csv files written by the analysis framework",
    )

    args = parser.parse_args()
    sim_input_yaml = Path(args.sim_input_yaml).resolve()
    sim_name = args.sim_name
    csv_dir = Path(args.csv_dir).resolve()

    print(f"\n{'=' * 70}")
    print(
        "Verifying the csv files written out by the local integration analysis framework for InelasticDefgradTransvIsotropElastViscoplast."
    )
    print(f"{'=' * 70}")
    print(f"Simulation input yaml: {sim_input_yaml}")
    print(f"Simulation output name: {sim_name}")
    print(f"Directory for csv files: {csv_dir}")
    print(f"{'=' * 70}\n")

    # verify existence of the input yaml and retrieve the material parameters
    if not sim_input_yaml.exists():
        raise FileNotFoundError(
            f"Simulation input yaml {sim_input_yaml} was not found!"
        )
    if not sim_input_yaml.is_file():
        raise IsADirectoryError(
            f"The simulation input yaml {sim_input_yaml} is not a file! Is it maybe a directory, a symlink, ...?"
        )
    if sim_input_yaml.suffix != ".yaml":
        raise ValueError(
            f"The given simulation input file {sim_input_yaml} is not a yaml file!"
        )
    material_params = retrieve_material_params(sim_input_yaml=sim_input_yaml)
    print("The following material parameters are specified: ")
    for k, v in material_params.items():
        print(f"{k}: ")
        for kk, vv in material_params[k].items():
            print(f"{' ' * 5}{kk}: {vv}")
    print(f"{'=' * 70}\n")

    # determine whether the adaptive estimate interpolation is used
    use_adaptive_estimate_interpolation = (
        "ADAPTIVE_ESTIMATE_INTERPOLATION" in material_params
    )

    # retrieve csv files
    if not csv_dir.exists():
        raise FileNotFoundError(f"{csv_dir} does not exist!")
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"{csv_dir} is not a directory!")
    csv_files = retrieve_all_csv_files(
        csv_dir=csv_dir,
        sim_name=sim_name,
        use_adaptive_estimate_interpolation=use_adaptive_estimate_interpolation,
    )
    print("Found the following csv files: ")
    for k, v in csv_files.items():
        if type(v) == pathlib.PosixPath:
            print(f"{k}: {str(v)}")
        elif type(v) == list:
            print(f"{k}: ")
            for l in v:
                print(f"- {l}")
        else:
            raise TypeError("Incompatible type {type(v)} for {k}")
    print(f"{'=' * 70}\n")

    # verify headers
    check_timint_output_headers(
        csv_file=csv_files["timint_output_summary_file"],
        is_summary_file=True,
        use_adaptive_estimate_interpolation=use_adaptive_estimate_interpolation,
    )
    for f in csv_files["timint_output_gp_files"]:
        check_timint_output_headers(
            csv_file=f,
            is_summary_file=False,
            use_adaptive_estimate_interpolation=use_adaptive_estimate_interpolation,
        )
    for f in csv_files["lnl_output_gp_files"]:
        check_lnl_output_headers(
            csv_file=f,
            use_adaptive_estimate_interpolation=use_adaptive_estimate_interpolation,
        )
    if "aei_output_gp_files" in csv_files:
        for f in csv_files["aei_output_gp_files"]:
            check_aei_output_headers(csv_file=f)
    print("Header verification successful!")
    print(f"{'=' * 70}\n")

    # verify consistency of the Eval. and Total columns tracking the accumulated variables
    check_eval_total_values_in_timint_files(
        csv_file=csv_files["timint_output_summary_file"],
        is_summary_file=True,
        material_params=material_params,
    )
    for f in csv_files["timint_output_gp_files"]:
        check_eval_total_values_in_timint_files(
            csv_file=f, is_summary_file=False, material_params=material_params
        )
    print("Eval. /  Total values verification successful!")
    print(f"{'=' * 70}\n")

    # verify consistency of summary timestep output file
    check_timestep_output_summary_file_based_on_gp_files(
        timint_output_summary_file=csv_files["timint_output_summary_file"],
        timint_output_gp_files=csv_files["timint_output_gp_files"],
    )
    print("Verification of the timestep output summary file successful!")
    print(f"{'=' * 70}\n")

    # verify consistency of the detailed Local Newton output Gauss point files
    for f in csv_files["lnl_output_gp_files"]:
        corresponding_timint_f = (
            f.parent / f"{str(f.name.replace('-lnl_output_gp_', '-timint_output_gp_'))}"
        )
        corresponding_aei_f = (
            (f.parent / f"{str(f.name.replace('-lnl_output_gp_', '-aei_output_gp_'))}")
            if use_adaptive_estimate_interpolation
            else None
        )
        if corresponding_timint_f not in csv_files["timint_output_gp_files"]:
            raise ValueError(
                f"Could not find corresponding timestep output file -timint_output_... for {f}"
            )
        if (
            use_adaptive_estimate_interpolation
            and corresponding_aei_f not in csv_files["aei_output_gp_files"]
        ):
            raise ValueError(
                f"Could not find corresponding AEI output file -aei_output_... for {f}"
            )

        check_lnl_output_gp_file(
            lnl_output_gp_file=f,
            aei_output_gp_file=corresponding_aei_f,
            timint_output_gp_file=corresponding_timint_f,
            material_params=material_params,
        )
    print(
        "Verification of the adaptive estimate interpolation Gauss point files successful!"
    )
    print(f"{'=' * 70}\n")

    # verify consistency of the adaptive estimate interpolation output Gauss point files
    if "aei_output_gp_files" in csv_files:
        for f in csv_files["aei_output_gp_files"]:
            corresponding_timint_f = (
                f.parent
                / f"{str(f.name.replace('-aei_output_gp_', '-timint_output_gp_'))}"
            )
            if corresponding_timint_f not in csv_files["timint_output_gp_files"]:
                raise ValueError(
                    f"Could not find corresponding timestep output file -timint_output_... for {f}"
                )
            check_aei_output_gp_file(
                aei_output_gp_file=f,
                timint_output_gp_file=corresponding_timint_f,
                material_params=material_params,
            )
        print(
            "Verification of the adaptive estimate interpolation Gauss point files successful!"
        )
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    cli()
