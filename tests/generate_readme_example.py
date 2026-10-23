# SPDX-License-Identifier: MIT
"""Compile the documented library flow, with a small application-owned system."""
import pathlib
import re
import sys

readme, output = map(pathlib.Path, sys.argv[1:])
snippet = re.search(r"```C\n(.*?)```", readme.read_text(), re.S).group(1)
includes = "\n".join(re.findall(r"^\s*#include.*$", snippet, re.M))
snippet = re.sub(r"^\s*#include.*$", "", snippet, flags=re.M)
# Fail at the documented operation that regresses, instead of running on with
# a pending error. All API calls in this front-door example return status.
snippet = re.sub(r"(HYPREDRV_\w+\([^;]*\));", r"HYPREDRV_SAFE_CALL(\1);", snippet)
snippet = snippet.replace(
    "HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());",
    "HYPREDRV_SAFE_CALL(HYPREDRV_Initialize());\n"
    "HYPRE_IJMatrix A = create_test_ijmatrix_1x1(2.0);\n"
    "HYPRE_IJVector b = create_test_ijvector_1x1(4.0);",
)
snippet = snippet.replace(
    "HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hdrv));",
    "HYPRE_Vector x = NULL;\n"
    "HYPREDRV_SAFE_CALL(HYPREDRV_LinearSystemGetSolution(hdrv, &x));\n"
    "ASSERT_EQ_DOUBLE(get_test_ijvector_1x1((HYPRE_IJVector)x), 2.0, 1e-12);\n"
    "HYPREDRV_SAFE_CALL(HYPREDRV_LinearSolverDestroy(hdrv));",
)
snippet = snippet.replace(
    "HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());",
    "HYPRE_IJMatrixDestroy(A);\nHYPRE_IJVectorDestroy(b);\n"
    "HYPREDRV_SAFE_CALL(HYPREDRV_Finalize());",
)
output.write_text(
    includes + '\n#include "test_ij_helpers.h"\n#include "HYPREDRV_utils.h"\n'
    + "int main(void)\n{\n" + snippet + "\nreturn 0;\n}\n"
)
