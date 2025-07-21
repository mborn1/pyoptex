# Implementation Plan

This document outlines the plan to convert Python code to Cython for performance optimization.

## Epic 1: Cythonize `optimize.py`

**Objective:** Convert the `optimize` function in `src/pyoptex/doe/fixed_structure/splitk_plot/optimize.py` to a Cython implementation.

**Sub-tasks:**

1.  **Create `_optimize_cy.pyx`:**
    *   **Dependencies:** None
    *   **Rationale:** Create a new Cython file to house the converted `optimize` function. This follows the existing convention in the project.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/_optimize_cy.pyx`
    *   **Explicit Instructions:** Create a new file named `_optimize_cy.pyx` in the `src/pyoptex/doe/fixed_structure/splitk_plot/` directory. Copy the contents of `optimize.py` into this new file as a starting point.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** None
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The file `_optimize_cy.pyx` is created with the same content as `optimize.py`.

2.  **Add Cython Typing to `_optimize_cy.pyx`:**
    *   **Dependencies:** Task 1.1
    *   **Rationale:** Add static typing to the Cython code to enable performance optimizations.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/_optimize_cy.pyx`
    *   **Explicit Instructions:** Add cdef and cpdef keywords to define types for variables, function arguments, and return values. Use memoryviews for efficient handling of NumPy arrays.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** Identifying the correct types for all variables can be complex.
    *   **Security/Performance Considerations:** Incorrect typing can lead to performance degradation.
    *   **Acceptance Criteria:** The Cython code is fully typed and compiles without errors.

3.  **Update `setup.py` for Cython Compilation:**
    *   **Dependencies:** Task 1.1
    *   **Rationale:** The `setup.py` file needs to be updated to include the new `.pyx` file in the Cython compilation process.
    *   **Target File(s):** `setup.py`
    *   **Explicit Instructions:** Add a new `Extension` object for `src/pyoptex/doe/fixed_structure/splitk_plot/_optimize_cy.pyx`.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** Ensuring the build process is correctly configured.
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The project compiles successfully with the new Cython module.

4.  **Integrate Cython Module:**
    *   **Dependencies:** Task 1.2, 1.3
    *   **Rationale:** The existing Python code needs to be updated to import and use the new Cython function.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/optimize.py`
    *   **Explicit Instructions:** Modify `optimize.py` to import the Cython version of the `optimize` function from `_optimize_cy.pyx` and use it.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** None
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The application runs correctly using the Cython implementation.

## Epic 2: Cythonize `formulas.py`

**Objective:** Convert the Numba-accelerated functions in `src/pyoptex/doe/fixed_structure/splitk_plot/formulas.py` to a Cython implementation.

**Sub-tasks:**

1.  **Create `_formulas_cy.pyx`:**
    *   **Dependencies:** None
    *   **Rationale:** Create a new Cython file for the converted `formulas.py` functions.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/_formulas_cy.pyx`
    *   **Explicit Instructions:** Create a new file named `_formulas_cy.pyx` in `src/pyoptex/doe/fixed_structure/splitk_plot/` and copy the contents of `formulas.py` into it.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** None
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The file `_formulas_cy.pyx` is created with the same content as `formulas.py`.

2.  **Convert Numba Functions to Cython:**
    *   **Dependencies:** Task 2.1
    *   **Rationale:** Translate the Numba JIT-compiled functions to Cython, applying static typing for performance.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/_formulas_cy.pyx`
    *   **Explicit Instructions:** Convert each Numba-decorated function to a `cpdef` function in Cython. Replace Numba-specific calls with their Cython or NumPy equivalents.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** The numerical computations are complex and must be translated accurately.
    *   **Security/Performance Considerations:** The correctness of the numerical results is critical.
    *   **Acceptance Criteria:** The Cython functions produce the same results as the original Numba functions.

3.  **Update `setup.py` for `_formulas_cy.pyx`:**
    *   **Dependencies:** Task 2.1
    *   **Rationale:** Add the new `_formulas_cy.pyx` to the build process.
    *   **Target File(s):** `setup.py`
    *   **Explicit Instructions:** Add a new `Extension` object for `src/pyoptex/doe/fixed_structure/splitk_plot/_formulas_cy.pyx`.
    *   **Suggested Tools:** `edit_file`
    *   **Anticipated Challenges:** None
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The project compiles successfully.

4.  **Integrate Cython Formulas:**
    *   **Dependencies:** Task 2.2, 2.3
    *   **Rationale:** Update the codebase to use the new Cython-based formula functions.
    *   **Target File(s):** `src/pyoptex/doe/fixed_structure/splitk_plot/metric.py` (and any other files that import from `formulas.py`)
    *   **Explicit Instructions:** Change the import statements to bring in the functions from `_formulas_cy.pyx`.
    *   **Suggested Tools:** `grep_search`, `edit_file`
    *   **Anticipated Challenges:** Identifying all files that use the original `formulas.py` module.
    *   **Security/Performance Considerations:** None
    *   **Acceptance Criteria:** The application runs correctly with the new Cython formulas. 