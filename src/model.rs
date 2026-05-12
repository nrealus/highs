use std::convert::{TryFrom, TryInto};
use std::ops::{Index, RangeBounds};
use std::ptr::{null, null_mut};

use highs_sys::*;

use crate::highs_ptr::{highs_call, HighsPtr};
use crate::options::HighsOptionValue;
use crate::problem::{bound_value, c, AsHighsMatrix, Col, Problem, Row};
use crate::status::{
    HighsIisBoundStatus, HighsIisStatus, HighsModelStatus, HighsStatus, InvalidStatus,
};

/// Whether to minimize or maximize the objective function.
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// Maximize the objective.
    Maximise = OBJECTIVE_SENSE_MAXIMIZE as isize,
    /// Minimize the objective.
    Minimise = OBJECTIVE_SENSE_MINIMIZE as isize,
}

/// The primal/dual solution returned by [`Model::get_solution`].
#[derive(Clone, Debug)]
pub struct Solution {
    pub(crate) colvalue: Vec<f64>,
    pub(crate) coldual: Vec<f64>,
    pub(crate) rowvalue: Vec<f64>,
    pub(crate) rowdual: Vec<f64>,
}

impl Solution {
    /// Primal variable values (in the order columns were added).
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    /// Dual variable values (reduced costs) for each column.
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    /// Values of the constraint expressions (Ax).
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
    /// Dual values (shadow prices) for each constraint.
    pub fn dual_rows(&self) -> &[f64] {
        &self.rowdual
    }
}

impl Index<Col> for Solution {
    type Output = f64;
    fn index(&self, col: Col) -> &f64 {
        &self.colvalue[col.0]
    }
}

/// An Irreducible Infeasible Subsystem, returned by [`Model::get_iis`].
#[derive(Clone, Debug)]
pub struct Iis {
    pub(crate) iis_cols: Vec<(Col, HighsIisBoundStatus)>,
    pub(crate) iis_rows: Vec<(Row, HighsIisBoundStatus)>,
    pub(crate) model_cols_iis_status: Vec<HighsIisStatus>,
    pub(crate) model_rows_iis_status: Vec<HighsIisStatus>,
}

impl Iis {
    /// Columns (variables) in the IIS with their bound status.
    pub fn columns(&self) -> &[(Col, HighsIisBoundStatus)] {
        &self.iis_cols
    }
    /// Rows (constraints) in the IIS with their bound status.
    pub fn rows(&self) -> &[(Row, HighsIisBoundStatus)] {
        &self.iis_rows
    }
    /// Whether the given column is part of the IIS.
    pub fn contains_column(&self, col: Col) -> bool {
        matches!(
            self.model_cols_iis_status.get(col.0),
            Some(HighsIisStatus::InConflict)
        )
    }
    /// Whether the given row is part of the IIS.
    pub fn contains_row(&self, row: Row) -> bool {
        matches!(
            self.model_rows_iis_status.get(row.0 as usize),
            Some(HighsIisStatus::InConflict)
        )
    }
    /// Whether the given column might be part of the IIS.
    pub fn contains_column_maybe(&self, col: Col) -> bool {
        matches!(
            self.model_cols_iis_status.get(col.0),
            Some(HighsIisStatus::MaybeInConflict)
        )
    }
    /// Whether the given row might be part of the IIS.
    pub fn contains_row_maybe(&self, row: Row) -> bool {
        matches!(
            self.model_rows_iis_status.get(row.0 as usize),
            Some(HighsIisStatus::MaybeInConflict)
        )
    }
}

/// A HiGHS model ready to be solved.
///
/// The model owns a HiGHS solver instance.
/// It can be solved multiple times (warm-starting is automatic) and modified between solves.
///
/// # Cloning
///
/// `Model` implements `Clone`. The clone preserves:
/// - All LP data (variables, constraints, bounds, costs, objective sense)
/// - Basis / warm-start state, if a solve has been run
///
/// The following are **not** preserved and must be re-applied by the caller on the clone:
/// - Solver options (`time_limit`, `solver`, `presolve`, `threads`, etc.)
///
/// # Example
/// ```
/// use highs::{Model, RowProblem, Sense, HighsModelStatus};
/// let mut pb = RowProblem::new();
/// let x = pb.add_column(1., 0..);
/// let y = pb.add_column(2., 0..);
/// pb.add_row(..=6., [(x, 3.), (y, 1.)]);
///
/// let mut model = Model::new(&pb, Sense::Maximise).unwrap();
/// model.solve().unwrap();
/// assert_eq!(model.status(), HighsModelStatus::Optimal);
/// let sol = model.get_solution();
/// assert_eq!(sol.columns(), &[0., 6.]);
/// ```
#[derive(Debug)]
pub struct Model {
    highs: HighsPtr,
}

impl Clone for Model {
    /// Clone this model.
    ///
    /// Preserves LP data, objective sense, and basis (warm-start state) if a solve has been run.
    ///
    /// **Not** preserved: solver options (`time_limit`, `solver`, `presolve`, `threads`, etc.).
    /// The caller is responsible for re-applying any options on the cloned model.
    fn clone(&self) -> Self {
        let cols = self.num_columns();
        let rows = self.num_rows();
        let nz = self.num_nz();

        // extract LP data
        let mut num_col: HighsInt = 0;
        let mut num_row: HighsInt = 0;
        let mut num_nz: HighsInt = 0;
        let mut sense: HighsInt = 0;
        let mut offset: f64 = 0.0;
        let mut col_cost = vec![0_f64; cols];
        let mut col_lower = vec![0_f64; cols];
        let mut col_upper = vec![0_f64; cols];
        let mut row_lower = vec![0_f64; rows];
        let mut row_upper = vec![0_f64; rows];
        let mut a_start = vec![0 as HighsInt; cols];
        let mut a_index = vec![0 as HighsInt; nz];
        let mut a_value = vec![0_f64; nz];

        unsafe {
            Highs_getLp(
                self.highs.ptr(),
                MATRIX_FORMAT_COLUMN_WISE,
                &mut num_col,
                &mut num_row,
                &mut num_nz,
                &mut sense,
                &mut offset,
                col_cost.as_mut_ptr(),
                col_lower.as_mut_ptr(),
                col_upper.as_mut_ptr(),
                row_lower.as_mut_ptr(),
                row_upper.as_mut_ptr(),
                a_start.as_mut_ptr(),
                a_index.as_mut_ptr(),
                a_value.as_mut_ptr(),
                null_mut(),
            );
        }

        // build new HiGHS instance
        let mut highs = HighsPtr::default();
        highs.make_quiet();

        unsafe {
            Highs_passLp(
                highs.mut_ptr(),
                num_col,
                num_row,
                num_nz,
                MATRIX_FORMAT_COLUMN_WISE,
                sense,
                offset,
                col_cost.as_ptr(),
                col_lower.as_ptr(),
                col_upper.as_ptr(),
                row_lower.as_ptr(),
                row_upper.as_ptr(),
                a_start.as_ptr(),
                a_index.as_ptr(),
                a_value.as_ptr(),
            );
        }

        // restore basis if a solve has been run
        if self.get_status() != HighsModelStatus::NotSet {
            let mut col_status = vec![0 as HighsInt; cols];
            let mut row_status = vec![0 as HighsInt; rows];
            let ok = unsafe {
                Highs_getBasis(
                    self.highs.ptr(),
                    col_status.as_mut_ptr(),
                    row_status.as_mut_ptr(),
                )
            };
            if ok == STATUS_OK {
                unsafe {
                    Highs_setBasis(highs.mut_ptr(), col_status.as_ptr(), row_status.as_ptr());
                }
            }
        }

        Self { highs }
    }
}

impl Model {
    /// Raw immutable pointer to the underlying HiGHS instance.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.highs.ptr()
    }

    /// Raw mutable pointer to the underlying HiGHS instance.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.highs.mut_ptr()
    }

    /// Create an empty model with no variables or constraints.
    ///
    /// Populate it incrementally using [`Model::add_col`] and [`Model::add_row`].
    pub fn default(sense: Sense) -> Self {
        let mut highs = HighsPtr::default();
        highs.make_quiet();
        let mut model = Self { highs };
        model.set_sense(sense);
        model
    }

    /// Create a model from a problem reference.
    /// Returns `Err` if HiGHS rejects the problem data.
    pub fn new<M: AsHighsMatrix + Default>(
        problem: &Problem<M>,
        sense: Sense,
    ) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        unsafe { Self::pass_problem(&mut highs, problem) }?;
        let mut model = Self { highs };
        model.set_sense(sense);
        model.make_quiet();
        Ok(model)
    }

    /// Pass the problem data to HiGHS via `Highs_passLp`.
    ///
    /// # Safety
    /// Caller must ensure `highs` is a valid, freshly-created (or cleared) HiGHS instance.
    unsafe fn pass_problem<M: AsHighsMatrix + Default>(
        highs: &mut HighsPtr,
        problem: &Problem<M>,
    ) -> Result<HighsStatus, HighsStatus> {
        let num_col = c(problem.num_cols());
        let num_row = c(problem.num_rows());
        let num_nz = c(problem.matrix.num_nz());
        let format = M::highs_format();
        let offset = 0.0_f64;

        let col_cost = problem.colcost.as_ptr();
        let col_lower = problem.collower.as_ptr();
        let col_upper = problem.colupper.as_ptr();
        let row_lower = problem.rowlower.as_ptr();
        let row_upper = problem.rowupper.as_ptr();
        let a_start = problem.matrix.astart().as_ptr();
        let a_index = problem.matrix.aindex().as_ptr();
        let a_value = problem.matrix.avalue().as_ptr();

        highs_call!(Highs_passLp(
            highs.mut_ptr(),
            num_col,
            num_row,
            num_nz,
            format,
            OBJECTIVE_SENSE_MINIMIZE,
            offset,
            col_cost,
            col_lower,
            col_upper,
            row_lower,
            row_upper,
            a_start,
            a_index,
            a_value
        ))
    }

    /// Run the solver.
    ///
    /// Returns `Ok(status)` if HiGHS ran without an API error.
    /// Returns `Err` only if the HiGHS API call itself failed.
    ///
    /// The model stays alive after the call. Call [`Model::get_solution`] or [`Model::get_iis`] to read results.
    /// Re-calling `solve` will warm-start automatically if the model was not modified.
    pub fn solve(&mut self) -> Result<HighsModelStatus, HighsStatus> {
        unsafe { highs_call!(Highs_run(self.highs.mut_ptr())) }?;
        Ok(self.get_status())
    }

    /// Number of columns (variables) in the model.
    pub fn num_columns(&self) -> usize {
        self.highs.num_cols().expect("Invalid number of columns")
    }

    /// Number of rows (constraints) in the model.
    pub fn num_rows(&self) -> usize {
        self.highs.num_rows().expect("Invalid number of rows")
    }

    /// Number of nonzeros in the constraint matrix.
    pub fn num_nz(&self) -> usize {
        self.highs.num_nz().expect("Invalid number of nonzeros")
    }

    /// The model status after the last solve.
    pub fn get_status(&self) -> HighsModelStatus {
        let raw = unsafe { Highs_getModelStatus(self.highs.ptr()) };
        HighsModelStatus::try_from(raw)
            .unwrap_or_else(|InvalidStatus(n)| panic!("HiGHS returned unexpected model status {n}"))
    }

    /// Objective value after the last solve.
    pub fn get_objective_value(&self) -> f64 {
        unsafe { Highs_getObjectiveValue(self.highs.unsafe_mut_ptr()) }
    }

    /// Primal/dual solution after a successful solve.
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_columns();
        let rows = self.num_rows();
        let mut colvalue = vec![0_f64; cols];
        let mut coldual = vec![0_f64; cols];
        let mut rowvalue = vec![0_f64; rows];
        let mut rowdual = vec![0_f64; rows];
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }
        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Compute an IIS (Irreducible Infeasible Subsystem) after an infeasible solve.
    pub fn get_iis(&self) -> Iis {
        let cols = self.num_columns();
        let rows = self.num_rows();
        let mut iis_numcol: HighsInt = 0;
        let mut iis_numrow: HighsInt = 0;
        let mut colindex: Vec<HighsInt> = vec![0; cols];
        let mut rowindex: Vec<HighsInt> = vec![0; rows];
        let mut colbound: Vec<HighsInt> = vec![0; cols];
        let mut rowbound: Vec<HighsInt> = vec![0; rows];
        let mut colstatus: Vec<HighsInt> = vec![0; cols];
        let mut rowstatus: Vec<HighsInt> = vec![0; rows];

        unsafe {
            Highs_getIis(
                self.highs.unsafe_mut_ptr(),
                &mut iis_numcol,
                &mut iis_numrow,
                colindex.as_mut_ptr(),
                rowindex.as_mut_ptr(),
                colbound.as_mut_ptr(),
                rowbound.as_mut_ptr(),
                colstatus.as_mut_ptr(),
                rowstatus.as_mut_ptr(),
            );
        }

        let nc = iis_numcol as usize;
        let nr = iis_numrow as usize;
        colindex.truncate(nc);
        rowindex.truncate(nr);
        colbound.truncate(nc);
        rowbound.truncate(nr);

        let iis_cols = colindex
            .into_iter()
            .zip(colbound)
            .map(|(i, b)| {
                (
                    Col(i.try_into().unwrap()),
                    HighsIisBoundStatus::try_from(b).unwrap(),
                )
            })
            .collect();
        let iis_rows = rowindex
            .into_iter()
            .zip(rowbound)
            .map(|(i, b)| (Row(i), HighsIisBoundStatus::try_from(b).unwrap()))
            .collect();
        let model_cols_iis_status = colstatus
            .into_iter()
            .map(|s| HighsIisStatus::try_from(s).unwrap())
            .collect();
        let model_rows_iis_status = rowstatus
            .into_iter()
            .map(|s| HighsIisStatus::try_from(s).unwrap())
            .collect();

        Iis {
            iis_cols,
            iis_rows,
            model_cols_iis_status,
            model_rows_iis_status,
        }
    }

    /// Set the optimization sense (min/max).
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe { Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as HighsInt) };
        assert_eq!(ret, STATUS_OK, "changeObjectiveSense failed");
    }

    /// Suppress all terminal / file output from HiGHS.
    pub fn make_quiet(&mut self) {
        self.highs.make_quiet();
    }

    /// Set a HiGHS solver option by name.
    ///
    /// See <https://ergo-code.github.io/HiGHS/dev/options/definitions/> for available options.
    pub fn set_option<S: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: S, value: V) {
        self.highs.set_option(option, value);
    }

    /// Add a new constraint to the live model.
    ///
    /// Returns the new [`Row`] index, or `Err` on an API error.
    pub fn add_row<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        bounds: B,
        col_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Result<Row, HighsStatus> {
        let (cols, factors): (Vec<_>, Vec<_>) = col_factors.into_iter().unzip();
        let col_indices: Vec<HighsInt> = cols.iter().map(|c| c.0.try_into().unwrap()).collect();
        unsafe {
            highs_call!(Highs_addRow(
                self.highs.mut_ptr(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                col_indices.len().try_into().unwrap(),
                col_indices.as_ptr(),
                factors.as_ptr()
            ))
        }?;
        Ok(Row((self.highs.num_rows()? - 1) as HighsInt))
    }

    /// Add a new continuous variable to the live model.
    ///
    /// Returns the new [`Col`] index, or `Err` on an API error.
    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        cost: f64,
        bounds: B,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Result<Col, HighsStatus> {
        let (rows, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();
        let row_indices: Vec<HighsInt> = rows.iter().map(|r| r.0).collect();
        unsafe {
            highs_call!(Highs_addCol(
                self.highs.mut_ptr(),
                cost,
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                row_indices.len().try_into().unwrap(),
                row_indices.as_ptr(),
                factors.as_ptr()
            ))
        }?;
        Ok(Col(self.highs.num_cols()? - 1))
    }

    /// Returns `(cost, lower, upper)` for a single column via `Highs_getColsByRange`.
    fn get_col_data(&self, col: Col) -> (f64, f64, f64) {
        let mut num_col: HighsInt = 0;
        let mut num_nz: HighsInt = 0;
        let mut cost = 0_f64;
        let mut lower = 0_f64;
        let mut upper = 0_f64;
        let idx = col.0 as HighsInt;
        unsafe {
            Highs_getColsByRange(
                self.highs.ptr(),
                idx,
                idx,
                &mut num_col,
                &mut cost,
                &mut lower,
                &mut upper,
                &mut num_nz,
                null_mut(),
                null_mut(),
                null_mut(),
            );
        }
        (cost, lower, upper)
    }

    /// Get the current bounds `(lower, upper)` of a column.
    pub fn get_column_bounds(&self, col: Col) -> (f64, f64) {
        let (_, lower, upper) = self.get_col_data(col);
        (lower, upper)
    }

    /// Get the current objective coefficient of a column.
    pub fn get_column_cost(&self, col: Col) -> f64 {
        let (cost, _, _) = self.get_col_data(col);
        cost
    }

    /// Update the objective coefficient of a column.
    pub fn change_column_cost(&mut self, col: Col, cost: f64) {
        unsafe {
            highs_call!(Highs_changeColCost(
                self.highs.mut_ptr(),
                col.0 as HighsInt,
                cost
            ))
            .expect("Highs_changeColCost failed");
        }
    }

    /// Update the bounds of a column.
    pub fn change_column_bounds<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col: Col,
        bounds: B,
    ) {
        unsafe {
            highs_call!(Highs_changeColBounds(
                self.highs.mut_ptr(),
                col.0 as HighsInt,
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)
            ))
            .expect("Highs_changeColBounds failed");
        }
    }

    /// Replace the entire model with new problem data (clears and re-passes).
    ///
    /// Useful to reuse the same `Model` allocation for a different problem
    /// without the overhead of `Highs_create` / `Highs_destroy`.
    pub fn overwrite<M: AsHighsMatrix + Default>(
        &mut self,
        problem: &Problem<M>,
        sense: Sense,
    ) -> Result<(), HighsStatus> {
        unsafe { highs_call!(Highs_clearModel(self.highs.mut_ptr())) }?;
        unsafe { Self::pass_problem(&mut self.highs, problem) }?;
        self.set_sense(sense);
        Ok(())
    }

    /// Reset all solver state while keeping the problem data.
    ///
    /// Call this to force a cold restart on the next solve.
    pub fn clear_solver(&mut self) -> Result<HighsStatus, HighsStatus> {
        unsafe { highs_call!(Highs_clearSolver(self.highs.mut_ptr())) }
    }

    /// Remove all variables and constraints (but keep the HiGHS instance alive).
    pub fn clear_model(&mut self) -> Result<HighsStatus, HighsStatus> {
        unsafe { highs_call!(Highs_clearModel(self.highs.mut_ptr())) }
    }

    /// Provide an initial primal/dual solution as a warm-start hint.
    ///
    /// Each slice, if `Some`, must have the matching length (`num_cols` / `num_rows`).
    /// Pass `None` for any component you don't want to set.
    pub fn set_solution(
        &mut self,
        col_values: Option<&[f64]>,
        row_values: Option<&[f64]>,
        col_duals: Option<&[f64]>,
        row_duals: Option<&[f64]>,
    ) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_setSolution(
                self.highs.mut_ptr(),
                col_values.map(|s| s.as_ptr()).unwrap_or(null()),
                row_values.map(|s| s.as_ptr()).unwrap_or(null()),
                col_duals.map(|s| s.as_ptr()).unwrap_or(null()),
                row_duals.map(|s| s.as_ptr()).unwrap_or(null())
            ))
        }?;
        Ok(())
    }
}
