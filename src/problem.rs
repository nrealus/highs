use std::convert::TryInto;
use std::ops::{Bound, RangeBounds};
use std::os::raw::c_int;

use highs_sys::HighsInt;

pub(crate) fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

pub(crate) fn c(n: usize) -> HighsInt {
    n.try_into().expect("size too large for HiGHS")
}

/// A variable (column) index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Col(pub(crate) usize);

impl Col {
    pub fn index(self) -> usize {
        self.0
    }
}

impl<T: Into<usize>> From<T> for Col {
    fn from(v: T) -> Self {
        Col(v.into())
    }
}

/// A constraint (row) index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Row(pub(crate) c_int);

impl Row {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Abstraction over CSC (column-wise) and CSR (row-wise) sparse matrices.
///
/// Implemented by [`ColMatrix`] and [`RowMatrix`].
/// Lets [`crate::Model::new`] pass either matrix format directly to HiGHS without any conversion.
pub trait AsHighsMatrix {
    /// The HiGHS matrix format constant.
    fn highs_format() -> HighsInt;

    /// `a_start` array: length = number of columns for CSC, number of rows for CSR.
    fn astart(&self) -> &[c_int];

    /// `a_index` array: row indices (CSC) or column indices (CSR) of each nonzero.
    fn aindex(&self) -> &[c_int];

    /// `a_value` array: values of each nonzero, parallel to `a_index`.
    fn avalue(&self) -> &[f64];

    /// Number of nonzeros.
    fn num_nz(&self) -> usize {
        self.avalue().len()
    }
}

/// Compressed sparse column (column-wise) constraint matrix.
///
/// Built column-by-column: first declare all rows, then add columns together
/// with their row coefficients via [`ColProblem::add_column`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColMatrix {
    /// `astart[j]` = start of column `j` in `aindex`/`avalue`. Length = num_cols.
    pub(crate) astart: Vec<c_int>,
    /// Row indices of each nonzero, parallel to `avalue`.
    pub(crate) aindex: Vec<c_int>,
    /// Values of each nonzero, parallel to `aindex`.
    pub(crate) avalue: Vec<f64>,
}

impl AsHighsMatrix for ColMatrix {
    fn highs_format() -> HighsInt {
        highs_sys::MATRIX_FORMAT_COLUMN_WISE
    }
    fn astart(&self) -> &[c_int] {
        &self.astart
    }
    fn aindex(&self) -> &[c_int] {
        &self.aindex
    }
    fn avalue(&self) -> &[f64] {
        &self.avalue
    }
}

/// Compressed sparse row (row-wise) constraint matrix.
///
/// Built row-by-row: first declare columns (variables), then add constraints
/// one at a time via [`RowProblem::add_row`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowMatrix {
    /// `astart[i]` = start of row `i` in `aindex`/`avalue`.
    pub(crate) astart: Vec<c_int>,
    /// Column indices of each nonzero, parallel to `avalue`.
    pub(crate) aindex: Vec<c_int>,
    /// Values of each nonzero, parallel to `aindex`.
    pub(crate) avalue: Vec<f64>,
}

impl AsHighsMatrix for RowMatrix {
    fn highs_format() -> HighsInt {
        highs_sys::MATRIX_FORMAT_ROW_WISE
    }
    fn astart(&self) -> &[c_int] {
        &self.astart
    }
    fn aindex(&self) -> &[c_int] {
        &self.aindex
    }
    fn avalue(&self) -> &[f64] {
        &self.avalue
    }
}

/// A complete optimization problem parameterized over its matrix storage format.
///
/// Use [`ColProblem`] when you know your constraints upfront and add variables
/// one by one, or [`RowProblem`] when you add constraints one by one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem<MATRIX = ColMatrix> {
    pub(crate) colcost: Vec<f64>,
    pub(crate) collower: Vec<f64>,
    pub(crate) colupper: Vec<f64>,

    pub(crate) rowlower: Vec<f64>,
    pub(crate) rowupper: Vec<f64>,

    pub(crate) matrix: MATRIX,
}

impl<MATRIX: Default + AsHighsMatrix> Problem<MATRIX> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_cols(&self) -> usize {
        self.colcost.len()
    }

    pub fn num_rows(&self) -> usize {
        self.rowlower.len()
    }

    /// Update the objective coefficient of a variable.
    pub fn change_column_cost(&mut self, col: Col, cost: f64) {
        self.colcost[col.0] = cost;
    }

    /// Update the bounds of a variable.
    pub fn change_column_bounds<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col: Col,
        bounds: B,
    ) {
        self.collower[col.0] = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        self.colupper[col.0] = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
    }

    /// Get the bounds `(lower, upper)` of a variable.
    pub fn get_column_bounds(&self, col: Col) -> (f64, f64) {
        (self.collower[col.0], self.colupper[col.0])
    }

    /// Get the objective coefficient of a variable.
    pub fn get_column_cost(&self, col: Col) -> f64 {
        self.colcost[col.0]
    }

    pub(crate) fn push_column_data<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        cost: f64,
        bounds: B,
    ) {
        self.colcost.push(cost);
        self.collower
            .push(bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY));
        self.colupper
            .push(bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY));
    }

    pub(crate) fn push_row_bounds<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        bounds: B,
    ) -> Row {
        let r = Row(self.num_rows().try_into().expect("too many rows"));
        self.rowlower
            .push(bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY));
        self.rowupper
            .push(bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY));
        r
    }
}

pub type ColProblem = Problem<ColMatrix>;

impl Problem<ColMatrix> {
    pub fn add_row<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        self.push_row_bounds(bounds)
    }

    /// Add a continuous variable with its coefficients in existing constraints.
    pub fn add_column<N, B, I>(&mut self, cost: f64, bounds: B, row_factors: I) -> Col
    where
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        I: IntoIterator<Item = (Row, f64)>,
    {
        let col = Col(self.num_cols());
        // CSC: record start of this column's entries
        self.matrix
            .astart
            .push(self.matrix.aindex.len().try_into().unwrap());
        for (row, factor) in row_factors {
            self.matrix.aindex.push(row.0);
            self.matrix.avalue.push(factor);
        }
        self.push_column_data(cost, bounds);
        col
    }
}

pub type RowProblem = Problem<RowMatrix>;

impl Problem<RowMatrix> {
    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        cost: f64,
        bounds: B,
    ) -> Col {
        let col = Col(self.num_cols());
        // CSR: columns have no matrix entry at creation time
        self.push_column_data(cost, bounds);
        col
    }

    /// Add a constraint with its variable coefficients.
    ///
    /// Returns the [`Row`] index.
    pub fn add_row<N, B, I>(&mut self, bounds: B, col_factors: I) -> Row
    where
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        I: IntoIterator<Item = (Col, f64)>,
    {
        // CSR: record start of this row's entries
        self.matrix
            .astart
            .push(self.matrix.aindex.len().try_into().unwrap());
        for (col, factor) in col_factors {
            self.matrix
                .aindex
                .push(col.0.try_into().expect("col index too large"));
            self.matrix.avalue.push(factor);
        }
        self.push_row_bounds(bounds)
    }
}
