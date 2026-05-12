//! Safe Rust bindings to the [HiGHS](https://highs.dev) linear programming solver.
//!
//! ## Quick start
//!
//! ### Row-by-row (declare variables first, then add constraints)
//!
//! ```
//! use highs::{RowProblem, Model, Sense, HighsModelStatus};
//!
//! let mut pb = RowProblem::new();
//! let x = pb.add_column(1., 0..);
//! let y = pb.add_column(2., 0..);
//! pb.add_row(..=6., [(x, 3.), (y, 1.)]);
//! pb.add_row(..=7., [(y, 1.)]);
//!
//! let mut model = Model::new(&pb, Sense::Maximise).unwrap();
//! model.solve().unwrap();
//! assert_eq!(model.status(), HighsModelStatus::Optimal);
//! let sol = model.get_solution();
//! assert_eq!(sol.columns(), &[0., 6.]);
//! ```
//!
//! ### Column-by-column (declare constraints first, then add variables)
//!
//! ```
//! use highs::{ColProblem, Model, Sense, HighsModelStatus};
//!
//! // max: x + 2y + z  s.t. 3x + y <= 6, y + 2z <= 7
//! let mut pb = ColProblem::new();
//! let c1 = pb.add_row(..=6.);
//! let c2 = pb.add_row(..=7.);
//! pb.add_column(1., 0.., [(c1, 3.)]);
//! pb.add_column(2., 0.., [(c1, 1.), (c2, 1.)]);
//! pb.add_column(1., 0.., [(c2, 2.)]);
//!
//! let mut model = Model::new(&pb, Sense::Maximise).unwrap();
//! model.solve().unwrap();
//! assert_eq!(model.status(), HighsModelStatus::Optimal);
//! assert_eq!(model.get_solution().columns(), &[0., 6., 0.5]);
//! ```
//!
//! ### Re-solving / warm-starting
//!
//! Because [`Model::new`] borrows the problem and [`Model::solve`] takes
//! `&mut self`, you can freely re-solve, modify, and re-solve again:
//!
//! ```
//! use highs::{RowProblem, Model, Sense};
//!
//! let mut pb = RowProblem::new();
//! let x = pb.add_column(1., 0..50);
//!
//! let mut model = Model::new(&pb, Sense::Maximise).unwrap();
//! model.solve().unwrap();
//! let obj1 = model.get_objective_value(); // 50
//!
//! // tighten the bound in the live model and re-solve (warm-started)
//! model.change_column_bounds(x, 0..30);
//! model.solve().unwrap();
//! let obj2 = model.get_objective_value(); // 30
//!
//! // problem is still alive — build a new model from it
//! pb.change_column_cost(x, 2.);
//! let mut model2 = Model::new(&pb, Sense::Maximise).unwrap();
//! ```

mod highs_ptr;
mod model;
mod options;
mod problem;
mod status;

// Public API re-exports

pub use model::{Iis, Model, Sense, Solution};
pub use options::HighsOptionValue;
pub use problem::{AsHighsMatrix, Col, ColMatrix, ColProblem, Problem, Row, RowMatrix, RowProblem};
pub use status::{
    HighsIisBoundStatus, HighsIisStatus, HighsModelStatus, HighsSolutionStatus, HighsStatus,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_problem_maximise() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., 0..);
        let y = pb.add_column(2., 0..);
        let z = pb.add_column(1., 0..);
        pb.add_row(..=6., [(x, 3.), (y, 1.)]);
        pb.add_row(..=7., [(y, 1.), (z, 2.)]);

        let mut model = Model::new(&pb, Sense::Maximise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        let sol = model.get_solution();
        assert_eq!(sol.columns(), &[0., 6., 0.5]);
        assert_eq!(sol.rows(), &[6., 7.]);
    }

    #[test]
    fn col_problem_maximise() {
        let mut pb = ColProblem::new();
        let c1 = pb.add_row(..6.);
        let c2 = pb.add_row(..7.);
        pb.add_column(1., 0.., [(c1, 3.)]);
        pb.add_column(2., 0.., [(c1, 1.), (c2, 1.)]);
        pb.add_column(1., 0.., [(c2, 2.)]);

        let mut model = Model::new(&pb, Sense::Maximise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        let sol = model.get_solution();
        assert_eq!(sol.columns(), &[0., 6., 0.5]);
        assert_eq!(sol.rows(), &[6., 7.]);
    }

    #[test]
    fn problem_stays_alive_after_model_build() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., 1..);
        let mut model = Model::new(&pb, Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 1.0);

        // Modify the problem and create a new model — no re-allocation of data
        pb.change_column_cost(x, 2.);
        let mut model2 = Model::new(&pb, Sense::Minimise).unwrap();
        model2.solve().unwrap();
        assert_eq!(model2.get_objective_value(), 2.0);
    }

    #[test]
    fn change_column_cost_on_model() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., 1..);
        let mut model = Model::new(&pb, Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 1.0);

        model.change_column_cost(x, 2.);
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 2.0);
    }

    #[test]
    fn change_column_bounds_on_model() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., 0..);
        let mut model = Model::new(&pb, Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 0.0);

        model.change_column_bounds(x, 1..);
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 1.0);
    }

    #[test]
    fn add_row_and_col_to_live_model() {
        let mut model = Model::new(&ColProblem::default(), Sense::Minimise).unwrap();
        let col = model.add_col(1., 1.., vec![]).unwrap();
        model.add_row(..1., vec![(col, 1.)]).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        assert_eq!(model.get_solution().columns(), &[1.]);
    }

    #[test]
    fn incremental_infeasible() {
        let mut model = Model::new(&ColProblem::default(), Sense::Minimise).unwrap();
        let col = model.add_col(1., 1.., vec![]).unwrap();
        model.add_row(..0.5, vec![(col, 1.)]).unwrap(); // col >= 1 but row <= 0.5
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Infeasible);
    }

    #[test]
    fn iis_simple() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(0., 1..); // x >= 1
        let y = pb.add_column(0., 1..); // y >= 1
        let z = pb.add_column(0., 1..); // z >= 1
        pb.add_row(..=1.1, [(x, 1.), (z, 1.)]); // x + z <= 1.1  (infeasible with x,z>=1)
        pb.add_row(5.., [(y, 1.)]); // y >= 5  (independent)

        let mut model = Model::new(&pb, Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Infeasible);
        let iis = model.get_iis();

        assert_eq!(
            iis.iis_cols,
            vec![
                (Col(0), HighsIisBoundStatus::Lower),
                (Col(2), HighsIisBoundStatus::Lower),
            ]
        );
        assert_eq!(iis.iis_rows, vec![(Row(0), HighsIisBoundStatus::Upper)]);
        assert_eq!(
            iis.model_cols_iis_status,
            vec![
                HighsIisStatus::InConflict,
                HighsIisStatus::NotInConflict,
                HighsIisStatus::InConflict,
            ]
        );
        assert_eq!(
            iis.model_rows_iis_status,
            vec![HighsIisStatus::InConflict, HighsIisStatus::NotInConflict,]
        );
    }

    #[test]
    fn objective_value() {
        let mut pb = RowProblem::new();
        pb.add_column(1., 0..50);
        let mut model = Model::new(&pb, Sense::Maximise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        assert_eq!(model.get_objective_value(), 50.0);
    }

    #[test]
    fn objective_value_empty_model() {
        let mut model = Model::new(&RowProblem::default(), Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.get_objective_value(), 0.0);
    }

    #[test]
    fn num_cols_and_rows() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., -1..);
        let y = pb.add_column(1., 0..);
        pb.add_row(..1., [(x, 1.), (y, 1.)]);
        let model = Model::new(&pb, Sense::Minimise).unwrap();
        assert_eq!(model.num_cols(), 2);
        assert_eq!(model.num_rows(), 1);
    }

    fn test_coefs(coefs: [f64; 2]) {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., -1..);
        let y = pb.add_column(1., 0..);
        pb.add_row(..1., [(x, coefs[0]), (y, coefs[1])]);
        let mut model = Model::new(&pb, Sense::Minimise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.get_solution().columns(), &[-1., 0.]);
    }

    #[test]
    fn test_single_zero_coef() {
        test_coefs([1.0, 0.0]);
        test_coefs([0.0, 1.0]);
    }

    #[test]
    fn test_all_zero_coefs() {
        test_coefs([0.0, 0.0]);
    }

    #[test]
    fn test_no_zero_coefs() {
        test_coefs([1.0, 1.0]);
    }

    #[test]
    fn clone_preserves_lp_and_basis() {
        let mut pb = RowProblem::new();
        let x = pb.add_column(1., 0..);
        let y = pb.add_column(2., 0..);
        pb.add_row(..=6., [(x, 3.), (y, 1.)]);

        let mut model = Model::new(&pb, Sense::Maximise).unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        assert_eq!(model.get_solution().columns(), &[0., 6.]);

        // clone should have the same LP data and basis
        let mut clone = model.clone();
        assert_eq!(clone.num_cols(), model.num_cols());
        assert_eq!(clone.num_rows(), model.num_rows());

        // re-solving the clone from the preserved basis should yield the same result
        clone.solve().unwrap();
        assert_eq!(clone.status(), HighsModelStatus::Optimal);
        assert_eq!(clone.get_solution().columns(), &[0., 6.]);

        // modifying the clone must not affect the original
        clone.change_column_bounds(x, 1..);
        clone.solve().unwrap();
        assert_eq!(clone.get_objective_value(), 7.0); // max x+2y s.t. 3x+y<=6, x>=1 => x=1,y=3 => obj=7
        model.solve().unwrap();
        assert_eq!(model.get_solution().columns(), &[0., 6.]); // original unchanged
    }

    #[test]
    fn clone_before_solve_has_no_basis() {
        let mut pb = RowProblem::new();
        pb.add_column(1., 0..10);
        let model = Model::new(&pb, Sense::Maximise).unwrap();
        // clone before any solve — should still be valid and solvable
        let mut clone = model.clone();
        clone.solve().unwrap();
        assert_eq!(clone.status(), HighsModelStatus::Optimal);
        assert_eq!(clone.get_solution().columns(), &[10.]);
    }

    #[test]
    fn set_solution_hint() {
        let mut pb = RowProblem::new();
        pb.add_column(1., 0..50);
        let mut model = Model::new(&pb, Sense::Maximise).unwrap();
        model.set_option("time_limit", 0_i32);
        model
            .set_solution(Some(&[50.0]), Some(&[]), Some(&[1.0]), Some(&[]))
            .unwrap();
        model.solve().unwrap();
        assert_eq!(model.status(), HighsModelStatus::Optimal);
        assert_eq!(model.get_solution().columns(), &[50.0]);
    }
}
