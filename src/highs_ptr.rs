use std::convert::TryFrom;
use std::ffi::c_void;
use std::num::TryFromIntError;
use std::os::raw::c_int;

use highs_sys::*;

use crate::status::HighsStatus;

pub(crate) fn try_handle_status(
    status: c_int,
    msg: &str,
) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status)
        .expect("HiGHS returned an unexpected status value. Please report it as a bug.");
    match status_enum {
        s @ HighsStatus::OK => Ok(s),
        s @ HighsStatus::Warning => {
            log::warn!("HiGHS emitted a warning: {msg}");
            Ok(s)
        }
        err => Err(err),
    }
}

macro_rules! highs_call {
    ($function_name:ident ($($param:expr),+)) => {
        $crate::highs_ptr::try_handle_status(
            $function_name($($param),+),
            stringify!($function_name),
        )
    };
}

pub(crate) use highs_call;

#[derive(Debug)]
pub(crate) struct HighsPtr(*mut c_void);

impl Default for HighsPtr {
    fn default() -> Self {
        Self(unsafe { Highs_create() })
    }
}

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl HighsPtr {
    /// Immutable raw pointer (for C API functions that take `const void*`).
    #[allow(dead_code)]
    pub(crate) const fn ptr(&self) -> *const c_void {
        self.0
    }

    /// Mutable raw pointer (requires `&mut self`).
    pub(crate) fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }

    /// Cast to `*mut c_void` from a shared reference.
    ///
    /// Needed because the HiGHS C API takes `*mut` even for logically
    /// read-only calls — see <https://github.com/ERGO-Code/HiGHS/issues/479>.
    /// Callers must ensure no aliased mutation occurs.
    pub(crate) unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    /// Number of columns currently in the model.
    pub(crate) fn num_cols(&self) -> Result<usize, TryFromIntError> {
        unsafe { Highs_getNumCols(self.0) }.try_into()
    }

    /// Number of rows currently in the model.
    pub(crate) fn num_rows(&self) -> Result<usize, TryFromIntError> {
        unsafe { Highs_getNumRows(self.0) }.try_into()
    }
}
