use std::convert::TryFrom;
use std::ffi::{c_void, CString};
use std::num::TryFromIntError;
use std::os::raw::c_int;

use highs_sys::*;

use crate::options::HighsOptionValue;
use crate::status::HighsStatus;

pub(crate) fn try_handle_status(status: c_int, msg: &str) -> Result<HighsStatus, HighsStatus> {
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

    /// Suppress all terminal / file output from HiGHS.
    pub(crate) fn make_quiet(&mut self) {
        self.set_option(&b"output_flag"[..], false);
        self.set_option(&b"log_to_console"[..], false);
    }

    /// Set a HiGHS solver option by name.
    pub(crate) fn set_option<S: Into<Vec<u8>>, V: HighsOptionValue>(
        &mut self,
        option: S,
        value: V,
    ) {
        let c_str = CString::new(option).expect("invalid option name");
        let status = unsafe { value.apply_to_highs(self.mut_ptr(), c_str.as_ptr()) };
        try_handle_status(status, "Highs_setOptionValue")
            .expect("An error was encountered in HiGHS.");
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
