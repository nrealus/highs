use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

use highs_sys::HighsInt;

pub trait HighsOptionValue {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt;
}

impl HighsOptionValue for bool {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        highs_sys::Highs_setBoolOptionValue(highs, option, if self { 1 } else { 0 })
    }
}

impl HighsOptionValue for i32 {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        highs_sys::Highs_setIntOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for f64 {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        highs_sys::Highs_setDoubleOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for &CStr {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        highs_sys::Highs_setStringOptionValue(highs, option, self.as_ptr())
    }
}

impl HighsOptionValue for &[u8] {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        CString::new(self)
            .expect("invalid highs option value")
            .apply_to_highs(highs, option)
    }
}

impl HighsOptionValue for &str {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> HighsInt {
        self.as_bytes().apply_to_highs(highs, option)
    }
}
