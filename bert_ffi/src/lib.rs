//! extern-C surface over fieldrun's encoder-only BERT: load a fieldrun bundle, encode token ids,
//! return the LAST FOUR hidden-state snapshots (what supar's scalar mix consumes), flat f32.
//! Mirrors tok_ffi's contract style: null-tolerant, owning handles, caller serializes.
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use fieldrun::{Bert, Bundle};

pub struct BertHandle {
    bert: Bert,
    d: usize,
}

/// Load a fieldrun bert bundle by stem (path without .fieldrun.json). Null on error.
#[no_mangle]
pub extern "C" fn be_load(stem: *const c_char) -> *mut BertHandle {
    if stem.is_null() {
        return std::ptr::null_mut();
    }
    let stem = match unsafe { CStr::from_ptr(stem) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };
    match std::panic::catch_unwind(|| {
        let b = Bundle::load(&stem).ok()?;
        let d = b.config[2] as usize;
        Some(BertHandle { bert: Bert::new(b), d })
    }) {
        Ok(Some(h)) => Box::into_raw(Box::new(h)),
        _ => std::ptr::null_mut(),
    }
}

/// Hidden size (768 for gbert-base); -1 on null.
#[no_mangle]
pub extern "C" fn be_dim(h: *const BertHandle) -> c_int {
    if h.is_null() { -1 } else { unsafe { (*h).d as c_int } }
}

/// Encode `n` token ids; write the last 4 snapshots (each seq*d f32, oldest layer first) into `out`
/// (capacity `cap` floats). Returns floats written, or -1 on error / insufficient capacity.
#[no_mangle]
pub extern "C" fn be_encode(h: *const BertHandle, ids: *const u32, n: c_int, out: *mut f32, cap: c_int) -> c_int {
    if h.is_null() || ids.is_null() || out.is_null() || n <= 0 {
        return -1;
    }
    let hh = unsafe { &*h };
    let ids: Vec<i64> = unsafe { std::slice::from_raw_parts(ids, n as usize) }.iter().map(|&x| x as i64).collect();
    let need = 4 * ids.len() * hh.d;
    if (cap as usize) < need {
        return -1;
    }
    let snaps = match std::panic::catch_unwind(|| hh.bert.hiddens(&ids)) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    if snaps.len() < 4 {
        return -1;
    }
    let outs = unsafe { std::slice::from_raw_parts_mut(out, need) };
    let mut o = 0usize;
    for sn in &snaps[snaps.len() - 4..] {
        outs[o..o + sn.len()].copy_from_slice(sn);
        o += sn.len();
    }
    o as c_int
}

#[no_mangle]
pub extern "C" fn be_free(h: *mut BertHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn null_safety() {
        assert!(be_load(std::ptr::null()).is_null());
        assert_eq!(be_dim(std::ptr::null()), -1);
        assert_eq!(be_encode(std::ptr::null(), std::ptr::null(), 0, std::ptr::null_mut(), 0), -1);
        be_free(std::ptr::null_mut());
    }
}
