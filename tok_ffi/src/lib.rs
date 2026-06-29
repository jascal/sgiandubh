//! Minimal C FFI over the HF `tokenizers` Rust lib, for sgiandubh's thin runtime to tokenize queries into the model's
//! BPE token-id space (the rosetta cover lives in that space). Exposes: load a bundle.tokenizer.json, encode text → ids.
//!
//! Ownership / lifetime contract (the C caller owns the handle):
//!   * `tk_new` returns an OWNING handle Rust allocated via `Box::into_raw`. Release it with EXACTLY ONE `tk_free`;
//!     never free it any other way, and never double-free.
//!   * `tk_encode` / `tk_decode` BORROW the handle (do not consume it) — it stays valid across calls until `tk_free`.
//!   * All four entry points are null-tolerant: a null/invalid handle yields a null return or -1, never a deref.
//!   * A handle is NOT internally synchronized — do not call into the same handle concurrently from multiple threads.
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use tokenizers::Tokenizer;

/// Load a tokenizer.json. Returns an opaque handle, or null on error.
#[no_mangle]
pub extern "C" fn tk_new(path: *const c_char) -> *mut Tokenizer {
    if path.is_null() { return std::ptr::null_mut(); }
    let p = match unsafe { CStr::from_ptr(path) }.to_str() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    match Tokenizer::from_file(p) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Encode `text` (no special tokens) into `out` (capacity `cap`). Returns the number of ids written, or -1 on error.
#[no_mangle]
pub extern "C" fn tk_encode(tk: *const Tokenizer, text: *const c_char, out: *mut u32, cap: c_int) -> c_int {
    if tk.is_null() || text.is_null() || out.is_null() || cap <= 0 { return -1; }
    let t = unsafe { &*tk };
    let s = match unsafe { CStr::from_ptr(text) }.to_str() { Ok(s) => s, Err(_) => return -1 };
    match t.encode(s, false) {
        Ok(enc) => {
            let ids = enc.get_ids();
            let n = ids.len().min(cap as usize);
            unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr(), out, n); }
            n as c_int
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn tk_free(tk: *mut Tokenizer) {
    if !tk.is_null() { unsafe { drop(Box::from_raw(tk)); } }
}

/// Decode a single token id to UTF-8 into `out` (capacity `cap`, null-terminated). Returns bytes written, or -1.
#[no_mangle]
pub extern "C" fn tk_decode(tk: *const Tokenizer, id: u32, out: *mut c_char, cap: c_int) -> c_int {
    if tk.is_null() || out.is_null() || cap <= 1 { return -1; }
    let t = unsafe { &*tk };
    let s = t.decode(&[id], false).unwrap_or_default();
    let b = s.as_bytes();
    let n = b.len().min((cap - 1) as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const c_char, out, n);
        *out.add(n) = 0;
    }
    n as c_int
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    // The null/invalid-input contract: every entry point must reject, never dereference. (Round-trip fidelity is
    // covered end-to-end against a real tokenizer.json; a committed fixture isn't worth carrying here.)
    #[test]
    fn null_inputs_are_rejected_not_dereferenced() {
        assert!(tk_new(ptr::null()).is_null());
        let txt = CString::new("hi").unwrap();
        let mut out = [0u32; 8];
        assert_eq!(tk_encode(ptr::null(), txt.as_ptr(), out.as_mut_ptr(), 8), -1);
        let mut buf = [0 as c_char; 8];
        assert_eq!(tk_decode(ptr::null(), 0, buf.as_mut_ptr(), 8), -1);
        tk_free(ptr::null_mut()); // must be a safe no-op
    }
}
