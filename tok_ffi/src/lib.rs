//! Minimal C FFI over the HF `tokenizers` Rust lib, for sgiandubh's thin runtime to tokenize queries into the model's
//! BPE token-id space (the rosetta cover lives in that space). Exposes: load a bundle.tokenizer.json, encode text → ids.
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
