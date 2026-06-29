// C FFI for the HF tokenizers Rust lib (libtok_ffi.a) — tokenize queries into the model's BPE token-id space.
// Ownership: tk_new returns an OWNING handle; release it with exactly one tk_free (never free it otherwise, never
// double-free). tk_encode/tk_decode BORROW the handle and it stays valid until tk_free. All entry points are
// null-tolerant (null → null/-1). Not thread-safe per handle: don't call concurrently on the same Tokenizer*.
#ifndef TOK_FFI_H
#define TOK_FFI_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Tokenizer Tokenizer;
Tokenizer* tk_new(const char* path);                                  // load tokenizer.json; null on error
int tk_encode(const Tokenizer* tk, const char* text, unsigned* out, int cap);  // text → ids; n written, or -1
int tk_decode(const Tokenizer* tk, unsigned id, char* out, int cap);  // token id -> UTF-8
void tk_free(Tokenizer* tk);
#ifdef __cplusplus
}
#endif
#endif
