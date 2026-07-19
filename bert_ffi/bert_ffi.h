// extern-C surface of bert_ffi (fieldrun BERT encoder). See bert_ffi/src/lib.rs.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct BertHandle BertHandle;
BertHandle* be_load(const char* bundle_stem);
int be_dim(const BertHandle*);
int be_encode(const BertHandle*, const uint32_t* ids, int n, float* out, int cap);
void be_free(BertHandle*);
#ifdef __cplusplus
}
#endif
