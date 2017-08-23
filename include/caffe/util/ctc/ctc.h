/** \file ctc.h
 * Contains a simple C interface to call fast CPU and GPU based computation
 * of the CTC loss.
 */

#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

//forward declare of CUDA typedef to avoid needing to pull in CUDA headers
typedef struct CUstream_st* CUstream;

typedef enum {
    CTC_STATUS_SUCCESS = 0,
    CTC_STATUS_MEMOPS_FAILED = 1,
    CTC_STATUS_INVALID_VALUE = 2,
    CTC_STATUS_EXECUTION_FAILED = 3,
    CTC_STATUS_UNKNOWN_ERROR = 4
} ctcStatus_t;

/** Returns a string containing a description of status that was passed in
 *  \param[in] status identifies which string should be returned
 *  \return C style string containing the text description
 *  */
const char* ctcGetStatusString(ctcStatus_t status);

typedef enum {
    CTC_CPU = 0,
    CTC_GPU = 1
} ctcComputeLocation;

/** Structure used to indicate where the ctc calculation should take place
 *  and parameters associated with that place.
 *  Cpu execution can specify the maximum number of threads that can be used
 *  Gpu execution can specify which stream the kernels should be launched in.
 *  */
struct ctcComputeInfo {
    ctcComputeLocation loc;
    union {
        unsigned int num_threads;
        CUstream stream;
    };
};


#ifdef __cplusplus
}
#endif
