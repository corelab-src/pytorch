#include "caffe2/operators/skip_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Skip, SkipOp<CUDAContext>);

} // namespace caffe2
