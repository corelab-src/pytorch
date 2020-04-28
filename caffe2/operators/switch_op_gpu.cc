#include "caffe2/operators/switch_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Switch, SwitchOp<CUDAContext>);

} // namespace caffe2
