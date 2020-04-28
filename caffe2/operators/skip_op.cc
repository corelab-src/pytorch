#include "caffe2/operators/skip_op.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Skip, SkipOp<CPUContext>);

OPERATOR_SCHEMA(Skip)
    .NumInputs(3, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC()DOC")
    .Arg("target_net", "Subnet to skip")
    .Arg("empty_net", "Empty subnet to run when skipping")
    .Input(0, "Deadline", "Relative deadline")
    .Input(1, "Elapsed time", "Elapsed time")
    .Input(2, "Threshold value", "Threshold value")
    .Input(3, "Input Tensors", "Input tensors")
    .Output(0, "Output Tensor", "Output tensor");
} // caffe2
