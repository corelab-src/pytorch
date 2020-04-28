#include "caffe2/operators/switch_op.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Switch, SwitchOp<CPUContext>);

OPERATOR_SCHEMA(Switch)
    .NumInputs(3, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC()DOC")
    .Arg("subnets", "Subnets to switch")
    .Input(0, "Deadline", "Relative deadline")
    .Input(1, "Elapsed time", "Elapsed time")
    .Input(2, "Threshold values", "Threshold values")
    .Input(3, "Input Tensors", "Input tensors")
    .Output(0, "Output Tensor", "Output tensor");
} // caffe2
