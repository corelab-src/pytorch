#ifndef SKIP_OP_H
#define SKIP_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/blob.h"

namespace caffe2 {

template <class Context>
class SkipOp final : public Operator<Context> {
 public:
  explicit SkipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {

    ArgumentHelper helper(operator_def);

    CAFFE_ENFORCE(helper.HasArgument("target_net"), "Must have the target_net argument");
    auto target_net_def = helper.GetSingleArgument<NetDef>("target_net", NetDef());

    target_net_ = CreateNet(target_net_def, ws);
    if(!target_net_) {
      CAFFE_ENFORCE(target_net_, "Failed to initialize the target net: %s", target_net_def.name());
    }

    CAFFE_ENFORCE(helper.HasArgument("empty_net"), "Must have the empty_net argument");
    auto empty_net_def = helper.GetSingleArgument<NetDef>("empty_net", NetDef());

    empty_net_ = CreateNet(empty_net_def, ws);
    if(!empty_net_) {
      CAFFE_ENFORCE(empty_net_, "Failed to initialize the empty net: %s", empty_net_def.name());
    }

  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto& deadline = Input(0);
    CAFFE_ENFORCE_EQ(deadline.numel(), 1,
       "Invalid deadline tensor in Skip operator: single value expected");
    auto deadlineValue = *deadline.template data<long>();

    const auto& time = Input(1);
    CAFFE_ENFORCE_EQ(time.numel(), 1,
       "Invalid elapsed time tensor in Skip operator: single value expected");
    auto timeValue = *time.template data<long>();

    const auto& threshold = Input(2);
    CAFFE_ENFORCE_EQ(threshold.numel(), 1,
       "Invalid threshold tensor in Skip operator: single value expected");
    auto thresholdValue = *threshold.template data<long>();

    bool result = true;
    if((deadlineValue - timeValue) > thresholdValue) {
        result = target_net_->Run();
    } else {
        result = empty_net_->Run();
    }

    return result;
  }

  private:
    std::unique_ptr<NetBase> target_net_;
    std::unique_ptr<NetBase> empty_net_;
};

} // caffe2

#endif
