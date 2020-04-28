#ifndef SWITCH_OP_H
#define SWITCH_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/blob.h"

namespace caffe2 {

template <class Context>
class SwitchOp final : public Operator<Context> {
 public:
  explicit SwitchOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {

    ArgumentHelper helper(operator_def);
    CAFFE_ENFORCE(helper.HasArgument("subnets"), "Must have the subnets argument");
    auto subnets_def = helper.GetRepeatedArgument<NetDef>("subnets");

    const unsigned NUM_SUBNETS = subnets_def.size();
    subnets_.resize(NUM_SUBNETS);
    for(unsigned i = 0; i < NUM_SUBNETS; ++i) {
      subnets_[i] = CreateNet(subnets_def[i], ws);
      CAFFE_ENFORCE(subnets_[i], "Failed to initialize the subnet: %s", subnets_def[i].name());
    }
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto& deadline = Input(0);
    CAFFE_ENFORCE_EQ(deadline.numel(), 1,
       "Invalid deadline tensor in Switch operator: single value expected");
    auto deadlineValue = *deadline.template data<long>();

    const auto& time = Input(1);
    CAFFE_ENFORCE_EQ(time.numel(), 1,
       "Invalid elapsed time tensor in Switch operator: single value expected");
    auto timeValue = *time.template data<long>();

    const auto& threshold = Input(2);
    CAFFE_ENFORCE_EQ(threshold.numel(), subnets_.size()-1,
       "Invalid threshold tensor in Switch operator: the number of threshold values != the number of subnets - 1");
    auto thresholdValue = threshold.template data<long>();

    bool result = true;
    if (subnets_.size() == 1){
      result = subnets_[0]->Run();
    } else {
      for (unsigned i = 0; i < subnets_.size(); i++) {
        if(i == subnets_.size() - 1 || (deadlineValue - timeValue) > thresholdValue[i]) {
          result = subnets_[i]->Run();
          break;
        }
      }
    }

    return result;
  }

  private:
    std::vector<std::unique_ptr<NetBase>> subnets_;
};

} // caffe2

#endif
