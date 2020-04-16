import torch.nn as nn
from torch.quantization import (
    DEFAULT_DYNAMIC_MODULE_MAPPING,
    DEFAULT_MODULE_MAPPING,
    DEFAULT_QAT_MODULE_MAPPING,
    default_eval_fn,
    quantize,
    DeQuantStub,
)
from torch.quantization._numeric_suite import (
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
)
from torch.testing._internal.common_quantization import (
    AnnotatedConvModel,
    QuantizationTestCase,
)


_EXCLUDE_QCONFIG_PROPAGATE_LIST = {DeQuantStub}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {nn.Sequential}


class EagerModeNumericSuiteTest(QuantizationTestCase):
    def test_compare_weights(self):
        r"""Compare the weights of float and quantized conv layer
        """
        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        weight_dict = compare_weights(
            annotated_conv_model.state_dict(),
            quantized_annotated_conv_model.state_dict(),
        )
        self.assertEqual(len(weight_dict), 1)
        for k, v in weight_dict.items():
            self.assertTrue(v["float"].shape == v["quantized"].shape)

    def test_compare_model_stub(self):
        r"""Compare the output of quantized conv layer and its float shadow module
        """
        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        data = self.img_data[0][0]
        module_swap_list = [nn.Conv2d]
        ob_dict = compare_model_stub(
            annotated_conv_model, quantized_annotated_conv_model, module_swap_list, data
        )
        self.assertEqual(len(ob_dict), 1)
        for k, v in ob_dict.items():
            self.assertTrue(v["float"].shape == v["quantized"].shape)

    def test_compare_model_outputs(self):
        r"""Compare the output of conv layer in quantized model and corresponding
        output of conv layer in float model
        """
        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        data = self.img_data[0][0]
        white_list = (
            set(DEFAULT_MODULE_MAPPING.values())
            | set(DEFAULT_QAT_MODULE_MAPPING.values())
            | set(DEFAULT_DYNAMIC_MODULE_MAPPING.values())
            | set(DEFAULT_MODULE_MAPPING.keys())
            | set(DEFAULT_QAT_MODULE_MAPPING.keys())
            | set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys())
            | _INCLUDE_QCONFIG_PROPAGATE_LIST
        ) - _EXCLUDE_QCONFIG_PROPAGATE_LIST
        act_compare_dict = compare_model_outputs(
            annotated_conv_model, quantized_annotated_conv_model, data, white_list
        )
        self.assertEqual(len(act_compare_dict), 2)
        for k, v in act_compare_dict.items():
            self.assertTrue(v["float"].shape == v["quantized"].shape)
