## @package control_ops
# Module caffe2.python.helpers.control_ops
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.control_ops_util import add_if_op, add_while_op
from caffe2.python.control_ops_util import add_skip_op, add_switch_op

def cond(model, cond_blob, external_blobs, then_model, else_model=None):
    """Condition"""
    add_if_op(
        model.net,
        cond_blob,
        external_blobs,
        then_model.net,
        else_model.net if else_model else None)


def loop(model, cond_blob, external_blobs, loop_model, cond_model=None):
    """Loop"""
    add_while_op(
        model.net,
        cond_blob,
        external_blobs,
        loop_model.net,
        cond_model.net if cond_model else None)

def skip(model, input_blobs, external_blobs, submodel,
    org_output_blobs, target_output_blobs, empty_output_blobs):
    """Skip"""
    return add_skip_op(
        model.net,
        input_blobs,
        external_blobs,
        submodel.net,
        org_output_blobs,
        target_output_blobs,
        empty_output_blobs
    )

def switch(model, input_blobs, external_blobs, submodels, blob_output_map):
    """Switch"""
    return add_switch_op(
        model.net,
        input_blobs,
        external_blobs,
        [ submodel.net for submodel in submodels ],
        blob_output_map
    )
