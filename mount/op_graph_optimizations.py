# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import json
import numpy
from operator import mul
from functools import reduce


from qti.aisw.converters.common.converter_ir import translation, op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import code_to_message, translation_utils


# ------------------------------
#   Module Level enum/Functions
# ------------------------------
REMOVE_NOOP = "REMOVE_NOOP"
REMOVE_DISCONNECTED = "REMOVE_DISCONNECTED"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
MATCH_HARDSWISH = "MATCH_HARDSWISH"
MATCH_CAFFE_SSD_TO_TF = "MATCH_CAFFE_SSD_TO_TF"
MATCH_SPACETODEPTH = "MATCH_SPACETODEPTH"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
SQUASH_BOX_DECODER = "SQUASH_BOX_DECODER"
SQUASH_SUM = "SQUASH_SUM"
SQUASH_PROD = "SQUASH_PROD"
SQUASH_DIV = "SQUASH_DIV"
SQUASH_SUB = "SQUASH_SUB"
FOLD_CONCATS = "FOLD_CONCATS"
BROADCAST_CONST = "BROADCAST_CONST"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
ADD_QPARAMS = "ADD_QPARAMS"
CHAIN_ELTWISE_OPS = "CHAIN_ELTWISE_OPS"
ADJUST_NMS_FEATURE_DIMS = "ADJUST_NMS_FEATURE_DIMS"
EXTRACT_COLOR_TRANSFROM = "EXTRACT_COLOR_TRANSFROM"
OPTIMIZE_NEG = "OPTIMIZE_NEG"
PREPROCESS_ROI_POOL_INPUTS = "PREPROCESS_ROI_POOL_INPUTS"

supported_opt_list = [SQUASH_SCALE, SQUASH_PROD, SQUASH_DIV, SQUASH_SUM, SQUASH_SUB, SQUASH_BATCHNORM, FOLD_CONCATS,
                      MATCH_CHANNELSHUFFLE, MATCH_HARDSWISH, AXES_TO_SPATIAL_FIRST_ORDER, REMOVE_NOOP, ADD_QPARAMS,
                      CHAIN_ELTWISE_OPS, ADJUST_NMS_FEATURE_DIMS, EXTRACT_COLOR_TRANSFROM, OPTIMIZE_NEG, MATCH_SPACETODEPTH,
                      PREPROCESS_ROI_POOL_INPUTS]

format_to_permute_order = {'NSC': AxisTracker.AxisFormat.NSC_TO_NCS,
                           'BTF': AxisTracker.AxisFormat.BTF_TO_TBF}
format_to_format = {'NSC': AxisTracker.AxisFormat.NCS, 'BTF': AxisTracker.AxisFormat.TBF}
OptimizationTranslations = translation.TranslationBank()


class IROptimizations(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(IROptimizations.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument("--dumpIR", action="store_true",
                                       help="This will dump the IR as JSON before and after the optimizations",
                                       default=False)
            self.add_optional_argument("--disable_batchnorm_folding",
                                       default=False,
                                       action="store_true")
            self.add_optional_argument("--squash_box_decoder",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--match_caffe_ssd_to_tf",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--disable_chaining_eltwise_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--adjust_nms_features_dims",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--extract_color_transform",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_roi_pool_inputs",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_axes_to_spatial_first_order",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")

            q_group = self.add_argument_group('Quantizer Options')
            q_group.add_argument('--quantization_overrides', type=str, default="",
                                 help='Use this option to specify a json file with parameters to use for '
                                      'quantization. These will override any quantization data carried from conversion '
                                      '(eg TF fake quantization) or calculated during the normal quantization process. '
                                      'Format defined as per AIMET specification.')

    def __init__(self, args):
        self.dump_ir_graph = args.dumpIR
        self.enable_batchnorm_folding = not args.disable_batchnorm_folding
        self.squash_box_decoder = args.squash_box_decoder
        self.match_caffe_ssd_to_tf = args.match_caffe_ssd_to_tf
        self.chain_eltwise_ops = not args.disable_chaining_eltwise_ops
        self.adjust_nms_features_dims = args.adjust_nms_features_dims
        self.extract_color_transform = args.extract_color_transform
        self.perform_axes_to_spatial_first_order = args.perform_axes_to_spatial_first_order
        self.preprocess_roi_pool_inputs = args.preprocess_roi_pool_inputs
        self.quantization_overrides = args.quantization_overrides

    def optimize(self, graph):
        # apply graph transformations
        log_debug2("Applying graph Optimizations...")

        # Dump the IR for debug before or after an optimization using graph.dump_json(<filename>)
        if self.dump_ir_graph:
            log_info("Dumping IR graph before all optimizations as IRGraph_before_optimizations.json")
            graph.dump_json("IRGraph_before_optimizations.json")

        if self.quantization_overrides:
            self.populate_quantization_params(self.quantization_overrides, graph)

        OptimizationTranslations.apply_method_to_graph(FOLD_CONCATS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_HARDSWISH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_SPACETODEPTH, graph, fail_if_no_method=False)

        # Element-wise squashing optimizations. This shall be done after matching larger sequences as they single-op
        # squashing into previous layer
        OptimizationTranslations.apply_method_to_graph(SQUASH_SCALE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PROD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_DIV, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUB, graph, fail_if_no_method=False)

        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)
        if self.squash_box_decoder:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BOX_DECODER, graph, fail_if_no_method=False)
        if self.match_caffe_ssd_to_tf:
            OptimizationTranslations.apply_method_to_graph(MATCH_CAFFE_SSD_TO_TF, graph, fail_if_no_method=False)
        if self.adjust_nms_features_dims:
            OptimizationTranslations.apply_method_to_graph(ADJUST_NMS_FEATURE_DIMS, graph, fail_if_no_method=False)
        if self.extract_color_transform:
            OptimizationTranslations.apply_method_to_graph(EXTRACT_COLOR_TRANSFROM, graph, fail_if_no_method=False)

        # ------------------------------------------------------------------------------
        #   PRE-PROCESSING
        # TODO: Move once optimizations are split into backend specific sections
        # ------------------------------------------------------------------------------
        # pre-process roi inputs
        if self.preprocess_roi_pool_inputs:
            OptimizationTranslations.apply_method_to_graph(PREPROCESS_ROI_POOL_INPUTS, graph, fail_if_no_method=False)

        # transition to NSC
        if self.perform_axes_to_spatial_first_order:
            OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

        # Apply Broadcasting after all Folding is Done and Axis Transformation is done
        OptimizationTranslations.apply_method_to_graph(BROADCAST_CONST, graph, fail_if_no_method=False)

        # remove layers disconnected from the main graph
        if graph.output_nodes:
            remove_disconnected_layers(graph)

        # Performs an expansion on eltwise ops with > 2 inputs which should occur after all optimizations are attempted
        if self.chain_eltwise_ops:
            OptimizationTranslations.apply_method_to_graph(CHAIN_ELTWISE_OPS, graph, fail_if_no_method=False)

        # Optimize negations which typically apply to binary eltwise operations, hence adding after the optional
        # chaining step.
        OptimizationTranslations.apply_method_to_graph(OPTIMIZE_NEG, graph, fail_if_no_method=False)

        # remove NOOPs, which may include trivial permutes at this point
        # This may happen because some ops result in constant attributes that are absorbed by the layers
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_NOOP, graph, fail_if_no_method=False)

        # add op-specific quantization encodings to QParams Record.
        OptimizationTranslations.apply_method_to_all_ops(ADD_QPARAMS, graph, fail_if_no_method=False)

        if self.dump_ir_graph:
            log_info("Dumping IR graph after all optimizations as IRGraph_after_optimizations.json")
            graph.dump_json("IRGraph_after_optimizations.json")

        return graph

    def populate_quantization_params(self, encoding_file, ir_graph):

        def _extract_encoding_dict(name, enc):
            # Offsets are sometimes stored as positive values, but converters require negative
            # Everything is optional except bw. Default to 0s if not provided.
            return {"name": name,
                    "min": float(enc["min"] if 'min' in enc else 0.0),
                    "max": float(enc["max"] if 'max' in enc else 0.0),
                    "bw": int(enc['bitwidth']),
                    "offset": int(-abs(enc['offset']) if 'offset' in enc else 0),
                    "scale": float(enc['scale'] if 'scale' in enc else 0.0),
                    "overridden": True }

        if not encoding_file:
            return

        log_info('Processing user provided quantization encodings: ', encoding_file)
        f = open(encoding_file)
        q = json.load(f)

        acts = q['activation_encodings']
        params = q['param_encodings']
        encoding_count = 0

        # Graph inputs are special cases because they aren't owned by a node until IR conversion
        inputs = ir_graph.get_input_nodes_to_graph()
        for i in inputs:
            n = i.op.name
            if n in acts:
                encoding_count += 1
                ir_graph.add_quantization_params(n, output_encodings=[_extract_encoding_dict(n, acts[n][0])])

        # Walk through the original source framework op->input mapping to find the weights
        for op_name, op in ir_graph.src_graph_op_info.items():
            param_encs = []

            inputs = op['inputs']
            if inputs:
                for idx, i in enumerate(inputs):
                    if i in params:
                        encoding_count += 1
                        param_encs.append(_extract_encoding_dict('weights', params[i][0]))

                ir_graph.add_quantization_params(op_name, param_encodings=param_encs)

        # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
        # may have moved the activation names to new ops.
        for act in acts:
            act_encs = []
            if ir_graph.has_buffer(act):
                op = ir_graph.get_producer_op(act)
                encoding_count += 1
                act_encs.append(_extract_encoding_dict(act, acts[act][0]))
                ir_graph.add_quantization_params(op.name, output_encodings=act_encs)

        log_info('Processed '+str(encoding_count)+' quantization encodings')


class OptimizationTranslationBase(translation.Translation):
    """
    This class is to be used to perform graph optimizations such as: folding, squashing,pruning, etc. Additionally,
    it is also used to perform axis tracking and by default implements to spatial first order function
    (NCHW to NHWC, or TBF to BTF). Use this base class to get the default function and call register_method to add a new
    optimization. For eg: The OptimizeBatchnormTranslation overloads the axes_to_spatial_first_order to handle weights
    as well as adds a squash_batchnorm function and registers the method in the __init__ function.
    """
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)

    def axes_to_spatial_first_order(self, node, graph):
        """
        Performs axis permutations(as needed) to get a spatial first order.

        Note: The eltwise_...() function that gets called re-populates the node's buffer "axis_format" and "shape" from
        source framework to the destination for certain ranks. If an overload of this function is done for a child class
        and this eltwise_...() function is not called make sure to understand and implement these changes to avoid
        conversion errors.

        :param node: an OpNode object to optimize from the IR graph
        :param graph: an IROpgraph object

        """
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


# ------------------------------------------------------------------------------------------------------------------
#   Graph Optimizations
# ------------------------------------------------------------------------------------------------------------------
def register_graph_optimization(graph_optimization_method):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param graph: a concrete class for a given optimization
    """
    return graph_optimization_method


@register_graph_optimization
def remove_disconnected_layers(graph):
    all_ops = set(graph.nodes_in_order)
    connected_ops = set()
    queue = []
    graph_output_nodes = graph.get_output_nodes_of_graph()

    if graph.output_nodes and not graph_output_nodes:
        log_warning("The output node names {} does not exist in the graph.".format(graph.output_nodes))

    if graph_output_nodes:
        queue.extend(graph_output_nodes)
        # Find nodes from Output to Input Op
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node
            node_inputs = graph.get_op_input_nodes(node)
            new_nodes = [node_ for node_ in node_inputs if (node_ not in connected_ops and node_ not in queue)]
            queue.extend(new_nodes)

    else:
        queue.extend(graph.get_input_nodes_to_graph())
        # Find nodes from Input Op to outputs
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node, this will add the Constant input Ops that will be otherwise missed
            node_inputs = graph.get_op_input_nodes(node)
            new_nodes = [node for node in node_inputs if node not in connected_ops]
            for new_node in new_nodes:
                queue.insert(0, new_node)

            # Extend the queue with output nodes
            node_outputs = graph.get_op_output_nodes(node)
            new_nodes = [node for node in node_outputs if node not in queue]
            queue.extend(new_nodes)

    disconnected_nodes = all_ops - connected_ops
    prunable_node_names = [node.op.name for node in disconnected_nodes]
    if disconnected_nodes:
        log_debug("Pruning Disconnected nodes {}".format(prunable_node_names))

    for node in disconnected_nodes:
        try:
            graph.prune(node, force_remove=True)
        except Exception as e:
            log_error("Cannot find node {}".format(node.op.name))
            raise e
    return graph


# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------
def register_layer_optimization(layer_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param optimization_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(layer_translation(), layer_translation().op_type)
    return layer_translation


@register_layer_optimization
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY
        self.register_method(EXTRACT_COLOR_TRANSFROM, self.extract_color_transform)

    @staticmethod
    def extract_color_transform(graph):
        """ Optional Optimization to create separate Op to handle color transformation pre-processing for network
            inputs
        """
        def validate_transformation(nodes_tuple):
            node_ = nodes_tuple[0]
            if node_.op.input_encoding_in != node_.op.input_encoding_out and \
                    node_.op.input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:
                return True
            return False

        sequence = [("input", (), ())]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_transformation)

        for node_tuple in matched_node_list:
            input_node = node_tuple[0]
            # adjust shape for input as that will be the expected shape after transformation
            color_transform_name = input_node.output_names[0] + "_post_transform"
            color_transform_output_shape = input_node.op.shape
            input_buf = graph.get_buffer(input_node.output_names[0])
            b, h, w, c = graph.src_axis_order.extract_spatial_dims(input_node.op.shape)
            if input_node.op.input_encoding_in == InputEncodings.NV21:
                # determine expected shape for yuv_nv21(width * height * 3 / 2)
                nv21_shape = int(h * w * (3 / 2))
                input_node.op.shape = [input_node.op.shape[0], nv21_shape]
                input_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                b, h, w, c = graph.src_axis_order.extract_spatial_dims(input_node.op.shape)
                input_node.op.shape = graph.src_axis_order.format_spatial_output_shape(b, h, w, 4)
            input_buf.set_buf_dims(input_node.op.shape)
            color_transform_op = op_adapter.ColorTransformOp(color_transform_name,
                                                             color_transform_output_shape,
                                                             input_encoding_in=input_node.op.input_encoding_in,
                                                             input_encoding_out=input_node.op.input_encoding_out)
            graph.inject(color_transform_op, input_name=input_node.output_names[0],
                         output_name=color_transform_name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_COLOR_TRANSFORM_EXTRACTION")
                       (input_node.op.name, input_node.op.shape, input_node.op.input_encoding_in))

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.TBF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.TBF_TO_BTF)
            buf.axis_format = AxisTracker.AxisFormat.BTF
            node.op.shape = buf.shape


@register_layer_optimization
class OptimizeArgMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeArgMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgMinOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.image_to_spatial_first_order(node, graph)
        elif input_buf.rank() == 2 or input_buf.rank() == 3:
            output_buf = graph.get_output_buffers(node)[0]
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            if input_buf.rank() == 3:
                # add custom permute for 3D use-case. This input use-case is added for batchnorm-1D
                permute_order = [0, 2, 1]  # channel must be last
                AxisTracker.enforce_input_type(graph, node.input_names[0],
                                               AxisTracker.AxisFormat.NONTRIVIAL, permute_order)
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, permute_order)
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank()))

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            if bn_node_.op.compute_statistics == True:
                log_debug("InstanceNorm layer {} cannot be squashed", bn_node_.op.name)
                return False
            return bn_node_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY and bn_input_buffer_.rank() == 4

        sequence = [
                    ("convolution",
                        (),
                        ("MATCH_NUM_BUFS", [("batchnorm", "ALL")])
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)

        for node_tuple in matched_node_list:
            # sanity check
            log_assert(len(node_tuple) == len(sequence),
                       "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                       len(node_tuple), len(sequence))

            conv_node = node_tuple[0]
            bn_node = next(iter(graph.get_output_buffers(conv_node)[0].consumers))
            bn_input_buffer = graph.get_input_buffers(bn_node)[0]

            if bn_input_buffer.axis_format == AxisTracker.AxisFormat.NCS:
                # The Conv weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for BN weight broadcasting and then revert
                weights = numpy.transpose(conv_node.op.weights, (2, 3, 1, 0))
                weights = (weights * bn_node.op.weights)
                weights = numpy.transpose(weights, (3, 2, 0, 1))
            else:
                weights = (conv_node.op.weights * bn_node.op.weights)
            conv_node.op.weights = weights
            conv_node.op.bias = conv_node.op.bias * bn_node.op.weights + bn_node.op.bias
            graph.add_quantization_params(conv_node.op.name, bn_params={"gamma": bn_node.op.gamma,
                                                                        "beta": bn_node.op.beta})
            graph.squash(bn_node, bn_input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                       conv_node.op.type,
                                                                                       conv_node.op.name))


@register_layer_optimization
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeChannelShuffleTranslation, self).axes_to_spatial_first_order(node, graph)
        for buf in graph.get_input_buffers(node):
            log_debug("input {} {} {}", buf.name, buf.axis_format, buf.shape)
        for buf in graph.get_output_buffers(node):
            log_debug("output {} {} {}", buf.name, buf.axis_format, buf.shape)


@register_layer_optimization
class OptimizeColorTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ColorTransformOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC


@register_layer_optimization
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_orders = [buf.axis_format for buf in input_buffers]
        if AxisTracker.AxisFormat.NSC in input_orders or AxisTracker.AxisFormat.NONTRIVIAL in input_orders:
            AxisTracker.image_to_spatial_first_order(node, graph)
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_CONVOLUTION_UNEXPECTED_INPUT_ORDER")
                             (input_orders))
        # if this method is called, current weight order for is NCHW but we want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 1, 0))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register_layer_optimization
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = graph.src_axis_order.permute_sequence[buf.rank() - 1]
            node.op.axis = axis_map[node.op.axis]

    @staticmethod
    def fold_concats(graph):
        def validate_concat_axis(nodes_tuple):
            concat_node_ = nodes_tuple[0]
            concat_node_input_bufs_ = graph.get_input_buffers(concat_node_)
            for buf_ in concat_node_input_bufs_:
                if buf_.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_node_ = buf_.producer
                    # only fold concats with same axis
                    if prev_concat_node_.op.axis != concat_node_.op.axis:
                        log_debug2("Found concat node({}) with a concat input, but axis does not match for input ({}), "
                                   "{} != {} ", concat_node_.op.name, prev_concat_node_.op.name,
                                   prev_concat_node_.op.axis, concat_node_.op.axis)
                        return False

            return True

        sequence = [
                    ("concatenation",
                     ("FLEXIBLE_NUM_BUFS", [("concatenation", "ANY")]),
                     ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_concat_axis)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_node_input_bufs = graph.get_input_buffers(concat_node)

            for buf in concat_node_input_bufs:
                if buf.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_buf = buf  # for readability
                    prev_concat_node = prev_concat_buf.producer

                    # remove prev concat as input from current concat and replace with prev concat's input names
                    prev_concat_inputs = prev_concat_node.input_names
                    idx = concat_node.input_names.index(prev_concat_buf.name)
                    concat_node.input_names.remove(prev_concat_buf.name)
                    # extend the inputs in the same index as prev concat
                    concat_node.input_names[idx:idx] = prev_concat_inputs

                    prev_concat_buf.consumers.remove(concat_node)

                    # we can prune the prev concat node if the current concat was the only consumer.
                    if len(prev_concat_buf.consumers) == 0:
                        graph.prune(prev_concat_node)

                    # remove prev concat as consumer for prev concat's input bufs and replace with current concat
                    for input_name in prev_concat_inputs:
                        input_buf = graph.get_buffer(input_name)
                        input_buf.consumers.add(concat_node)

                    log_debug2(code_to_message.get_debugging_message("DEBUG_CONCAT_FOLD")(prev_concat_node.op.name,
                                                                                          concat_node.op.name))


@register_layer_optimization
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])

        # Permute the constant data if necessary
        if output_buf.axis_format == AxisTracker.AxisFormat.NCS:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            output_buf.axis_format = AxisTracker.AxisFormat.NSC
        elif output_buf.axis_format == AxisTracker.AxisFormat.TBF:
            node.op.tensor = numpy.ascontiguousarray(numpy.transpose(node.op.tensor, AxisTracker.AxisFormat.TBF_TO_BTF))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TBF_TO_BTF)
            output_buf.axis_format = AxisTracker.AxisFormat.BTF

    @staticmethod
    def remove_noop(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if getattr(graph, "weights", None) and getattr(graph.weights, "consumed", None) \
                and graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)


@register_layer_optimization
class OptimizeCropTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        target_buf = None
        if len(node.input_names) > 1:
            target_name = node.input_names[1]
            target_buf = graph.get_buffer(target_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC and (target_buf is None or target_buf.rank() == 4):
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.NCS_TO_NSC)
            node.op.counts = AxisTracker.permute_shape(node.op.counts, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and (target_buf is None or target_buf.rank() == 3):
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, [1, 2, 0])
            node.op.counts = AxisTracker.permute_shape(node.op.counts, [1, 2, 0])
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.offsets = AxisTracker.permute_shape(node.op.offsets, AxisTracker.AxisFormat.TBF_TO_BTF)
            node.op.counts = AxisTracker.permute_shape(node.op.counts, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeCrossCorrelationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CrossCorrelationOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCustomOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CustomOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeCustomOpTranslation, self).axes_to_spatial_first_order(node, graph)

        for i, buf in enumerate(graph.get_output_buffers(node)):
            node.op.output_dims[i] = buf.shape


@register_layer_optimization
class OptimizeDeconvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DeconvolutionOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeDeconvolutionTranslation, self).axes_to_spatial_first_order(node, graph)

        # weights are in CNHW, want HWCN
        weights = numpy.transpose(node.op.weights, (2, 3, 0, 1))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)


@register_layer_optimization
class OptimizeDetectionOutTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DetectionOutputOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)
        self.register_method(MATCH_CAFFE_SSD_TO_TF, self.caffe_ssd_to_tf)

    @staticmethod
    def fold_concats(graph):
        def process_ssd_priorbox_concat_layer(input_buffers_):
            concatenated_priorbox_data = []
            concatenated_priorbox_cz_data = []
            concatenated_priorbox_variance = []
            scale_factors_ = input_buffers_[0].producer.op.scale_factors
            for input_buffer in input_buffers_:
                priorbox_op = input_buffer.producer.op
                concatenated_priorbox_data.extend(priorbox_op.priorbox_box_output[0])
                concatenated_priorbox_variance.extend(priorbox_op.priorbox_box_output[1])
                concatenated_priorbox_cz_data.extend(priorbox_op.priorbox_box_cz_output)
                if scale_factors_ != priorbox_op.scale_factors:
                    # Currently only support 1 set of scale factor for priorboxes.
                    raise ValueError(code_to_message.get_error_message("ERROR_INVALID_PRIORBOX_VARIANCES")
                                     (scale_factors_, input_buffers_[0].producer.op.name,
                                      priorbox_op.scale_factors, priorbox_op.name))

            return concatenated_priorbox_data + concatenated_priorbox_variance, concatenated_priorbox_cz_data, \
                   scale_factors_

        sequence = [
            ("concatenation",
                ("FLEXIBLE_NUM_BUFS", [("noop", "ALL")]),  # noop here since all priorboxes are mapped to noopOp
                ("MATCH_NUM_BUFS", [("detection_output", "ALL")])
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_input_buffers = graph.get_input_buffers(concat_node)
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            detection_out_node = concat_output_buffer.consumers.pop()
            priorbox_data, priorbox_cz_data, scale_factors = process_ssd_priorbox_concat_layer(concat_input_buffers)
            detection_out_node.op.priorbox_data = priorbox_data
            detection_out_node.op.priorbox_center_size_data = priorbox_cz_data
            # order determined per caffe/util/bbox_util.cpp
            detection_out_node.op.scale_x = scale_factors[0]
            detection_out_node.op.scale_y = scale_factors[1]
            detection_out_node.op.scale_w = scale_factors[2]
            detection_out_node.op.scale_h = scale_factors[3]

            # remove concat node.
            detection_out_node.input_names.remove(concat_output_buffer.name)
            graph.prune(concat_node)

            # remove priorboxes
            for buf in concat_input_buffers:
                graph.prune(buf.producer)

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_FOLDING")(concat_node.op.name,
                                                                                           detection_out_node.op.name))

    @staticmethod
    def caffe_ssd_to_tf(graph):
        sequence = [
            ("detection_output",
                ("MATCH_NUM_BUFS", [("reshape", "ANY"), ("concatenation", "ANY")]),  # flattened scores and boxes
                ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            detection_out_node = node_tuple[0]
            for input_name in detection_out_node.input_names:
                node = graph.get_producer_node(input_name)
                if node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    reshape_node = node
                elif node.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    concat_node = node
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_DETECTIONOUT_UNKNOWN_INPUTS")
                                     (node.op.type))

            # 0. Verify valid anchors/priorboxes
            log_assert(detection_out_node.op.code_type == "PRIORBOX_TYPE_CENTER_SIZE",
                       "DetectionOut Op only supports center size code type. Got {}".
                       format(detection_out_node.op.code_type))

            # 1. Pre-process steps
            # Caffe score input is flattened, remove reshape to match shape [batch, num_anchors, num_classes]
            reshape_output_buffer = graph.get_output_buffers(reshape_node)[0]
            detection_out_node.input_names.remove(reshape_output_buffer.name)
            detection_out_node.input_names.insert(0, reshape_node.input_names[0])
            graph.get_buffer(reshape_node.input_names[0]).consumers.add(detection_out_node)

            reshape_output_buffer.consumers.remove(detection_out_node)
            # remove reshape node if applicable.
            if len(reshape_output_buffer.consumers) == 0:
                graph.prune(reshape_node)

            # Caffe boxes(location) data is also flattened. Reshape to [batch, num_boxes, 4]
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            concat_buf_shape = concat_output_buffer.shape
            # add reshape node
            reshape_name = concat_node.op.name + "_preprocess_reshape"
            reshape_op = op_adapter.ReshapeOp(reshape_name, output_shape=[concat_buf_shape[0],
                                                                          int(concat_buf_shape[1] / 4),
                                                                          4])
            graph.inject(reshape_op, input_name=concat_node.output_names[0], output_name=reshape_name,
                         consumer_names=detection_out_node.output_names)

            # DetectionOut in IR has priorboxes as param, need to add those to input instead
            detection_out_name = detection_out_node.op.name
            detection_out_node_idx = graph.nodes_in_order.index(detection_out_node)
            prior_box_name = detection_out_name + "_anchors"
            pbox_data = numpy.asarray(detection_out_node.op.priorbox_center_size_data, dtype=numpy.float32)\
                        .reshape(int(len(detection_out_node.op.priorbox_center_size_data)/4), 4)
            prior_box_op = op_adapter.ConstantOp(name=prior_box_name, tensor=pbox_data)
            graph.add(prior_box_op, input_names=[], output_names=[prior_box_name], idx=detection_out_node_idx-1)
            detection_out_node.input_names.append(prior_box_name)

            # Caffe Ssd scales is the reciprocal compared to TF scales
            detection_out_node.op.scale_y = 1 / detection_out_node.op.scale_y
            detection_out_node.op.scale_x = 1 / detection_out_node.op.scale_x
            detection_out_node.op.scale_h = 1 / detection_out_node.op.scale_h
            detection_out_node.op.scale_w = 1 / detection_out_node.op.scale_w

            # 2. Change DetectionOut's single output to multiple. Outputs:
            #    Expected: scores[1, max_num_det], boxes[1, max_num_det, 4], classes[1, max_num_det], num_det[batch],
            #    Caffe Style: 1 output of shape [1, 1, max_num_det, 7]
            #                   7(last dim above): [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            detection_out_buf = graph.get_buffer(detection_out_node.output_names[0])
            boxes_shape = [detection_out_buf.shape[0], detection_out_node.op.keep_top_k, 4]  # [batch, max_num_detections, 4)
            boxes_name = detection_out_name + "_boxes"
            boxes_buf = op_graph.Buffer(boxes_name, boxes_shape, detection_out_node)
            graph.buffers[boxes_name] = boxes_buf

            scores_name = detection_out_name + "_scores"
            scores_buf = op_graph.Buffer(scores_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[scores_name] = scores_buf

            classes_name = detection_out_name + "_classes"
            classes_buf = op_graph.Buffer(classes_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[classes_name] = classes_buf

            num_det_name = detection_out_name + "_num_detections"
            num_det_buf = op_graph.Buffer(num_det_name, [boxes_shape[0]], detection_out_node)
            graph.buffers[num_det_name] = num_det_buf

            del graph.buffers[detection_out_node.output_names[0]]
            detection_out_node.output_names = [boxes_name, scores_name, classes_name, num_det_name]

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_CAFFE_TO_TF_STYLE")
                       (detection_out_node.op.name))


@register_layer_optimization
class OptimizeElementwiseAndTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseAndOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseBinaryDivTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseBinaryDivOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_div)

    @staticmethod
    def squash_div(graph):
        def validate_node(nodes_tuple):
            return translation_utils.validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
                    ("elementwise_binary_div",
                        ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),  # if either of the inputs are const
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_eltwisebinary_to_nn_node(graph, matched_node_list)


@register_layer_optimization
class OptimizeElementwiseBinaryProductTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseBinaryProductOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_prod)

    @staticmethod
    def squash_prod(graph):
        def validate_node(nodes_tuple):
            return translation_utils.validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
                    ("elementwise_binary_product",
                        ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),  # if either of the inputs are const
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_eltwisebinary_to_nn_node(graph, matched_node_list)


@register_layer_optimization
class OptimizeElementwiseBinarySubTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseBinarySubOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_sub)

    @staticmethod
    def squash_sub(graph):
        def validate_node(nodes_tuple):
            return translation_utils.validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
                    ("elementwise_binary_sub",
                        ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),  # if either of the inputs are const
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_eltwisebinary_to_nn_node(graph, matched_node_list)


@register_layer_optimization
class OptimizeElementwiseDivTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseDivOp.TRANSLATION_KEY
        self.register_method(SQUASH_DIV, self.squash_div)

    @staticmethod
    def squash_div(graph):
        def validate_node(nodes_tuple):
            prod_node = nodes_tuple[0]
            if hasattr(prod_node.op, 'weights'):
                input_buffer_ = graph.get_input_buffers(prod_node)[0]
                prev_ = input_buffer_.producer
                log_assert(hasattr(prev_.op, 'weights'),
                           code_to_message.get_error_message("ERROR_DIV_SCALE_PREV_NO_WEIGHTS")(prev_.op.name,
                                                                                                   prev_.op.type))
                return True
            return False

        sequence = [
            ("elementwise_div", (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_nodes_into_previous(graph, matched_node_list, "DEBUG_ELEMENTWISEDIV_SQUASH")


@register_layer_optimization
class OptimizeElementwiseEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseGreaterTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseGreaterOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseGreaterEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseGreaterEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseLessTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseLessOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseLessEqualTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseLessEqualOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseMaxOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseMaxOp)


@register_layer_optimization
class OptimizeElementwiseMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseMinOp.TRANSLATION_KEY
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseMinOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseMinOp)


@register_layer_optimization
class OptimizeElementwiseProductTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseProductOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_prod)
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def squash_prod(graph):
        def validate_node(nodes_tuple):
            prod_node = nodes_tuple[0]
            if hasattr(prod_node.op, 'weights'):
                input_buffer_ = graph.get_input_buffers(prod_node)[0]
                prev_ = input_buffer_.producer
                log_assert(hasattr(prev_.op, 'weights'),
                           code_to_message.get_error_message("ERROR_MUL_SCALE_PREV_NO_WEIGHTS")(prev_.op.name,
                                                                                                prev_.op.type))
                return True
            return False

        sequence = [
                    ("elementwise_product", (), ())
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_nodes_into_previous(graph, matched_node_list, "DEBUG_ELEMENTWISEPRODUCT_SQUASH")

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseProductOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseProductOp)


@register_layer_optimization
class OptimizeElementwiseSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSumOp.TRANSLATION_KEY
        self.register_method(SQUASH_SUM, self.squash_sum)
        self.register_method(BROADCAST_CONST, self.broadcast_const)
        self.register_method(CHAIN_ELTWISE_OPS, self.chain_eltwise_ops)

    @staticmethod
    def squash_sum(graph):
        def validate_node(nodes_tuple):
            if (translation_utils.validate_eltwise_pattern(graph, nodes_tuple, "bias")):
                return True
            return False

        sequence = [
                    ("elementwise_sum", (), ())
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_eltwisebinary_to_nn_node(graph, matched_node_list)

    @staticmethod
    def broadcast_const(graph):
        sequence = [
            (op_adapter.ElementwiseSumOp.TRANSLATION_KEY,
             ("FLEXIBLE_NUM_BUFS", [(op_adapter.ConstantOp.TRANSLATION_KEY, 1)]),  # If either input is Const
             ())
        ]

        # Find the const and sum nodes in question
        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            sum_node = nodes_tuple[0]
            const_input_nodes = []
            non_const_input_nodes = []

            # Get the const buffer and the broadcast shape
            for node in graph.get_op_input_nodes(sum_node):
                if node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    const_input_nodes.append(node)
                else:
                    non_const_input_nodes.append(node)
            if len(non_const_input_nodes) < 1:
                # Need at least 1 non-const input to broadcast to
                return

            non_const_output_bufs = [in_buf
                                     for in_node
                                     in non_const_input_nodes
                                     for in_buf
                                     in graph.get_output_buffers(in_node)
                                    ]
            input_bufs = graph.get_input_buffers(sum_node)
            non_const_shapes = [in_buf.shape
                                for in_buf
                                in input_bufs
                                if in_buf in non_const_output_bufs
                                ]

            # Converts shapes to NSC, calculates broadcasted shape, and converts result to source axis ordering
            non_const_shapes_ir = [graph.src_axis_order.permute_shape_to_ir(shape) for shape in non_const_shapes]
            non_const_max_shape_ir = translation_utils.get_broadcasted_shape(non_const_shapes_ir)
            non_const_max_shape = graph.src_axis_order.permute_shape_from_ir(non_const_max_shape_ir)

            if non_const_max_shape not in non_const_shapes:
                raise ValueError("Since broadcasting is not supported, Add Operation is not "
                                 "supported with the following dynamic dimensions: {}".format(non_const_shapes))

            for const_input_node in const_input_nodes:
                const_shape = list(numpy.shape(const_input_node.op.tensor))
                const_input_buf = graph.get_output_buffers(const_input_node)[0]
                const_tensor = const_input_node.op.tensor

                if non_const_max_shape > const_shape:
                    try:
                        broadcasted_tensor = numpy.zeros(non_const_max_shape, dtype=numpy.float32)
                        const_tensor = broadcasted_tensor + const_tensor
                    except ValueError as e:
                        log_error("Broadcasting failed in Op {} for const tensor {} of shape {} to shape {}".format(
                            sum_node.op.name, const_input_buf.name, const_shape, non_const_shapes
                        ))
                        raise e
                else:
                    # No change in const tensor, no need to modify the Const Node
                    continue

                # If the const Op has multiple consumers, they may require different broadcast shapes
                # So Create a copy of the Constant Op before assigning the broadcast tensor
                # Else, replace the tensor in the Constant Op and assign the new shape to Buffer
                const_consumers = graph.get_op_output_nodes(const_input_node)
                if len(const_consumers) > 1:
                    # Create new Constant Op and insert in the Graph
                    new_op_name = const_input_node.op.name + "_copy_" + sum_node.op.name
                    new_const_op = op_adapter.ConstantOp(new_op_name, const_tensor)
                    idx = graph.list_nodes().index(sum_node)
                    graph.add(new_const_op, [], [new_op_name], idx=idx)
                    const_input_idx = sum_node.input_names.index(const_input_buf.name)
                    sum_node.input_names[const_input_idx] = new_const_op.name
                    const_input_buf.consumers.remove(sum_node)
                else:
                    const_input_node.op.tensor = const_tensor
                    const_input_buf.shape = non_const_max_shape

    @staticmethod
    def chain_eltwise_ops(graph):
        def validate_node(nodes_tuple):
            return len(nodes_tuple[0].input_names) > 2

        sequence = [
            (op_adapter.ElementwiseSumOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.chain_matched_eltwise_ops(graph, matched_node_list, op_adapter.ElementwiseSumOp)


@register_layer_optimization
class OptimizeElementwiseOrTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseOrOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryAbsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryAbsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryExpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryExpOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryFloorTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryFloorOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryLogTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryLogOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnaryNegTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY
        self.register_method(OPTIMIZE_NEG, self.optimize_negation)

    @staticmethod
    def optimize_negation(graph):
        def validate_neg(nodes_tuple):
            for input_name_ in nodes_tuple[0].input_names:
                node_ = graph.get_producer_node(input_name_)
                if node_.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                        all(val == -1 for val in numpy.array(node_.op.tensor).flatten()):
                    return True

            return False

        # Optimization: -1 * A => Neg(A)
        sequences = [
            [
                ("elementwise_binary_product",
                 ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                 ())
            ],
            [
                ("elementwise_product",
                 ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_neg)
            for node_tuple in matched_node_list:
                prod_node = node_tuple[0]
                non_const_input_node = None
                const_input_node = None
                for input_name in prod_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    if input_node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
                        non_const_input_node = input_node
                    else:
                        const_input_node = input_node
                const_input_buf = graph.get_buffer(const_input_node.output_names[0])

                # remove const as input to prod, the prod node will then be replaced as Neg
                const_input_buf.consumers.remove(prod_node)
                prod_node.input_names.remove(const_input_node.output_names[0])
                if len(const_input_buf.consumers) == 0:
                    graph.prune(const_input_node)

                neg_op = op_adapter.ElementwiseUnaryNegOp(None)
                neg_op.name = graph.naming_policy.get_op_name(neg_op)
                graph.replace(prod_node.op, neg_op)
                log_debug2("Optimization of -1 * A => Neg(A) complete. Op {} replaced with NegOp"
                           .format(prod_node.op.name))

        # Optimization: A + Neg(B) => A - B
        #               Neg(A) + B => B - A
        #               Neg(A) + Neg(B) => Neg(A) - B
        sequences = [
            [
                ("elementwise_sum",
                 ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_neg", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence)
            for node_tuple in matched_node_list:
                sum_node = node_tuple[0]
                neg_node_to_prune = None
                for input_name in sum_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    input_buf = graph.get_buffer(input_name)
                    if input_node.op.type == op_adapter.ElementwiseUnaryNegOp.TRANSLATION_KEY:
                        # if more than consumer of NegOp then we cant remove it hence optimization
                        # is not really relevant.
                        if len(input_buf.consumers) == 1:
                            neg_node_to_prune = input_node

                if neg_node_to_prune is not None:
                    # Update the input and consumer list and remove NegOp from graph
                    neg_idx = sum_node.input_names.index(neg_node_to_prune.output_names[0])
                    sum_input_names = sum_node.input_names[:]
                    neg_input_name = neg_node_to_prune.input_names[0]
                    neg_input_buf = graph.get_buffer(neg_input_name)
                    graph.prune(neg_node_to_prune, force_remove=True)
                    if neg_idx == 0:
                        # got Neg(A) + B, need B - A
                        sum_input_names[0] = sum_input_names[1]
                        sum_input_names[1] = neg_input_name
                    else:
                        # Neg(A) + Neg(B) or A + Neg(B)
                        sum_input_names[neg_idx] = neg_input_name
                    neg_input_buf.consumers.add(sum_node)
                    sum_node.input_names = sum_input_names

                    sub_op = op_adapter.ElementwiseBinarySubOp(None)
                    sub_op.name = graph.naming_policy.get_op_name(sub_op)
                    graph.replace(sum_node.op, sub_op)
                    log_debug2("Optimization of addition to a negative of an op (e.g: A + Neg(B) => A - B) complete. "
                               "Op {} replaced with SubOp"
                               .format(sum_node.op.name))


@register_layer_optimization
class OptimizeElementwiseUnaryNotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryNotOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseUnarySinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnarySinOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseSubTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseSubOp.TRANSLATION_KEY
        self.register_method(SQUASH_SUB, self.squash_sub)

    @staticmethod
    def squash_sub(graph):
        def validate_node(nodes_tuple):
            sub_node = nodes_tuple[0]
            if hasattr(sub_node.op, 'bias'):
                input_buffer_ = graph.get_input_buffers(sub_node)[0]
                prev_ = input_buffer_.producer
                log_assert(hasattr(prev_.op, 'bias'),
                           code_to_message.get_error_message("ERROR_BIAS_SUB_PREV_NO_BIAS")(sub_node.op.name,
                                                                                            prev_.op.name,
                                                                                            prev_.op.type))
                return True
            return False

        sequence = [
                    ("elementwise_sub", (), ())
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        translation_utils.squash_nodes_into_previous(graph, matched_node_list, "DEBUG_ELEMENTWISESUB_SQUASH")


@register_layer_optimization
class OptimizeElementwiseUnarySqrtTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnarySqrtOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_to_spatial_first_order(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            AxisTracker.enforce_input_type(graph, input_buf.name, AxisTracker.AxisFormat.NSC,
                                             AxisTracker.AxisFormat.NCS_TO_NSC)

            # weights expect NCHW order, need to permute
            input_buf = graph.get_input_buffers(node)[0]
            batch, height, width, depth = input_buf.shape
            weights = node.op.weights

            # Assuming FC: W^Tx + b and weights have shape (input_size, output_size)
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            log_assert(input_size == depth * height * width,
                       code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                      (input_size, output_size),
                                                                                      (batch,  height, width, depth)))
            weights.shape = (depth, height, width, output_size)
            weights = numpy.transpose(weights, (3, 1, 2, 0))
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)
            weights.shape = (output_size, input_size)
            node.op.weights = weights
        else:
            # again, need to transpose weights for spatial_first order
            weights = node.op.weights
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))
            node.op.weights = weights

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.FEATURE

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            if bn_node_.op.compute_statistics == True:
                log_debug("InstanceNorm layer {} cannot be squashed", bn_node_.op.name)
                return False
            return True

        sequence = [
            ("fully_connected",
                (),
                ("MATCH_NUM_BUFS", [("batchnorm", "ALL")])
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)

        for node_tuple in matched_node_list:
            # sanity check
            log_assert(len(node_tuple) == len(sequence),
                       "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                       len(node_tuple), len(sequence))

            fc_node = node_tuple[0]
            bn_node = next(iter(graph.get_output_buffers(fc_node)[0].consumers))
            bn_input_buffer = graph.get_input_buffers(bn_node)[0]
            weights = fc_node.op.weights
            broadcasted_tensor = numpy.zeros(len(bn_node.op.weights), dtype=numpy.float32)
            if fc_node.op.transpose_b == False:
                weight_tensor = numpy.transpose(weights, (1, 0)).copy()
            else:
                weight_tensor = weights.copy()
            broadcasted_tensor = broadcasted_tensor + weight_tensor
            broadcasted_tensor = broadcasted_tensor * bn_node.op.weights
            if fc_node.op.transpose_b == False:
                broadcasted_transpose = numpy.transpose(broadcasted_tensor, (1, 0)).copy()
            else:
                broadcasted_transpose = broadcasted_tensor.copy()
            fc_node.op.weights = broadcasted_transpose
            fc_node.op.bias = fc_node.op.bias * bn_node.op.weights + bn_node.op.bias
            graph.squash(bn_node, bn_input_buffer.name)
            log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                       fc_node.op.type,
                                                                                       fc_node.op.name))


@register_layer_optimization
class OptimizeGatherTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis and if needed permute it for NSC
        # In addition, output buffer axis tracking stays the same as input so long
        # as the rank of indices == 1. Otherwise it's non trivial as the rank will change
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        output_buf = graph.get_output_buffers(node)[0]
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            if indices_buf.rank() > 1:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        else:
            if indices_buf.rank() > 1:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                output_buf.axis_format = input_buf.axis_format


@register_layer_optimization
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeL2NormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2NormOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeL2NormTranslation, self).axes_to_spatial_first_order(node, graph)

        # transform axis to the correct index, also ensures axis is always positive
        input_buf = graph.get_input_buffers(node)[0]
        axis_map = graph.src_axis_order.permute_sequence[input_buf.rank() - 1]
        node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeL2PoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2PoolOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeLstmTranslation, self).axes_to_spatial_first_order(node, graph)

        # weights are expected to be  NxK, we want KxN
        node.op["input_weights"] = numpy.ascontiguousarray(node.op.input_weights.transpose(), dtype=numpy.float32)
        node.op["hidden_state_weights"] = numpy.ascontiguousarray(node.op.hidden_state_weights.transpose(),
                                                                  dtype=numpy.float32)


@register_layer_optimization
class OptimizeMaxYTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MaxYOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NeuronOp.TRANSLATION_KEY
        self.register_method(MATCH_HARDSWISH, self.match_hardswish)

    @staticmethod
    def match_hardswish(graph):
        def is_valid_hardswish(node_tuple):
            def check_for_valid_add_node(input_const_name):
                const_input_node = graph.get_producer_node(input_const_name)
                const_input_value = const_input_node.op.tensor
                const_input_length = reduce(lambda x,y:x * y, const_input_value.shape)
                temp = set(const_input_value.reshape(const_input_length))
                if len(temp) != 1 or int(temp.pop()) != 3:
                    return False
                return True

            def check_for_valid_neuron_node(node):
                if node.op.neuron_type != 'NEURON_RELU_MIN_MAX' \
                        or int(node.op.attrs['min_clamp']) != 0 \
                        or int(node.op.attrs['max_clamp']) != 6:
                    return False
                return True

            def check_for_valid_div_node(node):
                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if const_input_value.shape != (1,) or int(const_input_value) != 6:
                    return False
                return True

            def check_for_valid_mul_node_with_const_input(node):
                def is_close_to_one_sixth(num):
                    return translation_utils.compare_values(float(num[0]), 1/6, rtol=1.e-3, atol=1.e-5)

                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if const_input_value.shape != (1,) or not is_close_to_one_sixth(const_input_value):
                    return False
                return True

            add_node, neuron_node = node_tuple[0], node_tuple[1]
            add_non_const_input_name, add_const_input_name, mul_node, mul_node_const_input, div_node = [None] * 5
            for input_name in add_node.input_names:
                if graph.get_producer_op(input_name).type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    add_const_input_name = input_name
                else:
                    add_non_const_input_name = input_name

            for node in node_tuple[2:]:
                if node.op.type == op_adapter.ElementwiseBinaryDivOp.TRANSLATION_KEY:
                    div_node = node
                else:
                    mul_input_names = node.input_names
                    if len(mul_input_names) != 2:
                        return False
                    if any(op_adapter.ConstantOp.TRANSLATION_KEY == graph.get_producer_op(input_name).type
                           for input_name in mul_input_names):
                        mul_node_const_input = node
                    else:
                        mul_node = node

            if not add_const_input_name or not mul_node or (not div_node and not mul_node_const_input):
                return False

            if add_non_const_input_name not in mul_node.input_names:
                # the add and mul must share same input_name to be matched as hswish
                return False

            return (check_for_valid_add_node(add_const_input_name) and
                    check_for_valid_neuron_node(neuron_node) and
                    (check_for_valid_div_node(div_node) if div_node else
                     check_for_valid_mul_node_with_const_input(mul_node_const_input)))

        def get_input_const_nodes(input_names):
            input_nodes = [graph.buffers[name].producer for name in input_names]
            const_nodes = [node for node in input_nodes if
                           node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY]
            return const_nodes

        def remove_const_nodes(node_tuple, matched_sequence_flag):
            if matched_sequence_flag[-1] in ['1', '3']:
                nodes_with_const_input = [node_tuple[0], node_tuple[3]]
                mul_node = node_tuple[2]
            else:
                nodes_with_const_input = [node_tuple[0], node_tuple[2]]
                mul_node = node_tuple[3]

            for node in nodes_with_const_input:
                const_node = get_input_const_nodes(node.input_names)[0]
                graph.prune(const_node, force_remove=True)

        # Y = X*RELU6(X+3)*(1/6)
        sequence1 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)*(1/6))
        sequence2 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             ("MATCH_NUM_BUFS", [("neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             (),
             ()
             )
        ]

        # Y = X*RELU6(X+3)/6
        sequence3 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_binary_div", "ALL")])
             ),
            ("elementwise_binary_div",
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)/6)
        sequence4 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("neuron", "ALL")])
             ),
            ("neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_div", "ALL")])
             ),
            ("elementwise_binary_div",
             ("MATCH_NUM_BUFS", [("neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_binary_product", "ALL")])
             ),
            ("elementwise_binary_product",
             (),
             ()
             )
        ]
        sequences = [sequence1, sequence2, sequence3, sequence4]

        for index, sequence in enumerate(sequences):
            matched_sequence_flag = 'matched_sequence' + str(index + 1)
            matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_hardswish, ignore_constants=True)

            for node_tuple in matched_node_list:
                remove_const_nodes(node_tuple, matched_sequence_flag)
                add_node = node_tuple[0]
                for node in node_tuple[:0:-1]:
                    for input_name in node.input_names:
                        if len(graph.get_buffer(input_name).consumers) == 1:
                            # per the sequence matching we know one of the inputs are squashable, hence
                            # check which either one has 1 consumer
                            graph.squash(node, input_name)

                add_op = add_node.op
                add_op_name = graph.naming_policy.get_op_name(add_op)
                hardswish_op_name = add_op_name + '_Hswish'
                hardswish_op = op_adapter.NeuronOp(hardswish_op_name, "NEURON_HSWISH")
                graph.replace(add_op, hardswish_op)


@register_layer_optimization
class OptimizeNoopTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NoopOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

    @staticmethod
    def remove_noop(node, graph):
        graph.squash(node, node.input_names[0])


@register_layer_optimization
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.pads = AxisTracker.permute_shape(node.op.pads, AxisTracker.AxisFormat.TBF_TO_BTF)
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PoolOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizePermuteTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PermuteOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            # special case: transforming to NSC, will become noop
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                return
            else:
                # going to nontrivial
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            if node.op.order == [0, 2, 3, 1]:
                node.op.order = [0, 1, 2, 3]
                output_buf.axis_format = AxisTracker.AxisFormat.BTF
            else:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TBF,
                                              AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
                output_buf.axis_format = AxisTracker. AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            if len(node.op.order) == 4:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif len(node.op.order) > 6:
                raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_TOO_MANY_DIMENSIONS")(node.op.order))
            else:
                # nothing to be done
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER")
                             (input_buf.axis_format))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.order == list(range(len(node.op.order))):
            # this permute is trivial, remove it
            graph.squash(node, input_buffer.name)


@register_layer_optimization
class OptimizePowerTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PowerOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        input_buf = graph.get_buffer(node.input_names[0])
        # Permute the power data if necessary
        power_rank = len(node.op.power.shape)
        if power_rank == 4 and input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            node.op.power = numpy.ascontiguousarray(numpy.transpose(node.op.power, AxisTracker.AxisFormat.NCS_TO_NSC))
        elif power_rank == 3 and input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            node.op.power = numpy.ascontiguousarray(numpy.transpose(node.op.power, AxisTracker.AxisFormat.TBF_TO_BTF))


@register_layer_optimization
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizePreluTranslation, self).axes_to_spatial_first_order(node, graph)

        input_buf = graph.get_buffer(node.input_names[0])
        coeff_shape = node.op.coeff.shape
        # input and coeff might not be broadcastable after axis transformation, hence adjust as needed
        if not translation_utils.broadcastable(input_buf.shape, coeff_shape):
            # determine the permute order(if any) after spatial first transformation
            # Note: only NSC, BTF formats imply permute was done.
            input_permute_order = None
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                input_permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
            elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                input_permute_order = AxisTracker.AxisFormat.TBF_TO_BTF

            if input_permute_order is not None:
                # The input has been permuted hence we need to adjust coeff so that broadcasting persists
                rank_diff = len(coeff_shape) - input_buf.rank()
                if rank_diff < 0:
                    # Prepending 1s as its needed if coeff shorter in length to properly permute below
                    coeff_shape = [1] * abs(rank_diff) + list(coeff_shape)
                    node.op.coeff = numpy.broadcast_to(node.op.coeff, coeff_shape)
                    rank_diff = 0

                # determine coefficient permute order. Only need to permute the length of the input.
                # eg: input_permute_order = [1, 0], coeff = [1, 10, 7, 5]:
                #     coeff_permute_order = [0, 1, 3, 2] and updated_coeff = [1, 10, 5, 7]
                coeff_permute_order = list(range(len(coeff_shape)))
                # loop backwards with the len of input order and update
                for i in range(-(len(input_permute_order)), 0):
                    coeff_permute_order[i] = input_permute_order[i] + rank_diff
                node.op.coeff = numpy.transpose(node.op.coeff, coeff_permute_order)
                coeff_shape = node.op.coeff.shape

            if not translation_utils.broadcastable(input_buf.shape, coeff_shape):
                raise ValueError(code_to_message.get_error_message("ERROR_OPERATION_INPUTS_NOT_BROADCASTABLE")
                                 (node.op.name, input_buf.name, "coeff", input_buf.shape, coeff_shape))


@register_layer_optimization
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):

        # change input dims to 4D as required by snpe. Handling this here since converter allows for
        # none 4D inputs. Note: only change dimensions if it is input and no other node is consuming it
        # TODO: how should this be really handled
        im_info_input_buf = graph.get_input_buffers(node)[-1]
        if im_info_input_buf.producer.op.type == op_adapter.InputOp.TRANSLATION_KEY \
                and len(im_info_input_buf.consumers) == 1 \
                and im_info_input_buf.rank() != 4:
            shape = translation_utils.expand_to_rank(im_info_input_buf.shape, 4)
            im_info_input_buf.shape = shape
            im_info_input_buf.producer.op.shape = shape
            im_info_input_buf.axis_format = AxisTracker.AxisFormat.NSC

        super(OptimizeProposalTranslation, self).axes_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeReduceMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        # TO-DO: We should be using a common function to do this
        # something that takes in the needed args
        if input_buf.axis_format in format_to_permute_order:
            target_format = format_to_format[input_buf.axis_format]
            permute_order = format_to_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, target_format,
                                              permute_order, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]


@register_layer_optimization
class OptimizeReduceMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMeanOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        # TO-DO: We should be using a common function to do this
        # something that takes in the needed args
        if input_buf.axis_format in format_to_permute_order:
            target_format = format_to_format[input_buf.axis_format]
            permute_order = format_to_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, target_format,
                                              permute_order, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]


@register_layer_optimization
class OptimizeReduceMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceMinOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        # TO-DO: We should be using a common function to do this
        # something that takes in the needed args
        if input_buf.axis_format in format_to_permute_order:
            target_format = format_to_format[input_buf.axis_format]
            permute_order = format_to_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, target_format,
                                              permute_order, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]


@register_layer_optimization
class OptimizeReduceSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceSumOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        if input_buf.axis_format in format_to_permute_order:
            target_format = format_to_format[input_buf.axis_format]
            permute_order = format_to_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                graph.inject_implicit_permute(input_name, target_format,
                                              permute_order, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.eltwise_to_spatial_first_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]


@register_layer_optimization
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)
        self.register_method(REMOVE_NOOP, self.remove_noop)

    @staticmethod
    def product(nums):
        if len(nums) == 0:
            return 1
        else:
            return reduce(mul, nums)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.PermuteOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TBF,
                                              AxisTracker.AxisFormat.TBF_TO_BTF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                pass
            elif input_buf.axis_format == AxisTracker.AxisFormat.FEATURE or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        # if output_buf.rank() > 4:
        #     log_assert(self.product(output_buf.shape[:-4]) == 1,
        #                code_to_message.get_error_message("ERROR_RESHAPE_BATCH_UNSUPPORTED"))
        #     output_buf.shape = output_buf.shape[-4:]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    @staticmethod
    def match_channelshuffle(graph):
        def is_valid_channelshuffle(nodes_tuple):
            def check_for_valid_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[3] and
                        reshape_1_input_shape[3] == reshape_1_output_shape[4])

            def check_for_valid_permute(node):
                # Assuming the input shape is N[GC']HW
                return node.op.type == op_adapter.PermuteOp.TRANSLATION_KEY and node.op.order == [0, 2, 1, 3, 4]

            def check_for_valid_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                        reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                        reshape_2_input_shape[3] == reshape_2_output_shape[2] and
                        reshape_2_input_shape[4] == reshape_2_output_shape[3])

            first_, second_, third_ = nodes_tuple
            input_shape_ = graph.get_input_buffers(first_)[0].shape
            output_shape_ = graph.get_output_buffers(third_)[0].shape

            return ((output_shape_ == input_shape_) and
                    check_for_valid_reshape_1(first_) and
                    check_for_valid_permute(second_) and
                    check_for_valid_reshape_2(third_))

        sequence = [
                    ("reshape",
                        (),
                        ("MATCH_NUM_BUFS", [("permute", "ALL")])
                     ),
                    ("permute",
                        (),
                        ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                     ),
                    ("reshape",
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_channelshuffle)

        for node_tuple in matched_node_list:

                # ChannelShuffle Op found,
                # Squash Permute and 2nd Reshape Op and
                # Replace 1st ReshapeOp with ShuffleOp
                first, second, third = node_tuple
                third_input_buffer = graph.get_input_buffers(third)[0]
                graph.squash(third, third_input_buffer.name)

                second_input_buffer = graph.get_input_buffers(second)[0]
                graph.squash(second, second_input_buffer.name)

                output_shape = first.op.output_shape
                # Assuming the shape is N[GC']HW
                groups = output_shape[1]
                shuffle_op = op_adapter.ChannelShuffleOp(None, groups=groups)
                shuffle_op.name = graph.naming_policy.get_op_name(shuffle_op)
                graph.replace(first.op, shuffle_op)
                log_debug2(code_to_message.get_debugging_message("DEBUG_CHANNEL_SHUFFLE_REPLACE")(first.op.name,
                                                                                                  second.op.name,
                                                                                                  third.op.name,
                                                                                                  shuffle_op.name))

    @staticmethod
    def remove_noop(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        if input_buffer.shape == node.op.output_shape:
            # this reshape has no effect, remove it
            ret = graph.squash(node, input_buffer.name)
            if ret:
                log_debug("Squash Reshape op {} due to Noop. "
                          "Input shape {}, shape attr {}".format(node.op.name,
                                                                 input_buffer.shape,
                                                                 node.op.output_shape))


@register_layer_optimization
class OptimizeRNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RNormOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_type(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC


@register_layer_optimization
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY
        self.register_method("PREPROCESS_ROI_POOL_INPUTS", self.preprocess_roi_pool_inputs)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.enforce_input_type(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC

    @staticmethod
    def preprocess_roi_pool_inputs(graph):
        def validate_node(nodes_tuple):
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])
            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY or len(roi_node.input_names) == 3:
                return True
            return False

        sequence = [(op_adapter.RoiPoolingOp.TRANSLATION_KEY, (), ())]

        matched_nodes_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_nodes_list:
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])

            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                if roi_buf.shape[-1] == 5:
                    # QNN needs roi values to be separated from batch index
                    roi_values = roi_buf.producer.op.tensor
                    roi_values_no_batch = roi_values[:, 1:]

                    # Update ROI values in constant op to new values
                    roi_buf.producer.op.tensor = roi_values_no_batch

                    # Set batch indices to first sub-tensor of ROI values
                    batch_indices_name = roi_buf.name + "_batch_indices"
                    batch_indices = numpy.asarray(roi_values[:, 0], numpy.int32)

                    # Add a new constant op to capture batch indices

                    # constant op needs to be added before roi node
                    roi_idx = graph.nodes_in_order.index(roi_node)
                    graph.add(op_adapter.ConstantOp(batch_indices_name, batch_indices, quantizable=False), [],
                              [batch_indices_name], idx=roi_idx)

                    # add input name to roi node
                    roi_node.input_names.append(batch_indices_name)

                else:
                    raise ValueError("Expected 5 dimensions for static ROI buffer: {}, instead got {}"
                                     .format(roi_buf.name, roi_buf.shape[-1]))
            elif len(roi_node.input_names) != 3:
                raise AttributeError("Missing batch indices input. "
                                     "Expected 3 inputs for ROI operation instead got: {}"
                                     .format(len(roi_node.input_names)))


@register_layer_optimization
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        node.op.output_shape = AxisTracker.permute_shape(node.op.output_shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        AxisTracker.image_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeRnnTransformationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RnnTransformationOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.time_series_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeScaleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScaleOp.TRANSLATION_KEY
        self.register_method(SQUASH_SCALE, self.squash_scale)

    @staticmethod
    def squash_scale(graph):
        def validate_node(nodes_tuple):
            scale_node_ = nodes_tuple[0]
            input_buffer_ = graph.get_input_buffers(scale_node_)[0]
            # scale should only be folded if it is the only layer that depends on the output of the previous
            # batchnorm layer/op.
            if len(input_buffer_.consumers) == 1:
                return True
            return False

        sequence = [
            ("scale",
             # Check if the previous layer was a batchnorm
             ("MATCH_NUM_BUFS", [("batchnorm", "ALL")])
             ,
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            # retain scale information in batchnorm op so that it can be used for quantization
            # scale_weights and scale_bias map to gamma and beta respectively.
            node = node_tuple[0]
            prev = graph.get_input_buffers(node)[0].producer
            prev.op.gamma = node.op.weights
            prev.op.beta = node.op.bias

        translation_utils.squash_nodes_into_previous(graph, matched_node_list, "DEBUG_SCALE_SQUASH")

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeScaleTranslation, self).axes_to_spatial_first_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NSC:
            axis_map = graph.src_axis_order.permute_sequence[buf.rank() - 1]
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]
        AxisTracker.eltwise_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        input_buf = graph.get_buffer(node.input_names[0])
        # Added this check for any 4D input for frcnn_vgg_compressed model
        # where it expects a permute after reshape
        if input_buf.rank() == 4:
            AxisTracker.image_to_spatial_first_order(node, graph)
        elif input_buf.axis_format == AxisTracker.AxisFormat.BTF:
            AxisTracker.time_series_to_spatial_first_order(node, graph)
        else:
            AxisTracker.feature_to_spatial_first_order(node, graph)

        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeStaticTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StaticOp.TRANSLATION_KEY
        self.register_method(REMOVE_NOOP, self.remove_noop)

    def axes_to_spatial_first_order(self, node, graph):
        pass

    @staticmethod
    def remove_noop(node, graph):
        graph.prune(node)


@register_layer_optimization
class OptimizeSubtractMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeUdlTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UdlOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_names = node.input_names
        for input_name in input_names:
            input_buf = graph.get_buffer(input_name)
            current_input_order = input_buf.get_axis_annotations()
            expected_input_order = []
            for dims in node.op.expected_input_axis_orders:
                if len(dims) == input_buf.rank():
                    expected_input_order = dims
            target_input_type = AxisTracker.get_axis_format_from_annotation(expected_input_order)
            permute_order = AxisTracker.compute_permute_order(current_input_order, expected_input_order)
            if len(permute_order) and permute_order != list(range(len(permute_order))):
                graph.inject_implicit_permute(input_name, target_input_type,
                                              permute_order, [node.op.name])

            target_output_order = []
            output_buffers = graph.get_output_buffers(node)
            for output_buf in output_buffers:
                for dims in node.op.expected_output_axis_orders:
                    if len(dims) == output_buf.rank():
                        target_output_order = dims
                output_buf.axis_format = AxisTracker.get_axis_format_from_annotation(target_output_order)


@register_layer_optimization
class OptimizeUdoTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UdoOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeUdoTranslation, self).axes_to_spatial_first_order(node, graph)

        for i, buf in enumerate(graph.get_output_buffers(node)):
            node.op.output_dims[i] = buf.shape


@register_layer_optimization
class OptimizeUpsampleIndexBaseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeUpsampleSparseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UpsampleSparseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCropAndResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropAndResizeOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseBinaryMinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseBinaryMinOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeElementwiseBinaryMaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseBinaryMaxOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeEmbeddingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.EmbeddingOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractGlimpseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeImageProjectiveTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMomentTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MomentOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeNonMaxSuppresionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonMaxSuppresionOp.TRANSLATION_KEY
        self.register_method(ADJUST_NMS_FEATURE_DIMS, self.adjust_nms_feature_dimensions)

    @staticmethod
    def adjust_nms_feature_dimensions(graph):
        """
        By default nms requires 2 inputs for boxes and score whose input and output shape is handled in
        TF translation. With the extra input_features they do not typically come with batch dimensions, so handle
        here by verifying required second dimension equality with num_boxes
        TODO: remove once backend consolidate input/output shapes of features to MultiClassNms. This should be
        handled during TF translation similar to the boxes and scores input.
        """

        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            # adjustment of features only needed if features are given as inputs
            if len(nms_node_.input_names) > 2 and len(nms_node_.output_names) > 4 and \
                    "scale_y" not in nms_node_.op.attrs:
                return True
            return False

        sequence = [
            ("non_max_suppression",
             (),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_input_names = nms_node.input_names
            nms_output_names = nms_node.output_names
            num_boxes = graph.get_buffer(nms_node.input_names[0]).shape[1]
            for i in range(2, len(nms_node.input_names)):
                input_feature_buf = graph.get_buffer(nms_input_names[i])
                input_feature_shape = input_feature_buf.shape
                if len(input_feature_shape) == 1 or input_feature_shape[1] != num_boxes:
                    input_feature_node = graph.get_producer_node(nms_input_names[i])
                    # add reshape node to add batch dimension to the input features
                    expected_input_feature_shape = [1, *input_feature_shape]
                    # verify this is will result in expected input
                    log_assert(expected_input_feature_shape[1] == num_boxes,
                               "Unable to adjust input feature to match expected num_boxes on second dimension. "
                               "Got: {}, Expected num_boxes {}".format(expected_input_feature_shape, num_boxes))

                    if input_feature_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
                            graph.get_buffer(input_feature_node.input_names[0]).shape == expected_input_feature_shape:
                        # there was a squeeze done to remove batch dim, remove it and adjust to expected
                        # input feature instead.
                        graph.squash(input_feature_node, input_feature_node.input_names[0])
                        graph.get_buffer(input_feature_node.output_names[0]).set_buf_dims(expected_input_feature_shape)
                    else:
                        # add the reshape to add batch dim
                        input_feature_reshape_node_name = nms_input_names[i] + "_reshape_batch_add"
                        input_feature_reshape_op = op_adapter.ReshapeOp(name=input_feature_reshape_node_name,
                                                                        output_shape=expected_input_feature_shape)
                        graph.inject(input_feature_reshape_op, input_name=nms_input_names[i],
                                     output_name=input_feature_reshape_node_name,
                                     consumer_names=[nms_node.op.name])

                    # since we are reshaping input, output from nms will need to be adjusted as intermediate and
                    # will require a post reshape to remove batch dimension added.
                    output_name_idx = i + 2  # accounting for class and num_det output
                    output_feature_name = nms_output_names[output_name_idx]
                    output_feature_buf = graph.get_buffer(output_feature_name)
                    output_feature_reshape_node_name = output_feature_name + "_reshape_batch_remove"
                    output_feature_reshape_op = op_adapter.ReshapeOp(name=output_feature_reshape_node_name,
                                                                     output_shape=output_feature_buf.shape)
                    # adjust to expected buffer shape for nms feature output(i.e with batch dim)
                    output_feature_buf.set_buf_dims([1, *output_feature_buf.shape])
                    graph.inject(output_feature_reshape_op, input_name=output_feature_name,
                                 output_name=output_feature_reshape_node_name)

                    # Addition of a const tensor to features should not be quantized
                    # TODO: add conditional that it should be set non quantizable based on tensortype and
                    #       quantization info of input tensor when irgraph supports these info
                    output_feature_reshape_buf = graph.get_buffer(output_feature_reshape_node_name)
                    for consumer in output_feature_reshape_buf.consumers:
                        if isinstance(consumer.op, op_adapter.ElementwiseOp) or \
                                isinstance(consumer.op, op_adapter.ElementwiseBinaryOp):
                            for input_name in consumer.input_names:
                                eltwise_input_node = graph.get_producer_node(input_name)
                                if eltwise_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                                    eltwise_input_node.op.quantizable = False


@register_layer_optimization
class OptimizePackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PackOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizePixelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PixelShuffleOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeReduceProdTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReduceProdOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeStridedSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeStridedSliceTranslation, self).axes_to_spatial_first_order(node, graph)
        # begin, end and strides need to reorder to follow axis format
        node.op.begin = graph.src_axis_order.permute_shape_to_ir(node.op.begin)
        node.op.end = graph.src_axis_order.permute_shape_to_ir(node.op.end)
        node.op.strides = graph.src_axis_order.permute_shape_to_ir(node.op.strides)


@register_layer_optimization
class OptimizeSpaceToDepthTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SpaceToDepthOp.TRANSLATION_KEY
        self.register_method(MATCH_SPACETODEPTH, self.match_spacetodepth)

    @staticmethod
    def match_spacetodepth(graph):
        # check shapes in the reshape layers
        # [n, c, h, w] -> [n, c * blk**2, h/blk, w/blk]
        def is_valid_spacetodepth(node_tuple):
            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            input_shape = input_buf.shape
            first_reshape_output_shape = node_tuple[0].op.output_shape
            if len(input_shape) == 4 and len(first_reshape_output_shape) == 6:
                blocksize = first_reshape_output_shape[3]
                sequence_output_shape = node_tuple[-1].op.output_shape

                batch, height, width, depth = graph.src_axis_order.extract_spatial_dims(input_shape)
                expected_shape = graph.src_axis_order.format_spatial_output_shape(batch_size=batch,
                                                                                  depth=depth * (blocksize**2),
                                                                                  height=height//blocksize,
                                                                                  width=width//blocksize)
                return sequence_output_shape == expected_shape
            else:
                return False

        # reshape:   [n, c, h/blk1, blk1, w/blk2, blk2], blk1 == blk2, number is for transpose order.
        # transpose: [n, c, h/blk1, w/blk2, blk1, blk2]
        # reshape:   [n, c, h/blk * w/blk, blk ** 2]
        # transpose: [n, c, blk ** 2, h/blk * w/blk]
        # reshape:   [n, c, blk ** 2, h/blk, w/blk]
        # transpose: [n, blk ** 2, c, h/blk, w/blk]
        # reshape:   [n, c*(blk**2), h/blk, w/blk]
        sequence = [
            ("reshape",
             (),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")])
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ("MATCH_NUM_BUFS", [("permute", "ALL")])
            ),
            ("permute",
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("reshape", "ALL")]),
            ),
            ("reshape",
             ("MATCH_NUM_BUFS", [("permute", "ALL")]),
             ()
            )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_spacetodepth, ignore_constants=True)
        for node_tuple in matched_node_list:
            blocksize = node_tuple[0].op.output_shape[3]
            reshape_node = node_tuple[0]
            # Squash all nodes except the first reshape in reverse order
            # the first reshape op will be replaced
            for node in node_tuple[:0:-1]:
                for input_name in node.input_names:
                    graph.squash(node, input_name)
            reshape_op = reshape_node.op
            reshape_op_name = graph.naming_policy.get_op_name(reshape_op)
            spacetodepth_op_name = reshape_op_name + '_space_to_depth'
            spacetodepth_op = op_adapter.SpaceToDepthOp(spacetodepth_op_name, downscale_factor=blocksize)
            graph.replace(reshape_op, spacetodepth_op)


@register_layer_optimization
class OptimizeSsdTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SsdOp.TRANSLATION_KEY
        self.register_method(SQUASH_BOX_DECODER, self.squash_box_decoder)

    @staticmethod
    def squash_box_decoder(graph):
        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            nms_input_names_ = nms_node_.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_input_names_[0]).type:
                # remove optional reshape input to check if previous is box decoder(ssd) below
                reshape_node_ = graph.get_producer_node(nms_node_.input_names[0])
                nms_input_names_ = [nms_input_names_[1], *reshape_node_.input_names]

            if any(op_adapter.SsdOp.TRANSLATION_KEY == graph.get_producer_node(name_).op.TRANSLATION_KEY
                   for name_ in nms_input_names_):
                return True

            return False

        sequence = [
                    ("non_max_suppression",
                        (),
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_op = nms_node.op
            # update the boxes input of nms to be box decoder's inputs along with box decoder's op attributes.
            #  [boxes]_______[anchor or priorboxes]
            #            |
            #       [box_decoder(ssd_op)]   <- remove
            #                  |
            #        remove->([Reshape] (optional))_______[scores]
            #                                         |
            #                                 [non_max_suppression]
            # Updated input for nms will be: [scores, boxes, anchor(priorboxes)]

            nms_boxes_input_name, nms_scores_input_name = nms_node.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_boxes_input_name).type:
                # update inputs for nms and subsequently the boxes_node
                reshape_node = graph.get_producer_node(nms_boxes_input_name)
                reshape_buf = graph.get_buffer(nms_boxes_input_name)
                nms_boxes_input_name = reshape_node.input_names[0]

                # update consumer relation with reshape buf and prune if applicable
                reshape_buf.consumers.remove(nms_node)
                if len(reshape_buf.consumers) == 0:
                    graph.prune(reshape_node)

            # fold box_decoder(ssd) node
            box_decoder_node = graph.get_producer_node(nms_boxes_input_name)
            box_decoder_buf = graph.get_buffer(nms_boxes_input_name)
            # Copy over input_names and all op attrs to nms op
            nms_node.input_names = [nms_scores_input_name, *box_decoder_node.input_names]
            for key, val in box_decoder_node.op.attrs.items():
                nms_op[key] = val

            # update consumer relation with nms node, box_decoder node and input to box_decoder and
            # prune if applicable
            for name in box_decoder_node.input_names:
                buf = graph.get_buffer(name)
                buf.consumers.add(nms_node)
            if nms_node in box_decoder_buf.consumers:
                box_decoder_buf.consumers.remove(nms_node)
            if len(box_decoder_buf.consumers) == 0:
                graph.prune(box_decoder_node)

            # Update Anchors inputs to fit DetectionOut spec
            anchor_buf = graph.get_buffer(nms_node.input_names[-1])
            anchor_data = anchor_buf.producer.op.tensor

            # TF style (decodeBox+nms) comes as CORNER_SIZE spec requires CENTER_SIZE
            for i in range(0, anchor_buf.shape[1]):
                y_min, x_min, y_max, x_max = anchor_data[0][i]
                height = (y_max - y_min)
                width = (x_max - x_min)
                anchor_data[0][i][0] = y_min + height / 2.  # center_y
                anchor_data[0][i][1] = x_min + width / 2.  # center_x
                anchor_data[0][i][2] = height  # height
                anchor_data[0][i][3] = width

            # Addition of a const tensor to class labels should not be quantized
            classes_buf = graph.get_buffer(nms_node.output_names[2])
            for consumer in classes_buf.consumers:
                if consumer.op.type == op_adapter.ElementwiseSumOp.TRANSLATION_KEY:
                    for input_name in consumer.input_names:
                        add_input_node = graph.get_producer_node(input_name)
                        if add_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            add_input_node.op.quantizable = False

            # change shape for anchor input from [batch, num_anchors, 4] to [batch * num_anchors, 4] per spec
            anchor_buf.shape = [anchor_buf.shape[0] * anchor_buf.shape[1], anchor_buf.shape[2]]
            anchor_buf.producer.op.tensor = anchor_data.reshape(anchor_buf.shape)

            log_debug2(code_to_message.get_debugging_message("DEBUG_BOXDECODER_SQUASH")(box_decoder_node.op.name,
                                                                                        nms_node.op.name))


@register_layer_optimization
class OptimizeTileTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TileOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.eltwise_to_spatial_first_order(node, graph)
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in format_to_permute_order:
            node.op.multiples = graph.src_axis_order.permute_shape_to_ir(node.op.multiples)


@register_layer_optimization
class OptimizeTopKTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TopKOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        super(OptimizeTopKTranslation, self).axes_to_spatial_first_order(node, graph)

        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in format_to_permute_order:
            axis_map = format_to_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]


@register_layer_optimization
class OptimizeUnpackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UnpackOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeUpsampleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Upsample.TRANSLATION_KEY
