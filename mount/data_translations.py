# ==============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class OnnxChannelShuffleTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        # Note: Schema is not used here since this is not a valid Onnx Op
        params = extract_attributes(src_op,
                                    ('groups', 'i'))
        return op_adapter.ChannelShuffleOp(src_op.name, groups=params.groups)


OnnxTranslations.register_translation(OnnxChannelShuffleTranslation(),
                                      converter_type('Channel_Shuffle', 'onnx'),
                                      op_adapter.ChannelShuffleOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Clip', [1, 6, 11, 12])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        min_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        if min_name and not graph.has_buffer(min_name):
            min_val = graph.weights.fetch(min_name, prunable=True)
        else:
            min_val = params.min if 'min' in params else -1 * numpy.inf

        max_name = str(src_op.input[2]) if len(src_op.input) > 2 else ''
        if max_name and not graph.has_buffer(max_name):
            max_val = graph.weights.fetch(max_name, prunable=True)
        else:
            max_val = params.max if 'max' in params else numpy.inf
        return op_adapter.NeuronOp(src_op.name,
                                   "NEURON_RELU_MIN_MAX",
                                   min_clamp=min_val,
                                   max_clamp=max_val)

    def extract_input_names(self, src_op, graph):
        return [list(src_op.input)[0]]


OnnxTranslations.register_translation(OnnxClipTranslation(), converter_type('Clip', 'onnx'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Concat', [1, 4, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())

        # static concatenation used for reshaping shape tensors
        if graph.weights.has_all(src_op.input):
            data = [graph.weights.fetch(input_name) for input_name in src_op.input]
            concat_data = numpy.concatenate(data, params.axis)
            graph.weights.insert(str(src_op.output[0]), concat_data)
            return op_adapter.StaticOp(src_op.name)

        # handle single input concats
        if len(src_op.input) == 1:
            if graph.weights.has_all(src_op.input):
                graph.weights.insert(str(src_op.output[0]), graph.weights.fetch(src_op.input[0]))
                return op_adapter.StaticOp(src_op.name)
            return op_adapter.NoopOp(src_op.name)

        return op_adapter.ConcatOp(src_op.name, params.axis)

    def infer_output_shapes(self, op, input_shapes):
        # Add batch dim
        axis = op.axis
        output_shape = input_shapes[0][:]
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        return [output_shape]

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has_all(src_op.input):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxConcatTranslation(),
                                      converter_type('Concat', 'onnx'),
                                      op_adapter.ConcatOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Constant
# ------------------------------------------------------------------------------
class OnnxConstantTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Constant', [1, 9])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        # ONNX return numpy "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        if not params.value.shape:
            params.value = params.value.reshape(1)

        graph.weights.insert(src_op.output[0], params.value)
        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(src_op.output[0], params.value)

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxConstantTranslation(),
                                      converter_type('Constant', 'onnx'),
                                      op_adapter.ConstantOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Initializer
# ------------------------------------------------------------------------------
class OnnxInitializerTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, initializer, graph):
        params = extract_initializer_tensor(initializer)

        # ONNX return numpy "array scalar" for ONNX scalar.
        # the problem is, "array scalar" has shape attribute as an empty tuple.
        # which may break backends.
        # So we reshape "array scalar" to exactly an array with shape (1, )
        if not params.shape:
            params = params.reshape(1)

        # Constant op is a special case... the output name is the real name
        return op_adapter.ConstantOp(initializer.name, params)

    def extract_input_names(self, src_op, graph):
        return []

    def extract_output_names(self, src_op, graph):
        return [src_op.name]

    def infer_output_shapes(self, op, input_shapes):
        return [list(op.tensor.shape)]


OnnxTranslations.register_translation(OnnxInitializerTranslation(),
                                      converter_type('Initializer', 'onnx'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Flatten', [1, 9, 11])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = params.axis

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            pre_axes = w.shape[:axis]
            post_axes = w.shape[axis:]
            output_shape = [product(pre_axes), product(post_axes)]
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))
            return op_adapter.StaticOp(src_op.name)

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [product(pre_axes), product(post_axes)]

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxFlattenTranslation(), converter_type('Flatten', 'onnx'))


# ------------------------------------------------------------------------------
#   Gather
# ------------------------------------------------------------------------------
class OnnxGatherTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gather', [1])

    def add_op(self, src_op, graph):
        ops = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        if len(ops) > 1:
            indices_node = graph.add(ops[0], [], input_names[1])
            # add src_op info for added constant op
            #  calling graph function directly as add_src_op_info is intended for ONNX ops
            graph.add_src_op_info(indices_node.op.name, None, indices_node.output_names[0])
        last_node = graph.add(ops[-1], input_names, output_names)

        # add src op info for gather operation
        self.add_src_op_info(last_node.op.name, src_op, graph)

        return last_node

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        ops = []
        indices_name = str(src_op.input[1])
        # If the input is stored as weights we need to create a const node
        if not graph.has_buffer(indices_name):
            indices = graph.weights.fetch(indices_name, prunable=False)
            ops.append(op_adapter.ConstantOp(indices_name, indices, quantizable=False))
        else:
            indices_op = graph.get_buffer(indices_name).producer.op
            if op_adapter.ConstantOp.TRANSLATION_KEY is indices_op.type:
                indices_op.quantizable = False
        ops.append(op_adapter.GatherOp(src_op.name, axis=params.axis))
        return ops

    def infer_output_shapes(self, op, input_shapes):
        output_shape = input_shapes[0][:op.axis] + list(input_shapes[1]) + input_shapes[0][op.axis + 1:]
        return [output_shape]


OnnxTranslations.register_translation(OnnxGatherTranslation(),
                                      converter_type('Gather', 'onnx'),
                                      op_adapter.GatherOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Pad
# ------------------------------------------------------------------------------
class OnnxPadTranslation(OnnxTranslationBase):
    supported_modes = {'constant': 'PADDING_CONSTANT',
                       'reflect': 'PADDING_REFLECT',
                       'edge': 'PADDING_EDGE'}

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pad', [1, 2, 11])\
            .register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        pads_name = str(src_op.input[1]) if len(src_op.input) > 1 else ''
        if pads_name and not graph.has_buffer(pads_name):
            pads = graph.weights.fetch(pads_name, prunable=True)
        else:
            if 'pads' in params:
                pads = params.pads
            else:
                pads = params.paddings

        # Pads/paddings need to be translated from r1_begin, r2_begin...r1_end, r2_end, ...
        # to pairs (r1_begin, r1_end), (r2_begin, r2_end)...
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)
        log_assert(rank == len(pads) / 2,
                   "Rank of input tensor: %d must equal (# pads/2): %d",
                   rank,
                   len(pads) / 2)

        pad_pairs = []
        for index in range(rank):
            pad_pairs.append([pads[index], pads[index + rank]])
        return op_adapter.PadOp(src_op.name,
                                mode=self.supported_modes[params.mode],
                                pads=pad_pairs,
                                constant_value=params.value if 'value' in params else 0)

    def extract_input_names(self, src_op, graph):
        # Filter if there are any parameters like 'pads' in inputs
        # For exampled 'pads' are already handled in extract_paramters
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        output_shape = []

        for i in range(0, len(input_shape)):
            output_shape.append(input_shape[i] + op.pads[i][0] + op.pads[i][1])

        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, output_shape))
        return [output_shape]

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in OnnxPadTranslation.supported_modes:
                raise ValueError(code_to_message.get_error_message("ERROR_PAD_UNSUPPORTED_MODE")(src_op_mode))


OnnxTranslations.register_translation(OnnxPadTranslation(),
                                      converter_type('Pad', 'onnx'),
                                      op_adapter.PadOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Reshape', [1, 5])

    def extract_parameters(self, src_op, graph):
        # There are two main versions of ONNX Reshape
        #    1. The old reshape, where shape is provided as an attribute
        #    2. The new reshape, where the shape is provided as a second input
        #
        # SNPE and the converter support two versions of Reshape:
        #    1. Dynamic reshaping with a statically provided output shape
        #    2. Static reshaping, performed at conversion time
        #
        # SNPE can't support the 2nd ONNX Reshape expclicitly, however we can
        # calculate the shape ahead of time and statically set in in the SNPE layer.
        # This will prevent the network from being resizable. In addition, if a
        # 'Shape' layer provided the shape it will have been saved as static,
        # eg weight data, in the converter and all ops operating on that data will
        # become static ops and will be pruned during the final conversion.
        shape = []
        if len(src_op.input) > 1:
            #import pdb; pdb.set_trace()
            #shape_input = str(src_op.input[1])

            #print(graph.get_buffer(str(src_op.input[0])).shape)

            # Retrieve number of inputs
            reshape_number_of_inputs = graph.get_buffer(str(src_op.input[1])).shape[0]
            if reshape_number_of_inputs == 6:
                # Retrieving Conv shape
                batch_size, channels, h, w = graph.get_buffer(str(src_op.input[0])).shape
                channels /= 4
                shape = [batch_size, channels, 2, 2, h, w]
            else:
                # Retrieving Conv shape
                batch_size, channels, _,_, h, w = graph.get_buffer(str(src_op.input[0])).shape
                shape = [batch_size, channels, 2*h, 2*w]

            print(shape)

            # if graph.weights.has(shape_input):
            #     shape = graph.weights.fetch(shape_input).astype(numpy.int64).tolist()
            # else:
            #     shape = graph.get_buffer(str(src_op.input[1])).shape.tolist()
        else:
            params = extract_attributes(src_op, schema=self.op_schema())
            if 'shape' in params:
                shape = params.shape

        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)
        else:
            # dynamic reshape of activations
            input_buf = graph.get_buffer(input_name)
            input_shape = input_buf.shape

            remainder_size = product(input_shape)
            remainder_index = -1
            output_shape = []
            for i, s in enumerate(shape):
                if s == -1:
                    remainder_index = i
                    output_shape.append(0)
                elif s == 0:
                    remainder_size /= input_shape[i]
                    output_shape.append(input_shape[i])
                else:
                    remainder_size /= s
                    output_shape.append(s)
            if remainder_index >= 0:
                output_shape[remainder_index] = remainder_size

            return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxReshapeTranslation(),
                                      converter_type('Reshape', 'onnx'),
                                      op_adapter.ReshapeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class OnnxResizeTranslation(OnnxTranslationBase):
    SUPPORTED_RESIZE_MODES = ['nearest', 'linear', 'bilinear']
    SUPPORTED_COORD_TRANSFORM_MODES = ['asymmetric', 'align_corners', 'half_pixel', 'tf_half_pixel_for_nn',
                                       'pytorch_half_pixel']

    def __init__(self):
        OnnxTranslationBase.__init__(self)
        schema_dict = self.register_op_schema('Resize', [10, 11])
        schema_dict.replace_default_values(mode='nearest')
        schema_dict.register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        resize_schema = self.op_schema()
        params = extract_attributes(src_op, attr_infos=[('mode', 's', 'nearest'),
                                                        ('coordinate_transformation_mode', 's', 'asymmetric')],
                                    schema=resize_schema, validate=True)
        transformation_mode = params.coordinate_transformation_mode
        input_buf = graph.get_buffer(str(src_op.input[0]))
        if input_buf.rank() != 4:
            raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_INPUT_DIMS")(input_buf.shape))

        align_corners = False
        half_pixel_centers = False
        # Determine coordinate transform mode include in Resize version >10
        if transformation_mode == "align_corners":
            align_corners = True
        elif (params.mode == "linear" and transformation_mode == "half_pixel") or \
                (params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn"):
            half_pixel_centers = True
        elif (not params.mode == "linear" and transformation_mode == "half_pixel") or \
                (not params.mode == "nearest" and transformation_mode == "tf_half_pixel_for_nn"):
            raise ValueError(
                code_to_message.get_error_message("ERROR_RESIZE_INVALID_COORDINATE_TRANSFORMATION_MODE_MIX")
                (params.mode, transformation_mode))

        sizes = None
        input_shape = input_buf.shape
        input_height = input_shape[2]
        input_width = input_shape[3]
        if len(src_op.input) > 1:
            scales = None
            scales_input = str(src_op.input[2]) if len(src_op.input) > 2 else str(src_op.input[1])
            if graph.weights.has(scales_input):
                scales = graph.weights.fetch(scales_input).astype(numpy.float32).tolist()
            elif graph.has_buffer(scales_input):
                scales = graph.get_buffer(scales_input).shape.tolist()
            if not scales and len(src_op.input) > 2:
                size_name = str(src_op.input[-1])
                if graph.weights.has(size_name):
                    # per onnx spec, size is int64. But int32 may be enough.
                    sizes = graph.weights.fetch(size_name).astype(numpy.int32).tolist()
                else:
                    sizes = graph.get_buffer(size_name).shape.tolist()
                # Opset 11 has 4th parameter as output sizes,
                # here we are calculating scales from output sizes
                # per onnx spec, scales is float.
                scales = list(map(float, sizes))
                scales[-1] = (scales[-1]-1) / (input_width-1) if align_corners else (scales[-1] / input_width)
                scales[-2] = (scales[-2]-1) / (input_height-1) if align_corners else (scales[-2] / input_width)
        else:
            # deprecated. Added for Upsample version 7 and below
            scales = extract_attributes(src_op, attr_infos=[('scales', 'lf')], schema=resize_schema, validate=True).scales

        scale_height = scales[2]
        scale_width = scales[3]

        # Generate output shape using output_dims. Note: doing round() first since casting to int gets the floor
        # which was causing output dim error in models.
        output_height = sizes[2] if sizes else int(round(input_height * scale_height))
        output_width = sizes[3] if sizes else int(round(input_width * scale_width))
        output_shape = [input_shape[0], input_shape[1], output_height, output_width]

        # per onnx spec pytorch_half_pixel is same as half_pixel when length resized for dimension > 1.
        if transformation_mode == "pytorch_half_pixel":
            if output_height > 1 and output_width > 1:
                half_pixel_centers = True
            else:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_RESIZE_PYTORCH_HALF_PIXEL_UNSUPPORTED_VALUE")
                    (transformation_mode, output_height, output_width))

        return op_adapter.ResizeOp(src_op.name,
                                   output_shape,
                                   resize_mode=params.mode,
                                   scale_height=scale_height,
                                   scale_width=scale_width,
                                   align_corners=align_corners,
                                   half_pixel_centers=half_pixel_centers)

    @classmethod
    def validate_attribute_values(cls, src_op, attr_name, attr_value):
        if attr_name == 'mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_RESIZE_MODES:
                raise ValueError(code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_MODE")
                                 (src_op_mode,  cls.SUPPORTED_RESIZE_MODES))
        elif attr_name == 'scales':
            scales = attr_value
            if scales[0] != 1 or scales[1] != 1:
                log_warning(code_to_message.get_warning_message("WARNING_RESIZE"))
        elif attr_name == 'coordinate_transformation_mode':
            src_op_mode = attr_value
            if src_op_mode not in cls.SUPPORTED_COORD_TRANSFORM_MODES:
                raise ValueError(
                    code_to_message.get_error_message("ERROR_RESIZE_UNSUPPORTED_COORDINATE_TRANSFORMATION_MODE")
                    (src_op_mode, cls.SUPPORTED_COORD_TRANSFORM_MODES))

    def extract_input_names(self, src_op, graph):
        if len(src_op.input) > 2:
            return [str(src_op.input[0])]
        else:
            return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def infer_output_shapes(self, op, input_shapes):
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(op.name, op.output_shape))
        return [op.output_shape]


OnnxTranslations.register_translation(OnnxResizeTranslation(),
                                      converter_type('Resize', 'onnx'),
                                      op_adapter.ResizeOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Shape
# ------------------------------------------------------------------------------
class OnnxShapeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Shape', [1])

    def extract_parameters(self, src_op, graph):
        log_warning(code_to_message.get_warning_message("WARNING_STATIC_SHAPE")(src_op.name))
        shape = graph.get_buffer(str(src_op.input[0])).shape
        output_name = str(src_op.output[0])
        graph.weights.insert(output_name, numpy.asarray(shape, dtype=numpy.int64))
        shape_param = numpy.asarray(shape, dtype=numpy.int32)
        return op_adapter.ConstantOp(output_name, shape_param, quantizable=False)

    def extract_input_names(self, src_op, graph):
        return []

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]



OnnxTranslations.register_translation(OnnxShapeTranslation(),
                                      converter_type('Shape', 'onnx'))


# ------------------------------------------------------------------------------
#   Slice, Crop
# ------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Slice', [1, 10, 11])

    def extract_parameters(self, src_op, graph):
        input_names = [str(x) for x in src_op.input]
        params = extract_attributes(src_op, schema=self.op_schema())
        const_inputs_params = self._fetch_inputs_as_params(src_op, graph, params)
        params.update(const_inputs_params)

        log_assert(len(params.starts) == len(params.axes),
                   "Node {}: expected same number of starts as axes",
                   src_op.name)
        log_assert(len(params.ends) == len(params.axes),
                   "Node {}: expected same number of ends as axes",
                   src_op.name)

        # Static slicing used for shape tensors
        if graph.weights.has(input_names[0]):
            data = graph.weights.fetch(input_names[0])
            for i in range(len(params.axes)):
                start, end = self.get_indices(params.starts[i], params.ends[i], data.shape[params.axes[i]])
                data = data.take(indices=list(range(start, end, params.steps[i])), axis=params.axes[i])
            output_name = str(src_op.output[0])
            graph.weights.insert(output_name, data)
            return op_adapter.StaticOp(src_op.name)

        input_buf = graph.get_buffer(input_names[0])
        rank = input_buf.rank()
        begin = [0] * rank
        end = [0] * rank
        strides = [0] * rank

        for index, axis in enumerate(params.axes):
            begin[axis], end[axis] = self.get_indices(params.starts[index], params.ends[index], input_buf.shape[axis])
            strides[axis] = params.steps[index]

        for i, stride in enumerate(strides):
            if not stride:
                begin[i], end[i] = 0, input_buf.shape[i]
                strides[i] = 1

        return op_adapter.StridedSliceOp(name=src_op.name,
                                         begin=begin,
                                         end=end,
                                         strides=strides)

    def _fetch_inputs_as_params(self, src_op, graph, params):
        # opset 10,11 need handle 5 inputs, fetch constant input and add it to params
        # NOTE: Runtime does not allow dynamic input for starts, ends, axes and steps
        # input indices: data: 0, starts: 1, ends: 2, axes: 3(optional), steps: 4(optional)
        input_names = [str(x) for x in src_op.input]
        rank = 0
        if graph.has_buffer(input_names[0]):
            input_buf = graph.get_buffer(input_names[0])
            rank = input_buf.rank()
        elif graph.weights.has(input_names[0]):
            rank = len(graph.weights.fetch(input_names[0], prunable=False).shape)
        keys = ['data', 'starts', 'ends', 'axes', 'steps']
        if len(src_op.input) >= 3:
            for key, name in zip(keys[1:], input_names[1:]):
                # ONNX may use empty string as a placeholder
                # So add an and-condition to further check it.
                if name and graph.weights.has(name):
                    # handle INT_MAX and INT_MIN case in ONNX spec, require fetch int64 directly
                    # case: INT64_MAX -> cast to float and cast to int64 -> INT64_MIN
                    # case: INT64_MAX -> cast to int32 -> -1
                    params[key] = graph.weights.fetch(name, dtype=numpy.int64).tolist()
                    if key == 'axes':
                        for axis in params['axes']:
                            log_assert(-rank <= axis <= rank-1,
                            "expected axis range from {} to {}, but got {}",
                            -rank, rank-1, axis)
                elif graph.has_buffer(name):
                    raise ValueError(code_to_message.get_error_message('ERROR_SLICE_DYNAMIC_INPUTS')(name))

        if 'axes' not in params or len(params.axes) == 0:
            params['axes'] = list(range(len(params.starts)))
        if 'steps' not in params or len(params.steps) == 0:
            params['steps'] = list([1] * len(params.starts))

        return params

    def extract_input_names(self, src_op, graph):
        # If this was translated to a static op don't return input names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            # Handle constant and initializer cases, do not add them to input_names to avoid prune error.
            actual_input_names = []
            for input_name in map(str, src_op.input):
                if input_name in graph.buffers and not graph.weights.has(input_name):
                    actual_input_names.append(input_name)
            return actual_input_names

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.has(str(src_op.input[0])):
            return []
        else:
            return list(map(str, src_op.output))

    def infer_output_shapes(self, op, input_shapes):
        if isinstance(op.output_shape[0], list):
            return op.output_shape
        return [op.output_shape]

    @staticmethod
    def get_indices(start, end, dim):
        # Negative values mean wrap around, like in python
        if start < 0:
            start = int(start % dim)
        if end < 0:
            end = int(end % dim)
        # higher than the size, however, means stop at the end.
        start = min(start, dim)
        end = min(end, dim)
        return start, end


# Onnx Crop should go here as well, but the documentation is really
# ambiguous so we won't add it until we see an example.
OnnxTranslations.register_translation(OnnxSliceTranslation(),
                                      converter_type('Slice', 'onnx'),
                                      op_adapter.CropOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Split', [1, 2, 11])\
            .replace_default_values(split=[])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_buf = graph.get_buffer(str(src_op.input[0]))
        output_shape = [input_buf.shape[:]]*len(src_op.output)
        if not params.split:
            number_of_splits = len(src_op.output)
            input_buffer_shape_along_given_axis = input_buf.shape[params.axis]
            if input_buffer_shape_along_given_axis % number_of_splits != 0:
                raise Exception('Input shape along the specified axis is not a multiple of '
                                'the number of splits specified')
            params.split = [input_buffer_shape_along_given_axis // number_of_splits] * number_of_splits

        slice_points = []
        next_slice_point = 0
        for split in params.split[1:]:
            next_slice_point += split
            slice_points.append(next_slice_point)

        for i, shape in enumerate(output_shape):
            shape[params.axis] = params.split[i]
            output_shape[i] = shape

        return op_adapter.SliceOp(src_op.name,
                                  axis=params.axis,
                                  slice_points=slice_points,
                                  output_shape=output_shape)

    def infer_output_shapes(self, op, input_shapes):
        return op.output_shape


OnnxTranslations.register_translation(OnnxSplitTranslation(),
                                      converter_type('Split', 'onnx'),
                                      op_adapter.SliceOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class OnnxSqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Squeeze', [1, 11])

    def extract_parameters(self, src_op, graph):
        input_name = str(src_op.input[0])
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]
        default_axes = [i for i, s in enumerate(input_shape) if s == 1]

        params = extract_attributes(src_op, schema=self.op_schema())

        if not params.axes or 'axes' not in params:
            params['axes'] = default_axes

        if not all(x < len(input_shape) for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIM_GREATER_THAN_RANK")(params.axes,
                                                                                                      len(input_shape)))

        if not all((input_shape[x] == 1) for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_SQUEEZE_DIMS_EQUAL_ONE")(params.axes,
                                                                                               input_shape))

        output_shape = [s for i, s in enumerate(input_shape) if i not in params.axes]

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))

            w = graph.weights.fetch(input_name)
            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # Otherwise this is a dynamic flatten so add the flatten/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxSqueezeTranslation(), converter_type('Squeeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class OnnxTileTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tile', [1, 6])

    def extract_parameters(self, src_op, graph):
        input_names = list(map(str, src_op.input))
        input_rank = len(graph.get_buffer(src_op.input[0]).shape)

        if len(input_names) == 3:
            # Represents Tile-1
            tiles = graph.weights.fetch(src_op.input[1])
            axis = graph.weights.fetch(src_op.input[2])
            repeats = [1] * input_rank
            repeats[axis] = tiles
        elif len(input_names) == 2:
            # Represents Tile-6
            repeats = graph.weights.fetch(src_op.input[1]).astype(numpy.uint32).tolist()
        else:
            raise ValueError("Only versions {} of {} node {} are supported".format(self.get_supported_version(),
                                                                                   src_op.op_type, src_op.name))

        return op_adapter.TileOp(src_op.name, multiples=repeats)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTileTranslation(),
                                      converter_type('Tile', 'onnx'),
                                      op_adapter.TileOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Transpose', [1])

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, schema=self.op_schema())
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            w = numpy.transpose(w, params.perm)
            graph.weights.insert(output_name, w)
            log_info(code_to_message.get_progress_message("INFO_STATIC_RESHAPE")(input_name, output_name, w.shape))

            return op_adapter.StaticOp(src_op.name)

        return op_adapter.PermuteOp(src_op.name, params.perm)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        # return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    def infer_output_shapes(self, op, input_shapes):
        output_shape = [input_shapes[0][i] for i in op.order]
        return [output_shape]


OnnxTranslations.register_translation(OnnxTransposeTranslation(),
                                      converter_type('Transpose', 'onnx'),
                                      op_adapter.PermuteOp.TRANSLATION_KEY)


# -----------------------------------------------------------------------------
#   Unsqueeze
# ------------------------------------------------------------------------------
class OnnxUnsqueezeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Unsqueeze', [1, 11])\
            .register_method(self.validate_attribute_values)

    def extract_parameters(self, src_op, graph):
        # default_axes = [i for i, s in enumerate(input_shape) if s == 1]
        params = extract_attributes(src_op, schema=self.op_schema(), validate=True)

        input_name = str(src_op.input[0])

        # SNPE uses weights at construction time, not dynamically. Ensure they
        # are preprocessed statically.
        if graph.weights.has(input_name):
            # static flatten of weight parameters
            output_name = str(src_op.output[0])
            w = graph.weights.fetch(input_name)
            output_shape = self._get_unsqueezed_shape(w.shape, params.axes)

            log_info(code_to_message.get_progress_message(
                "INFO_STATIC_RESHAPE")(input_name, output_name, output_shape))

            w = numpy.reshape(w, output_shape)
            graph.weights.insert(output_name, w)
            return op_adapter.StaticOp(src_op.name)

        # input is not a static parameter
        input_buf = graph.get_buffer(input_name)
        input_shape = input_buf.shape[:]

        new_rank = len(input_shape) + len(params.axes)
        if not all(x < new_rank for x in params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DIMS_GREATER_THAN_RANK")(params.axes,
                                                                                                         new_rank))

        if len(set(params.axes)) != len(params.axes):
            raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_DUPLICATE_DIMS")(params.axes))

        output_shape = self._get_unsqueezed_shape(input_shape, params.axes)

        # Otherwise this is a dynamic unsqueeze so add the unsqueeze/reshape op
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    def extract_input_names(self, src_op, graph):
        return [name for name in list(map(str, src_op.input)) if not graph.weights.consumed(name)]

    def extract_output_names(self, src_op, graph):
        # If this was translated to a static op don't return output names
        if graph.weights.consumed(str(src_op.input[0])):
            return []
        else:
            return [str(src_op.output[0])]

    @staticmethod
    def validate_attribute_values(src_op, attr_name='', attr_value=None):
        if attr_name == 'axes':
            if not all(x >= 0 for x in attr_value):
                raise ValueError(code_to_message.get_error_message("ERROR_UNSQUEEZE_NEGATIVE_DIMS")(attr_value))

    @staticmethod
    def _get_unsqueezed_shape(org_shape, axes):
        output_shape = list(org_shape)
        for i in sorted(axes):
            output_shape.insert(i, 1)
        return output_shape


OnnxTranslations.register_translation(OnnxUnsqueezeTranslation(), converter_type('Unsqueeze', 'onnx'))


# ------------------------------------------------------------------------------
#   Upsample
# ------------------------------------------------------------------------------
class OnnxUpsampleTranslation(OnnxResizeTranslation):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Upsample', [1, 7, 9])\
            .register_method(self.validate_attribute_values)


OnnxTranslations.register_translation(OnnxUpsampleTranslation(),
                                      converter_type('Upsample', 'onnx'),
                                      op_adapter.Upsample.TRANSLATION_KEY)
