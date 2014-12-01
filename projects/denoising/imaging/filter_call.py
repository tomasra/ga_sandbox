import itertools as it
import copy
from projects.denoising.imaging.filters import one_argument_filters
from projects.denoising.imaging.filters import two_argument_filters


class FilterCall(object):
    """
    Wraps filter call on specified image channels
    (represented as list indexes)
    """
    def __init__(
            self, filter_function,
            src_channel_indexes, dest_channel_index):
        self.name = filter_function.func_name
        self.filter_function = filter_function
        # Source channel indexes in image
        self.src_channel_indexes = src_channel_indexes
        # Destination channel index
        self.dest_channel_index = dest_channel_index

    def __call__(self, image, overwrite=True):
        """
        Runs filter function on specified image source channels
        and writes result into destination channel.
        Can either return new image or overwrite existing one
        """
        # positional filter arguments
        input_channels = [
            image.channels[i]
            for i in self.src_channel_indexes
        ]
        result_channel = self.filter_function(*input_channels)
        if overwrite is True:
            # Put results into existing image channel
            image.channels[self.dest_channel_index] = result_channel
            return image
        else:
            # Create and return new image
            new_image = copy.deepcopy(image)
            new_image.channels[self.dest_channel_index] = result_channel
            return new_image

    def __repr__(self):
        return self.name + ': ' +\
            str(self.src_channel_indexes) + ' -> ' +\
            str(self.dest_channel_index)

    @staticmethod
    def all(channel_count=3):
        """
        Create a list of callable filters
        for various configurations of input color planes.
        """
        channel_indexes = xrange(0, channel_count)

        # Make a list of one-argument filters
        one_arg_calls = []
        for channel_index in channel_indexes:
            for one_arg_filter in one_argument_filters:
                # Source and destination channels match
                filter_call = FilterCall(
                    one_arg_filter,
                    [channel_index],
                    channel_index)
                one_arg_calls.append(filter_call)

        # List of two-argument filter calls
        two_arg_calls = []
        combos = [
            # All possible pairs of distinct channel indexes
            list(c)
            for c in it.combinations(channel_indexes, 2)
        ]
        for combo in combos:
            for dst_channel_index in combo:
                for two_arg_filter in two_argument_filters:
                    filter_call = FilterCall(
                        two_arg_filter,
                        combo,
                        dst_channel_index)
                    two_arg_calls.append(filter_call)

        return one_arg_calls + two_arg_calls


    # @staticmethod
    # def make_calls(
    #         function_list,
    #         color_planes=3,
    #         keep_list_order=False):
    #     """
    #     Create a list of callable filters
    #     for various configurations of input color planes.
    #     """
    #     calls = []
    #     plane_indexes = xrange(0, color_planes)
    #     if not keep_list_order:
    #         # Group output by color planes
    #         arg_count = set([
    #             flt.func_code.co_argcount
    #             for flt in function_list
    #         ])
    #         if len(arg_count) > 1:
    #             raise ValueError(
    #                 "Filters should have the same number of arguments")
    #         elif list(arg_count)[0] > color_planes:
    #             raise ValueError(
    #                 "Filter argument count > color plane count")
    #         else:
    #             arg_count = list(arg_count)[0]
    #             calls = []
    #             plane_indexes = xrange(0, color_planes)
    #             if arg_count == 1:
    #                 for i in plane_indexes:
    #                     for flt in function_list:
    #                         calls += [FilterCall(flt, [i], i)]
    #             else:
    #                 combos = [
    #                     list(c)
    #                     for c in it.combinations(plane_indexes, arg_count)
    #                 ]
    #                 for combo in combos:
    #                     for dst in combo:
    #                         for flt in function_list:
    #                             calls += [FilterCall(flt, combo, dst)]
    #             return calls
    #     else:
    #         # Group output by filters
    #         for flt in function_list:
    #             # Filter argument (color planes) count
    #             arg_count = flt.func_code.co_argcount
    #             if arg_count <= color_planes:
    #                 if arg_count == 1:
    #                     # Single argument filter:
    #                     # src and dst planes are the same
    #                     for i in plane_indexes:
    #                         calls += [FilterCall(flt, [i], i)]
    #                 else:
    #                     # Multiple argument filter:
    #                     # overwrite result into one of input planes.
    #                     # Take all possible combinations with regards to
    #                     # filter argument count
    #                     combos = [
    #                         list(c)
    #                         for c in it.combinations(plane_indexes, arg_count)
    #                     ]
    #                     for combo in combos:
    #                         for dst in combo:
    #                             calls += [FilterCall(flt, combo, dst)]
    #             else:
    #                 # Filter argument count is higher than color plane count
    #                 # Skip this one?
    #                 continue
    #         return calls

    # @staticmethod
    # def all_calls():
    #     import imaging.filters as flt

    #     # Filter setup
    #     filters_one_arg = [
    #         flt.mean,
    #         flt.minimum,
    #         flt.maximum,
    #         flt.hsobel,
    #         flt.vsobel,
    #         flt.sobel,
    #         flt.lightedge,
    #         flt.darkedge,
    #         flt.erosion,
    #         flt.dilation,
    #         flt.inversion
    #     ]

    #     filters_two_args = [
    #         flt.logical_sum,
    #         flt.logical_product,
    #         flt.algebraic_sum,
    #         flt.algebraic_product,
    #         flt.bounded_sum,
    #         flt.bounded_product
    #     ]

    #     # filter_calls = FilterCall.make_calls(filters)
    #     filters1 = FilterCall.make_calls(filters_one_arg)
    #     filters2 = FilterCall.make_calls(filters_two_args)
    #     filter_calls = filters1 + filters2
    #     return filter_calls
