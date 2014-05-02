import itertools as it


class FilterCall(object):
    """
    Wraps filter call on specified image planes
    (represented as list indexes)
    """
    def __init__(self, function, src_planes, dst_plane):
        self.name = function.func_name
        self.function = function        # filter to call
        self.src_planes = src_planes    # source plane indexes
        self.dst_plane = dst_plane      # destination plane index

    def __call__(self, image_planes):
        # positional filter arguments
        input_planes = [
            image_planes[i]
            for i in self.src_planes
        ]
        result = self.function(*input_planes)
        idx = self.dst_plane
        return image_planes[:idx] + [result] + image_planes[idx + 1:]

    def __repr__(self):
        return self.name + ': '
        + str(self.src_planes) + ' -> '
        + str(self.dst_plane)

    @staticmethod
    def run_sequence(image, filter_list):
        """
        Runs filter sequence on provided image
        """
        filtered_image = image
        for flt in filter_list:
            filtered_image = flt(filtered_image)
        return filtered_image

    @staticmethod
    def make_calls(
            function_list,
            color_planes=3,
            keep_list_order=False):
        """
        Create a list of callable filters
        for various configurations of input color planes.
        """
        calls = []
        plane_indexes = xrange(0, color_planes)
        if not keep_list_order:
            # Group output by color planes
            arg_count = set([
                flt.func_code.co_argcount
                for flt in function_list
            ])
            if len(arg_count) > 1:
                raise ValueError(
                    "Filters should have the same number of arguments")
            elif list(arg_count)[0] > color_planes:
                raise ValueError(
                    "Filter argument count > color plane count")
            else:
                arg_count = list(arg_count)[0]
                calls = []
                plane_indexes = xrange(0, color_planes)
                if arg_count == 1:
                    for i in plane_indexes:
                        for flt in function_list:
                            calls += [FilterCall(flt, [i], i)]
                else:
                    combos = [
                        list(c)
                        for c in it.combinations(plane_indexes, arg_count)
                    ]
                    for combo in combos:
                        for dst in combo:
                            for flt in function_list:
                                calls += [FilterCall(flt, combo, dst)]
                return calls
        else:
            # Group output by filters
            for flt in function_list:
                # Filter argument (color planes) count
                arg_count = flt.func_code.co_argcount
                if arg_count <= color_planes:
                    if arg_count == 1:
                        # Single argument filter:
                        # src and dst planes are the same
                        for i in plane_indexes:
                            calls += [FilterCall(flt, [i], i)]
                    else:
                        # Multiple argument filter:
                        # overwrite result into one of input planes.
                        # Take all possible combinations with regards to
                        # filter argument count
                        combos = [
                            list(c)
                            for c in it.combinations(plane_indexes, arg_count)
                        ]
                        for combo in combos:
                            for dst in combo:
                                calls += [FilterCall(flt, combo, dst)]
                else:
                    # Filter argument count is higher than color plane count
                    # Skip this one?
                    continue
            return calls
