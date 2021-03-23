def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters =int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))