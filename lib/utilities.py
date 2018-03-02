
def extent_list(dst_list, elem):
    if len(dst_list) == 0:
        return elem
    dst_list.extend(elem)
    return dst_list

