def reshape_1rest(arr):
    # type: (list) -> list[list]
    arr = [arr]
    return arr

def reshape_rest1(arr):
    # type: (list) -> list[list]
    arr = [[x] for x in arr]
    return arr

def squeeze(arr):
    # type: (list[list]) -> list
    return [x[0] for x in arr]