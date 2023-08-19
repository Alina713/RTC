import ctypes

# 加载C++库
cpp_lib = ctypes.CDLL('./cpp_lib.so')

# 定义C++函数的参数和返回类型
cpp_lib.my_function.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpp_lib.my_function.restype = ctypes.c_int


def my_function(array):
    # 将Python列表转换为ctypes数组
    c_array = (ctypes.c_int * len(array))(*array)

    # 调用C++函数
    result = cpp_lib.my_function(c_array, len(array))

    return result


# 示例调用
my_array = [1, 2, 3, 4, 5]
output = my_function(my_array)
print(output)