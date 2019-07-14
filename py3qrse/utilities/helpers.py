import datetime
import inspect



### utilities and convenience tools

def date_to_datetime(d):
    return datetime.date(int(d[:4]),int(d[4:6]), int(d[6:]) )

def datetime_to_date(d):
    day = d.day
    month = d.month
    day = '0'+str(day) if day < 10 else str(day)
    month = '0'+str(month) if month < 10 else str(month)
    return '{}{}{}'.format(d.year, month, day)

def split_strip_parser(parser, key1, key2):
    out = parser[key1][key2]
    out = out.split(',')
    return [x.strip() for x in out]

def kernel_hierarchy_to_hash_bfs(kernel):
    """
    returns all subclasses of a kernel function as a hash
     key = kernel's code , value = non-instantiated kernel object
     example: kernel_hash['S'] = SQRSEKernel
    :param kernel:
    :return:
    """

    kernel_hash = {}
    queue = [kernel]

    while queue:

        the_kernel = queue.pop(0)
        sub_classes = the_kernel.__subclasses__()
        queue += sub_classes
        code = the_kernel.getcode()

        if code is not None:

            if code in kernel_hash:

                k1_name = the_kernel.__name__
                k2_name = kernel_hash[code].__name__
                print("Warning: {} was used for {} and {}".format(code, k1_name, k2_name))

                while True:
                    code = code + '_'
                    if code not in kernel_hash:
                        break

                print('  Code = {} was used for {} instead'.format(code, k1_name))

            kernel_hash[code] = the_kernel

    return kernel_hash

def docthief(mark_function):
    """
    decorator function that allows the thief to return the mark_function's docstring
    :param mark_function:
    :return:
    """
    def decorator(thief_function):
        def wrapper(*args, **kwargs):
            return thief_function(*args, **kwargs)
        wrapper.__doc__ = mark_function.__doc__
        return wrapper
    return decorator

def kwarg_filter(kwargs, target_function):
    """
    filters kwargs of outside function to make sure they work in target function.
    Note: this function may have unexpected results if outer and target functions have the same
    keyword arguments assigned to different uses.

    :param kwargs: kwargs from outer function
    :param target_function: function filtered kwargs will be given to
    :return: dictionary of keyword arguments
    """
    t_args = inspect.getfullargspec(target_function)

    if t_args.varkw:
        return kwargs

    new_kwargs = {}

    for k, v in kwargs.items():
        if k in t_args.args:
            new_kwargs[k]=v

    return new_kwargs

