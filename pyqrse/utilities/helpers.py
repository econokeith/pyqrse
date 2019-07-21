__author__='Keith Blackwell'
import datetime
import inspect
import copy
import numpy
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
    creates a dictionary of kernels (values) and their string codes (key)

    key = kernel's code
    value = non-instantiated kernel object

    Example: ::

        >>>kernel_hash['S']
        pyqrse.kernels.binarykernels.SQRSEKernel

    Arg:
        kernel (QRSEKernelBase object) :

    Returns:
        dict

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

                print("Warning: {} was used for {} and {}".format(code,
                                                                  k1_name,
                                                                  k2_name))

                while True:
                    code += '_'
                    if code not in kernel_hash:
                        break

                print('  Code = {} was used for {} instead'.format(code,
                                                                   k1_name))

            kernel_hash[code] = the_kernel

    return kernel_hash

def docthief(mark_function):
    """
    Gives decorated function the __docstring__ of the mark_function

    Args:
        mark_function : function to copy __docstring__ from

    """
    def decorator(thief_function):
        def wrapper(*args, **kwargs):
            return thief_function(*args, **kwargs)
        wrapper.__doc__ = mark_function.__doc__
        return wrapper
    return decorator

def kwarg_filter(kwargs, target_function):
    """
    filters kwargs of outside function to make sure they work in
    target function. Note: this function may have unexpected results
    if outer and target functions have the same keyword arguments
    assigned to different uses.

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

def matrix_print(mat_to_print, rb=None, cb="|", f=2):
    """
    prints matrix or np.arrays prettier

    Args:
        mat_to_print (2d list) : list of lists where each item in each row
            is either a str, int, or float
        rb(str): Single character to string for rows in between the
            rows of the matrix. If None (default), no breaks between rows
        cb(str): Single character to demarcate column breaks. Defaults is "|"
        f(int): the number of decimals to show in floats.

    """

    new_body = copy.deepcopy(mat_to_print)

    f_check = "{: ."+ str(f)+ "f}"

    if isinstance(new_body, numpy.ndarray):
        body = []
        for row in new_body:
            body.append(list(row))
    else:
        body=new_body

    max_len = 0
    for line in body:
        if len(line) > max_len:
            max_len = len(line)

    for line in body:
        if len(line) < max_len:
            line += ['']*(max_len-len(line))

    max_width = [0]*max_len
    for i, line in enumerate(body):
        for j in range(len(line)):
            try:
                body[i][j] = f_check.format(body[i][j])
            except:
                pass

            if len(body[i][j]) > max_width[j]:
                max_width[j] = len(body[i][j])

    line_style = ''
    for width in max_width:
        line_style += (cb+' {: ^')
        line_style += str(width)
        line_style += '} '
    line_style += cb

    output = ''
    for line in body:
        add_line = line_style.format(*line)
        output+= add_line
        output+="\n"

        if rb:
            output+= rb*len(add_line)
            output+= "\n"

    # if row_break:
    #     output = row_break*len(add_line)+"\n" + output

    print(output)

#short-hand to use in notebooks
@docthief(matrix_print)
def mprint(mat_to_print, rb=None, cb="|", f=2):
    return matrix_print(mat_to_print, rb=rb, cb=cb, f=f)

# Todo : Figure out how to have sphinx recognize this as a propery or class method
# Todo : embed in an easier to

# this can't been tested with @classmethod decorator
class ReadOnlyClassProperty:
    """
    read-only class property

    Allows class methods to be called like property with no
    setter declared.

    """
    def __init__(self, method):
        self.method = method

    def __get__(self, owner, otype):
        return self.method(otype)

    def __set__(self, owner, value):
        raise AttributeError("this attribute is protected")


def read_only_class_property(function):
    return ReadOnlyClassProperty(function)
