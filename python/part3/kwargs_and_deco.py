import time
import functools

def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if(args):
            arg_lst.append(', '.join(repr(arg) for arg in args))
            # ', '.join(the generator) -> 'arg1, arg2, ..., argn'
        if(kwargs):
            pairs = ['%s=%r' %(k, w) for k, w in sorted(kwargs.items())]
            # ['keyword1=arg1', 'keyword2=arg2', ...] sorted according to keywords
            # This 'pairs' var is omitted in the 'args' branch.
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %r ' % (elapsed, name, arg_str, result))
        return result
    return clocked

def test_args(*args, **kwargs):
    print(args)
    print(kwargs)
    print(kwargs.items())
    print(sorted(kwargs.items()))
    return 0

@clock
def testf(a, b=2, c=3):
    t = [2 * x for x in range(10**5)]
    return a, c

if __name__ == '__main__':
    test_args(3, z=1, x=4, y=5)
    """output:
    (3,)
    {'z': 1, 'x': 4, 'y': 5}
    dict_items([('z', 1), ('x', 4), ('y', 5)])
    [('x', 4), ('y', 5), ('z', 1)]
    """
    print(sorted(((3,1),(1,3),(1,2))))
    # -> [(1, 2), (1, 3), (3, 1)]

    for i in range(2):
        testf(i,c=3)
        testf(i, b=4, c=5)
    """output:
    [0.01109362s] testf(0, c=3) -> (0, 3)
    [0.01510334s] testf(0, b=4, c=5) -> (0, 5)
    [0.01061773s] testf(1, c=3) -> (1, 3)
    [0.00986600s] testf(1, b=4, c=5) -> (1, 5)
    """