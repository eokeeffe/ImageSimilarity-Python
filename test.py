
from __future__ import division
from pylab import linspace, meshgrid, arange, imshow, show, zeros, asarray
from time import time

# a function in parallel is easy
def parallel_function(f):
    def easy_parallize(f, sequence):
        # I didn't see gains with .dummy; you might
        from multiprocessing import Pool
        pool = Pool(processes=8)
        #from multiprocessing.dummy import Pool
        #pool = Pool(16)

        # f is given sequence. guaranteed to be in order
        result = pool.map(f, sequence)
        cleaned = [x for x in result if not x is None]
        cleaned = asarray(cleaned)
        # not optimal but safe
        pool.close()
        pool.join()
        return cleaned
    from functools import partial
    # this assumes f has one argument, fairly easy with Python's global scope
    return partial(easy_parallize, f)

def parallel_attribute(f):
    def easy_parallize(f, sequence):
        # I didn't see gains with .dummy; you might
        from multiprocessing import Pool
        pool = Pool(processes=8)
        #from multiprocessing.dummy import Pool
        #pool = Pool(16)

        # f is given sequence. Guaranteed to be in order
        result = pool.map(f, sequence)
        cleaned = [x for x in result if not x is None]
        cleaned = asarray(cleaned)
        # not optimal but safe
        pool.close()
        pool.join()
        return cleaned
    from functools import partial
    # This assumes f has one argument, fairly easy with Python's global scope
    return partial(easy_parallize, f)

# how it should be done, but doesn't give iteration results
def mandel_numpy():
    N = 100
    i = linspace(-2, 1, num=N)
    x, y = meshgrid(i, i)
    z = x + y*1j
    for i in arange(50):
        z = pow(z, 2) + x+y*1j
    imshow(abs(z))
    show()

# test one pixel
def mandel_pixel(x, y):
    limit = 100
    z = x + y*1j
    for i in arange(50):
        z = pow(z, 2) + (x+y*1j)
        if abs(z) > limit:
            #m[argwhere(y_arr == y),  argwhere(x_arr == x)] = i
            #m[argwhere(y_arr == y),  x] = i
            return i
            break
    return 0

# test each pixel
def mandel_serial():
    m = zeros((N,N))
    i=-1;
    for x in linspace(-2, 1, num=N):
        i += 1
        j = -1
        for y in linspace(-2, 1, num=N):
            j += 1
            m[j,i] = mandel_pixel(x, y)
    return m

# test each *row* of pixels. called with mandel_p.parallel(-2...1)
def mandel_p(x):
    #m = zeros((N,N))
    m = zeros(N)
    #for x in linspace(-2, 1, num=N):
    j = -1
    for y in linspace(-2, 1, num=N):
        j += 1
        #m[j,i] = mandel_pixel(x, y)
        m[j] = mandel_pixel(x, y)

    return m


# time test
print "mandel"
global N
N = 1e2
start = time()
serial = mandel_serial()
mandel_serial = time() - start
print "    serial:", mandel_serial

start = time()
mandel_p.parallel = parallel_attribute(mandel_p)
parallel = mandel_p.parallel(linspace(-2, 1, num=N))
parallel = parallel.T
mandel_parallel = time() - start

print "    parallel:", mandel_parallel
print "    speedup: ", mandel_serial / mandel_parallel
print "    check:", (parallel == serial).all()


def test_prime(n):
    prime = True
    for i in arange(2, n):
        if n/i % 1 == 0:
            prime = False
    return prime
def test_prime_p(n):
    prime = True
    for i in arange(2, n):
        if n/i % 1 == 0:
            prime = False
    return prime

print "prime"
N = 1e3


serial = zeros(N)
start = time()
for i in arange(N):
    serial[i] = test_prime(i)
serial_time = time() - start
print "    serial:", serial_time

test_prime.parallel = parallel_attribute(test_prime)
start = time()
parallel = test_prime.parallel(arange(N))
parallel_time = time() - start
print "    parallel:", parallel_time

speedup = serial_time / parallel_time
print "    speedup:", speedup
print "    check:", (serial == parallel).all()

# this method is slightly slower than multiprocessing (for this example on my
# machine at least)
#from IPython.parallel import Client
#rc = Client() # requires `ipython cluster` to be run. reason commented out.
#dview = rc[:]
#with dview.sync_imports():
    #from pylab import linspace, meshgrid, arange, imshow, show, zeros, asarray
#start = time()
#ipy = dview.map_sync(test_prime, arange(N))
#results = asarray([x for x in ipy])
#ipy_time_prime = time() - start
#print "    ipy:", ipy_time_prime
