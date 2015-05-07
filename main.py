#!/bin/python
#
#   DHash implementation for image matching
#   Optimized for mass image loads
#   Uses VP-tree for hash comparisons
#
from PIL import Image
from fnmatch import filter
from multiprocessing import Process
from pylab import linspace, meshgrid, arange, imshow, show, zeros, asarray
import os, re
import argparse
import shelve
import sys, random, heapq

db = None

class VP_Node:
    def minimum_distance(self, distances):
        minimum = 0.0
        for i in xrange(len(distances)):
            if distances[i] < self.lower_bounds[i]:
                minimum = max(minimum, self.lower_bounds[i] - distances[i])
            elif distances[i] > self.upper_bounds[i]:
                minimum = max(minimum, distances[i] - self.upper_bounds[i])
        return minimum

    def help_find(self, item, distances, heap, distance):
        d = distance(self.vantage, item)
        new_distances = distances + (d,)

        heapq.heappush(heap, (d, 0, self.vantage))

        for child in self.children:
            heapq.heappush(heap, (child.minimum_distance(new_distances), 1, child, new_distances))

class VP_tree:
    def __init__(self, items, distance, max_children=2):
        """ items        : list of items to make tree out of
            distance     : function that returns the distance between two items
            max_children : maximum number of children for each node

            Using larger max_children will reduce the time needed to construct the tree,
            but may make queries less efficient.
        """

        self.distance = distance

        items = [ (item, ()) for item in items ]
        random.shuffle(items)
        self.root = make_VP_Node(items, distance, max_children)

    def find(self, item):
        """ Return iterator yielding items in tree in order of distance from supplied item.
        """

        if not self.root: return

        heap = [ (0, 1, self.root, ()) ]

        while heap:
            top = heapq.heappop(heap)
            if top[1]:
                top[2].help_find(item, top[3], heap, self.distance)
            else:
                yield top[2], top[0]
    def insert(self,item):
        print a

def make_VP_Node(items, distance, max_children):
    if not items:
        return None

    node = VP_Node()

    node.lower_bounds = [ ]
    node.upper_bounds = [ ]
    for i in xrange(len(items[0][1])):
        distance_list = [ item[1][i] for item in items ]
        node.lower_bounds.append(min(distance_list))
        node.upper_bounds.append(max(distance_list))

    node.vantage = items[0][0]
    items = items[1:]

    node.children = [ ]

    if not items:
        return node

    items = [ (item[0], item[1]+(distance(node.vantage, item[0]),)) for item in items ]

    distances = { }
    for item in items: distances[item[1][-1]] = True
    distance_list = distances.keys()
    distance_list.sort()
    n_children = min(max_children, len(distance_list))
    split_points = [ -1 ]
    for i in xrange(n_children):
        split_points.append(distance_list[(i+1)*(len(distance_list)-1)//n_children])

    for i in xrange(n_children):
        child_items = [ item for item in items if split_points[i] < item[1][-1] <= split_points[i+1] ]
        child = make_VP_Node(child_items,distance,max_children)
        if child: node.children.append(child)

    return node

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

def avhash(im):
    '''
        Perceptual Image Hash algorithm
    '''
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.0
    return reduce(lambda x,
     (y, z): x | (z << y),
     enumerate(map(lambda i: 0 if i < avg else 1,
         im.getdata())),
         0)

def dhash(image,hash_size=16):
    #Grayscale and shrink image
    image = image.convert('L').resize(
    (hash_size+1,hash_size),
    Image.ANTIALIAS,
    )

    pixels = list(image.getdata())

    #compare adjacent pixels
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)

    #Convert the binary array to hex string
    decimal_value = 0
    hex_string = []
    for index,value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8)==7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    return ''.join(hex_string)

def compute_hashs(filenames):
    #print filenames
    func = dhash
    if type(filenames) == []:
        for file in filenames:
            image = Image.open(file)
            image_hash = func(image)
            db[image_hash] = file
    else:
        image = Image.open(filenames)
        image_hash = func(image)
        db[image_hash] = filenames
    db.sync()

def hamming_distance_num(hash1,hash2):
    h,d = 0,hash1^hash2
    while d:
        h += 1
        d &= d-1
    return h

def hamming_distance_string(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def load_images(directory):
    '''
        Load in the image file names
    '''
    jpegs = [filename for filename in os.listdir(directory)
              if re.search(r'\.jpg$', filename, re.IGNORECASE)]
    jpegs2 = [filename for filename in os.listdir(directory)
              if re.search(r'\.jpeg$', filename, re.IGNORECASE)]
    png = [filename for filename in os.listdir(directory)
              if re.search(r'\.png$', filename, re.IGNORECASE)]
    results = jpegs+jpegs2+png
    for i in xrange(0,len(results)):
        results[i] = directory+"/"+results[i]
    return results

if __name__ =='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset",
        required = True,
        help = "path to input dataset of images")
    ap.add_argument("-s", "--shelve",
        required = True,
        help = "output shelve database")
    args = vars(ap.parse_args())
    db = shelve.open(args["shelve"], writeback = True)

    files = load_images(args["dataset"])
    #for file in files: print file
    compute_hashs.parallel = parallel_attribute(compute_hashs)
    parallel = compute_hashs.parallel(files)

    tree = VP_tree(db.keys(), hamming_distance_string, 100)

    print "Search for:",db.keys()[0]
    results = []
    count = 0
    k = 10
    for result in tree.find(db.keys()[0]):
        #print "Found:",result
        results.append(result)
        count+=1
        if(count>=k):
            break

    for result in results: print result

    db.close()
