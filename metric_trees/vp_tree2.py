
""" VP-tree data structure

    Can be used for nearest-neighbour querys in any metric space.

    This software was written by Paul Harrison, and is released into the public domain.
"""

import sys, random, heapq

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

if __name__ == '__main__':
    # Test: Simple spelling corrector
    #python vp_tree2.py /usr/share/dict/words
    try:
        import psyco
        psyco.full()
    except: pass

    if len(sys.argv) != 2:
        print 'Please supply the filename of your dictionary, eg /usr/share/dict/words'
        sys.exit(1)

    comparison_count = 0

    def distance(a,b):
        """ Calculates the Levenshtein distance between a and b.
            (from http://hetland.org/python/) """
        global comparison_count
        comparison_count += 1

        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a,b = b,a
            n,m = m,n

        current = range(n+1)
        for i in range(1,m+1):
            previous, current = current, [i]+[0]*n
            for j in range(1,n+1):
                add, delete = previous[j]+1, current[j-1]+1
                change = previous[j-1]
                if a[j-1] != b[i-1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    print 'Load dictionary'
    words = [ item.strip() for item in open(sys.argv[1],'r') if item.strip() ]
    print '%d words' % len(words)

    print
    print 'Construct tree'
    tree = VP_tree(words, distance, 100)
    print '%d comparisons' % comparison_count
    comparison_count = 0

    print
    print 'Ready to answer queries'
    while True:
        print
        query = raw_input('query> ').strip()
        if not query: break

        n = 0
        for result in tree.find(query):
            print '% 6d comparisons later... % 5d %s' % (comparison_count, result[1], result[0])
            comparison_count = 0
            n += 1
            if n >= 5: break
