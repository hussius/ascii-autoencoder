import numpy as np
import colored
import sys
import math

lookup = [  1,196,197,198,199,208,209,172,149,  2]
def print_row(a):
    for x in a:
        foo = np.around(math.floor(x * 999))
        index = int(foo / 100)
        color = colored.bg(lookup[10-index])
        res   = colored.attr('reset')
        sys.stdout.write (color + str(int(foo)) + res)
        #sys.stdout.write(" ")
    sys.stdout.write("\n")


def print_row2(a, a2):
    for i in range(0,len(a)):
        x = a[i]
        color = colored.bg(lookup[9-x])
        if a2 is not None:
            color += colored.fg(lookup[9-a2[i]])
        res   = colored.attr('reset')
        sys.stdout.write (color + str(x) + res)
        #sys.stdout.write(" ")
    sys.stdout.write("\n")

#print_row([  1,196,197,198,199,208,209,172,149,  2])
