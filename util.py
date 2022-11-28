import numpy as np
import matplotlib.pyplot as plt

def unitVector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angleBetween(v1, v2):
    # https://blog.finxter.com/calculating-the-angle-clockwise-between-2-points/
    ang1 = np.arctan2(v1[1], v1[0])
    ang2 = np.arctan2(v2[1], v2[0])
    return (ang1 - ang2) % (2 * np.pi)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# https://stackoverflow.com/a/53586826
def multipleFormatter(den=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    def _multipleFormatter(x, pos):
        denominator = den
        num = np.int(np.rint(denominator * x / number))
        com = gcd(num, denominator)
        (num, denominator) = (int(num/com), int(denominator/com))
        if denominator == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, denominator)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, denominator)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, denominator)
    return _multipleFormatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multipleFormatter(self.denominator, self.number, self.latex))