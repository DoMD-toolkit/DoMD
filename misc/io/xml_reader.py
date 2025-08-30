import re
from io import StringIO
from xml.etree import cElementTree

import numpy as np


def control_in(control_file):
    pass


class Box(object):
    def __init__(self):
        self.xy = 0
        self.xz = 0
        self.yz = 0
        return

    def update(self, dic):
        self.__dict__.update(dic)


class XmlParser(object):
    def __init__(self, filename, needed=None):
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        self.box = Box()
        self.data = {}
        needed = [] if needed is None else needed
        for key in root[0].attrib:
            self.__dict__[key] = int(root[0].attrib[key])
        for element in root[0]:
            if element.tag == 'box':
                self.box.update(element.attrib)
                continue
            if (len(needed) > 0) and (element.tag not in needed):
                continue
            if element.tag == 'reaction':
                self.data['reaction'] = []
                reaction_list = element.text.strip().split('\n')
                while '' in reaction_list:
                    reaction_list.remove('')
                for l in reaction_list:
                    r = re.split(r'\s+', l)
                    while '' in r:
                        r.remove('')
                    r[1:] = [int(_) for _ in r[1:]]
                    self.data['reaction'].append(r)
                continue
            if element.tag == 'template':
                self.data['template'] = eval('{%s}' % element.text)
                continue
            if len(element.text.strip()) > 0:
                self.data[element.tag] = np.genfromtxt(StringIO(element.text), dtype=None, encoding=None)
