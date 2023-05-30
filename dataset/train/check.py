#!/usr/bin/env python
# import required module
import os
# assign directory
directory = 'images'
sourcedir = 'labels'
destdir = 'checked'
# iterate over files in
# that directory
for filename in os.listdir(directory):
    size = len(filename)
    name = filename[:size - 3]
    target = name + 'txt'
    for files in os.listdir(sourcedir):
        if target not in files:
            src = os.path.join(sourcedir, target)
            dest = os.path.join(destdir, target)
            os.rename(src, dest)