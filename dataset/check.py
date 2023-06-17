import os
# assign directory
directory = 'dataset/train/images'
sourcedir = 'dataset/train/labels'
destdir = 'dataset/train/checked'
# iterate over files in
# that directory
for filename in os.listdir(directory):
    size = len(filename)
    name = filename[:size - 3]
    target = name + 'txt'
    print(target)
    for files in os.listdir(sourcedir):
        if target in files:
            src = sourcedir + '/' + target
            dest = destdir + '/' + target
            os.rename(src,dest)