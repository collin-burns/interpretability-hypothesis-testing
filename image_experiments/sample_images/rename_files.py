import os
from os import walk
from shutil import copyfile

dir = "."
new_dir = "renamed_imgs"
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

for (dirpath, dirnames, filenames) in walk(dir):
    for i, filename in enumerate(filenames):
        new_name = "img{}.jpg".format(i+1)
        copyfile(filename, os.path.join(new_dir, new_name))

    break




