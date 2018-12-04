import os
import shutil
import hashlib
from multiprocessing import Pool
import imageio
import random

##Get all filenames
all_files = []

for f in os.listdir('./all_photos/default_photos/'):
    all_files.append('default_photos/{}'.format(f))

for f in os.listdir('./all_photos/opioid_photos/'):
    all_files.append('opioid_photos/{}'.format(f))



def hasher(fileName):
    hasher = hashlib.md5()
    with open('./all_photos/{}'.format(fileName),'rb') as inFile:
        buf = inFile.read()
        hasher.update(buf)
    a = imageio.imread('./all_photos/{}'.format(fileName))
    least_size = a.shape[0] if a.shape[0] <= a.shape[1] else a.shape[1]
    return (hasher.hexdigest(),least_size)  

with Pool(20) as p:
    hexes = p.map(hasher,all_files)

set_deduper = set()
filtered_files = []

for idx,info in enumerate(hexes):
    h,size = info
    if size >= 224:
        if h not in set_deduper:
            set_deduper.add(h)
            filtered_files.append(all_files[idx])
        else:
            print('file filtered for duplicate {}'.format(all_files[idx]))
    else:
        print('file filtered for size {}'.format(all_files[idx]))

##sample files
cut_size = len(filtered_files)//10

random.shuffle(filtered_files)
test = filtered_files[0:cut_size]
val = filtered_files[cut_size:cut_size*2]
train = filtered_files[cut_size*2:]

def setup_folder(folder_name,files):
    os.makedirs('{}/default_photos'.format(folder_name))
    os.makedirs('{}/opioid_photos'.format(folder_name))

    for f in files:
        shutil.copy('./all_photos/{}'.format(f),'{}/{}'.format(folder_name,f))

setup_folder('./train_data',train)
setup_folder('./val_data',val)
setup_folder('./test_data',test)
