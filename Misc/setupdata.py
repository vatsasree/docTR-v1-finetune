import glob
import shutil
import os
import json
from PIL import Image
import hashlib
import argparse
import random
import math
parser = argparse.ArgumentParser()

parser.add_argument("--pathd", 
            help="Path to dataset", 
            type=str, default="./Consortium_dataset/")
parser.add_argument("--pathj", 
            help="Path to json", 
            type=str, default="./sorted_recalls_docs.json")
parser.add_argument("--paths", 
            help="Path to save files", 
            type=str, default="./newfolder")
parser.add_argument("--pathg", 
            help="Path to ground truth", 
            type=str, default="GT/docvisor_consortium_gt/")
parser.add_argument("--ext", 
            help="Extension of files to be considered (supports one at this moment)", 
            type=str, default='tif')
parser.add_argument("--cutoffs",
            help='list of two cutoffs for dividing into easy, medium, hard',
            nargs='+',type=float, default = [10,40])
parser.add_argument("--languages",
            help='list of languages in dataset/to be used in current run',
            nargs='+',type=str, default = ["Assamese","Bangla","Gujarati","Gurumukhi","Hindi","Kannada","Malayalam","Manipuri","Marathi","Oriya","Tamil","Telugu","Urdu"])
parser.add_argument("--region",
            help="Region of interest (line or word)", 
            type=str, default='word')
parser.add_argument("--split",
            help='train val test split percents',
            nargs='+',type=float, default = [60,20,20])            
args = parser.parse_args()


#extension of image to be considered
ext = args.ext
#paths to different folders
path_to_dataset = args.pathd
path_to_json = args.pathj
path_to_groundtruth = args.pathg
path_to_save = args.paths
#iou values to consider
cutoffs = args.cutoffs
#languages under consideration for current run
languages = args.languages
#region under consideration. Either word or line
regiondecision = args.region
split = args.split
#loading all ground truths for languages
print('Loading ground truths....')
data = {}
for language in languages:
    path_to_groundtruth = "GT/docvisor_consortium_gt/"+language+ ".json"
    f = open(path_to_groundtruth)
    data[language] = json.load(f)

#makes the labels for training
def make_labels(impath,labels):
    img = Image.open(impath)

    # get width and height
    width = img.width
    height = img.height
    
    dimensions = img.size

    # display width and height
    # print("dimensions: ",dimensions)
    # print("The height of the image is: ", height)
    # print("The width of the image is: ", width)

    readable_hash = ""
    with open(impath,"rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest();
        # print(type(readable_hash))
    
    ans_key = ""
    
    test_img_path = os.path.basename(impath)
        
    for language in languages:
        image_keys = list(data[language].keys())
        for k in image_keys:
            path = os.path.basename(data[language][k]["imagePath"])
            if path == test_img_path:
                ans_key = k
                break

        if ans_key != "":
            break
    
    if ans_key == "":
        print('something is wrong1')

    regions = []
    for i,region in enumerate(data[language][ans_key]["regions"]):
            if region["regionLabel"] == regiondecision:
                regions.append(region["groundTruth"])
    # print(regions)


    labels[test_img_path] = {
        'img_dimensions': dimensions,
        'img_hash': readable_hash,
        'polygons': regions
    }

    


f = open(path_to_json)
all_recalls_docs = json.load(f)
f.close()
docs = [document_path for (recall,document_path) in all_recalls_docs]
         
print('making the files and folders...')
if os.path.isdir(path_to_save):
    shutil.rmtree(path_to_save)
    
os.mkdir(path_to_save)
os.mkdir(path_to_save + '/train/')
os.mkdir(path_to_save + '/val/')
os.mkdir(path_to_save + '/val/images')
os.mkdir(path_to_save + '/test/')
os.mkdir(path_to_save + '/test/images')
os.mkdir(path_to_save + '/train/Easy/')
os.mkdir(path_to_save + '/train/Medium/')
os.mkdir(path_to_save + '/train/Hard/')
os.mkdir(path_to_save + '/train/Easy/images')
os.mkdir(path_to_save + '/train/Medium/images')
os.mkdir(path_to_save + '/train/Hard/images')

easylist = docs[:cutoffs[0]]
mediumlist = docs[cutoffs[0]:cutoffs[1]]
hardlist = docs[cutoffs[1]:]

random.shuffle(easylist)
random.shuffle(mediumlist)
random.shuffle(hardlist)

traineasy = easylist[:math.floor(len(easylist)*split[0]/100)]
val = easylist[math.floor(len(easylist)*split[0]/100):math.floor(len(easylist)*(split[0]+split[1])/100)]
test = easylist[math.floor(len(easylist)*(split[0]+split[1])/100): ]

trainmedium = mediumlist[:math.floor(len(mediumlist)*split[0]/100)]
val = val + mediumlist[math.floor(len(mediumlist)*split[0]/100):math.floor(len(mediumlist)*(split[0]+split[1])/100)]
test = test + mediumlist[math.floor(len(mediumlist)*(split[0]+split[1])/100): ]

trainhard = hardlist[:math.floor(len(hardlist)*split[0]/100)]
val = val + hardlist[math.floor(len(hardlist)*split[0]/100):math.floor(len(hardlist)*(split[0]+split[1])/100)]
test = test + hardlist[math.floor(len(hardlist)*(split[0]+split[1])/100): ]

print("starting train hard")
labels = {}
for tiffile in traineasy:
    shutil.copy(tiffile, path_to_save + '/train/Hard/images')
    make_labels(tiffile,labels)

outfile = open(path_to_save + '/train/Hard/labels.json',"w")
json.dump(labels, outfile, indent = 6)
outfile.close()
    
print("starting train medium")
labels = {}
for tiffile in trainmedium:
    shutil.copy(tiffile, path_to_save + '/train/Medium/images')
    make_labels(tiffile,labels)

outfile = open(path_to_save + '/train/Medium/labels.json',"w")
json.dump(labels, outfile, indent = 6)
outfile.close()

print("starting train easy")
labels = {}
for tiffile in trainhard:
    shutil.copy(tiffile, path_to_save + '/train/Easy/images')
    make_labels(tiffile,labels)

outfile = open(path_to_save + '/train/Easy/labels.json',"w")
json.dump(labels, outfile, indent = 6)
outfile.close()

print("starting val")
labels = {}
for tiffile in val:
    shutil.copy(tiffile, path_to_save + '/val/images')
    make_labels(tiffile,labels)

outfile = open(path_to_save + '/val/labels.json',"w")
json.dump(labels, outfile, indent = 6)
outfile.close()

print("starting test")
labels = {}
for tiffile in test:
    shutil.copy(tiffile, path_to_save + '/test/images')
    make_labels(tiffile,labels)

outfile = open(path_to_save + '/test/labels.json',"w")
json.dump(labels, outfile, indent = 6)
outfile.close()
print('done!')