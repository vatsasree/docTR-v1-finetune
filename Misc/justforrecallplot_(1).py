import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import random
import cv2
import os
import torch
import torchvision
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from tqdm import tqdm
import shutil
import copy
import argparse
from tabulate import tabulate

#arguments taken from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--k", 
            help="Number of documents in Easy/Medium/Hard", 
            type=int, default=20)
parser.add_argument("--ext", 
            help="Extension of files to be considered (supports one at this moment)", 
            type=str, default='tif')
parser.add_argument("--pathg", 
            help="Path to ground truth", 
            type=str, default='./Consortium_GT/')
parser.add_argument("--paths", 
            help="Path to save files", 
            type=str, default='./docvisor_saves/Original/')
parser.add_argument("--pathl", 
            help="Path to reclist", 
            type=str, default='./reclists/image_recall_list.json')
parser.add_argument("--pathlf", 
            help="Path to reclist", 
            type=str, default='./reclists/image_recall_list_finetuned.json')
parser.add_argument("--region",
            help="Region of interest (line or word)", 
            type=str, default='word')
parser.add_argument("--iouthresholds",
            help='list of IoU thresholds for determining True/False positives',
            nargs='+',type=float, default = [0.6])
parser.add_argument("--languages",
            help='list of languages in dataset/to be used in current run',
            nargs='+',type=str, default = ["Assamese","Bangla","Gujarati","Gurumukhi","Hindi","Kannada","Malayalam","Manipuri","Marathi","Oriya","Tamil","Telugu"])
parser.add_argument("--test", 
            help="Is this running locally (for testing)", 
            type=bool, default=0)
args = parser.parse_args()

#number of images in easy/medium/hard
K = args.k
#extension of image to be considered
ext = args.ext
#region under consideration. Either word or line
regiondecision = args.region
#paths to different folders
path_to_groundtruth = args.pathg
path_to_save = args.paths
path_to_reclist = args.pathl
path_to_reclistf = args.pathlf

#languages under consideration for current run
languages = args.languages
#iou values to consider
iouthresholds = args.iouthresholds
test = args.test

print(f'K ={K}')
print(f'extension ={ext}')
print(f'decision region ={regiondecision}')
print(f'path to ground truth ={path_to_groundtruth}')
print(f'path to save files ={path_to_save}')
print(f'path to reclists ={path_to_reclist}')
print(f'languages ={languages}')
print(f'IoU thresholds ={iouthresholds}')
print(f'Test ={test}')

print()


f = open(path_to_reclist)
image_recall_list = json.load(f)
f.close()

f = open(path_to_reclistf)
image_recall_list_finetuned = json.load(f)
f.close()

recalls = [rec for (rec,pre,test_img_path,pred_list,gt_list) in image_recall_list]
recalls_finetuned = [rec for (rec,pre,test_img_path,pred_list,gt_list) in image_recall_list_finetuned]

recalls = np.array(recalls)
recalls.sort()
recalls_finetuned = np.array(recalls_finetuned)
recalls_finetuned.sort()
# print(len(recalls))
# print(recalls)
# plt.plot(list(range(len(recalls))),recalls)#,label = 'recalls sorted, orginal model')
# plt.xlabel('index of document')
# plt.ylabel('recall')
# plt.title('sorted recall - original')
# plt.savefig('recallplot_original.eps')

plt.plot(list(range(len(recalls_finetuned))),recalls_finetuned)#,label = 'recalls sorted, finetuned model')
plt.xlabel('index of document')
plt.ylabel('recall')
plt.title('sorted recall - fine tuned')
plt.savefig('recallplot_finetuned.eps')


