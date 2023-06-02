
# Goal: 
# take default model, OR fine tuned model
# sort based on recall/iou and make a json list with recall, precision, image name, pred
# makes recall lists
# Might need to run this if entire dataset is needed for evaluation metrics. Otherwise we can use the already computed ones


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
from collections import OrderedDict


#arguments taken from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--k", 
            help="Number of documents in Easy/Medium/Hard", 
            type=int, default=20)
parser.add_argument("--ext", 
            help="Extension of files to be considered (supports one at this moment)", 
            type=str, default='tif')
parser.add_argument("--languages",
            help='list of languages in dataset/to be used in current run',
            nargs='+',type=str, default = ["Assamese","Bangla","Gujarati","Gurumukhi","Hindi","Kannada","Malayalam","Manipuri","Marathi","Oriya","Tamil","Telugu"])
parser.add_argument("--test", 
            help="Is this running locally (for testing)", 
            type=bool, default=0)
parser.add_argument("--pathd", 
            help="Path to dataset", 
            type=str, default="/scratch/sreevatsa/scratch/abhaynew/newfolder/test/images/")
parser.add_argument("--pathg", 
            help="Path to ground truth", 
            type=str, default='/scratch/sreevatsa/scratch/abhaynew/newfolder/test/labels.json')
parser.add_argument("--paths", 
            help="Path to save files", 
            type=str, default='/home2/sreevatsa/reclists')
parser.add_argument("--region",
            help="Region of interest (line or word)", 
            type=str, default='word')
parser.add_argument("--iouthresholds",
            help='list of IoU thresholds for determining True/False positives',
            nargs='+',type=float, default = [0.6])

parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")

args = parser.parse_args()

#number of images in easy/medium/hard
K = args.k
#extension of image to be considered
ext = args.ext
#region under consideration. Either word or line
regiondecision = args.region
#paths to different folders
path_to_dataset = args.pathd
path_to_groundtruth = args.pathg
path_to_save = args.paths
#languages under consideration for current run
languages = args.languages
#iou values to consider
iouthresholds = args.iouthresholds
test = args.test

print(f'K ={K}')
print(f'extension ={ext}')
print(f'decision region ={regiondecision}')
print(f'path to dataset ={path_to_dataset}')
print(f'path to ground truth ={path_to_groundtruth}')
print(f'path to save files ={path_to_save}')
print(f'languages ={languages}')
print(f'IoU thresholds ={iouthresholds}')
print(f'Test ={test}')
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor = []
if isinstance(args.resume, str):
    predictor = ocr_predictor(pretrained=True).to(device)
    # original saved file with DataParallel
    state_dict = torch.load(args.resume)
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    predictor.det_predictor.model.load_state_dict(new_state_dict)
else:
    predictor = ocr_predictor(pretrained=True).to(device)

#changed to add iou_threshold
# def get_precision_recall(gt, pred, iou_threshold):
#     matrix = torchvision.ops.box_iou(gt,pred).numpy()
#     try:    
#         max_iou_gt = np.max(matrix, axis=1)
#         max_iou_pred = np.max(matrix, axis=0)

#         TP = np.sum((max_iou_pred>iou_threshold)*max_iou_pred)
#         FP = max_iou_pred.shape[0] - np.sum(max_iou_pred>iou_threshold)
#         FN = np.sum(max_iou_gt<iou_threshold)

#         return TP/(TP+FN), TP/(TP+FP)
        
#     except IndexError:
#         pass    

def get_precision_recall(gt, pred, iou_threshold):
    gt = torch.Tensor(gt)
    pred = torch.Tensor(pred)
    
    if len(gt) == 0 or len(pred) == 0:
        return 0, 0
    
    # if gt.dim() != 2 or gt.size(1) != 4:
    #     raise ValueError("Invalid shape for gt. Expected (N, 4).")
    
    # if pred.dim() != 2 or pred.size(1) != 4:
    #     raise ValueError("Invalid shape for pred. Expected (N, 4).")
    
    if gt.dim() != 2 or gt.size(1) != 4:
        pass
    
    if pred.dim() != 2 or pred.size(1) != 4:
        pass

    matrix = torchvision.ops.box_iou(gt, pred).numpy()
    max_iou_gt = np.max(matrix, axis=1)
    max_iou_pred = np.max(matrix, axis=0)

    TP = np.sum((max_iou_pred > iou_threshold) * max_iou_pred)
    FP = max_iou_pred.shape[0] - np.sum(max_iou_pred > iou_threshold)
    FN = np.sum(max_iou_gt < iou_threshold)

    return TP / (TP + FN), TP / (TP + FP)

#added regionLabel
def doctr_predictions(directory, regionLabel):
    doc = DocumentFile.from_images(directory)
    result = predictor(doc)
    dic = result.export()
    
    page_dims = [page['dimensions'] for page in dic['pages']]
    
    regions = []
    abs_coords = []
    if regionLabel == 'word':
        regions = [[word for block in page['blocks'] for line in block['lines'] for word in line['words']] for page in dic['pages']]
        abs_coords = [
        [[int(round(word['geometry'][0][0] * dims[1])), 
          int(round(word['geometry'][0][1] * dims[0])), 
          int(round(word['geometry'][1][0] * dims[1])), 
          int(round(word['geometry'][1][1] * dims[0]))] for word in words]
        for words, dims in zip(regions, page_dims)
        ]
    elif regionLabel == 'line':
        regions = [[line for block in page['blocks'] for line in block['lines']] for page in dic['pages']]
        abs_coords = [
        [[int(round(line['geometry'][0][0] * dims[1])), 
          int(round(line['geometry'][0][1] * dims[0])), 
          int(round(line['geometry'][1][0] * dims[1])), 
          int(round(line['geometry'][1][1] * dims[0]))] for line in lines]
        for lines, dims in zip(regions, page_dims)
        ]
    pred = torch.Tensor(abs_coords[0])
    return pred

#generates the jsons for doctr
def generate_jsons(difficulty, dictionary, regiondecision,metadata, language):
    for iou in dictionary[language].keys():
        docvisor_dic = {}
        for image_data in dictionary[language][iou][0]:
            recall, precision, imagePath,ans_key = image_data
            docvisor_dic[ans_key] = dictionary[language][iou][1][ans_key]
        jsonname = difficulty+'_' + language + '_' + str(iou) + "_doctr_" + regiondecision + ".json"
        outfile = open(path_to_save + 'Jsons/RegionData/'+jsonname,"w")
        json.dump(docvisor_dic, outfile, indent = 6)
        outfile.close()
        
        metadata["metaData"]["dataPaths"][difficulty + " " + language + " - iou " + str(iou)] = "Jsons/RegionData/" + jsonname


#stores the sorted list of images based on recall
#image_recall_list = list(recall,precision, document path, pred)
image_recall_list = []

f = open(path_to_groundtruth)
data = json.load(f)
f.close()

dirs = []
for root,d_names,f_names in os.walk(path_to_dataset):
    for f in f_names:
        if(f[-len(ext):] == ext):
            dirs.append(os.path.join(root, f))

print(f"Running for test dataset....")

print(len(dirs))
for c,directory in tqdm(enumerate(dirs)):
    if(c%100 == 0):
        print('iteration number : ', c)
    test_img_path = os.path.basename(directory)#just name of image

    pred = doctr_predictions(directory,regiondecision)

    if(c%100 == 0):
        print('Got preds')

    gt = []
    for i,region in enumerate(data[test_img_path]["polygons"]):
        p1 = region[0]
        p2 = region[2]
        gt.append([p1[0],p1[1],p2[0],p2[1]])
    gt = torch.Tensor(gt)

    if(c%100 == 0):
        print('Got gts')

    iou = iouthresholds[0]
    rec, pre = get_precision_recall(gt,pred,iou) 
    # print(type(gt),type(pred))
    print(gt.size(),pred.size())

    if(c%100 == 0):
        print('Got recall and precision')

    image_recall_list.append((rec,pre,test_img_path,pred.tolist(),gt.tolist()))

    if(c%100 == 0):
        print('appended to list')

print('Saving.... ')
res = ''
if isinstance(args.resume, str):
    file_name = os.path.basename(args.resume)
    exp_name = file_name[:file_name.find("_epoch")]
    res = '_finetuned_' + exp_name
outfile = open(path_to_save + "image_recall_list" + res +".json","w")
json.dump(image_recall_list, outfile, indent = 6)
outfile.close()
print('Done.... ')
