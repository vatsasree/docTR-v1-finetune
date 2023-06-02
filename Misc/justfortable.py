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
            type=str, default='/share3/sreevatsa/docvisor_consortium_gt/NF/docvisor_consortium_gt')
parser.add_argument("--paths", 
            help="Path to save files", 
            type=str, default='./docvisor_saves/Original/')
parser.add_argument("--pathl", 
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


f = open(path_to_reclist)
image_recall_list = json.load(f)
f.close()

print(len(image_recall_list))
print(len(image_recall_list[0]))


image_recall_dict = {}
easy_dict = {}
medium_dict = {}
hard_dict = {}

#loop through language
#   loop through the test list
#       check if item in test list is part of language, if it is only then continue?
#       otherwise just go to next item in test list, so test list will be enumerated language times...

for language in languages:
    image_recall_dict = {}

    f = open(path_to_groundtruth + language + '.json')
    data = json.load(f)
    f.close()

    docvisor_dic = {}
    precision = []
    recall = []

    print(f"Running for language:{language}")

    for c, item in tqdm(enumerate(image_recall_list)):
        #compare the test_image_path with the ground truth 
        rec,pre,test_img_path,pred_list,gt_list = item
        ans_key = ""
        image_keys = list(data.keys())
        # print(test_img_path)

        for k in image_keys:
            path = os.path.basename(data[k]["imagePath"][30:]) #language/*/*....
            # print(path)
            if path == test_img_path:
                ans_key = k
                break
        
        if(ans_key == ""):
            continue
        
        docvisor_dic[ans_key] = {}

        if(not test):
            docvisor_dic[ans_key]["imagePath"] = data[ans_key]["imagePath"]
        else:
            docvisor_dic[ans_key]["imagePath"] = '/home/abhayram/Desktop/DocTR/docTR/' + data[ans_key]["imagePath"][11:]

        regions = []
        for i,region in enumerate(data[ans_key]["regions"]):
            if region["regionLabel"] == regiondecision:
                regions.append(region)
        docvisor_dic[ans_key]["regions"] = regions


        pred = pred_list

        gt = []
        for i,region in enumerate(docvisor_dic[ans_key]["regions"]):
            p1 = region["groundTruth"][0]
            p2 = region["groundTruth"][2]
            gt.append([int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1])])
        gt = torch.Tensor(gt)

        for p in pred:
            region = {}
            region["outputs"] = {}
            region["regionLabel"] = regiondecision
            region["id"] = ans_key
            # p = p.to(torch.int).tolist()
            region["groundTruth"] = [[int(p[0]),int(p[1])]]
            region["modelPrediction"] = [[int(p[0]),int(p[1])],[int(p[0]),int(p[3])],[int(p[2]),int(p[3])],[int(p[2]),int(p[1])]]
            docvisor_dic[ans_key]["regions"].append(region)
        
        
        average_recall = 0
        average_precision = 0
        
        if len(gt) == 0 or len(pred) == 0:
            # print(docvisor_dic[ans_key]["imagePath"])
            average_recall = 0
            average_precision = 0
        else:
            average_recall = 0
            average_precision = 0
            
            for iou in iouthresholds:
                average_recall += rec
                average_precision += pre
                if iou not in image_recall_dict.keys():
                    image_recall_dict[iou] = [(rec,pre,docvisor_dic[ans_key]["imagePath"],ans_key)]
                else:
                    image_recall_dict[iou].append((rec,pre,docvisor_dic[ans_key]["imagePath"],ans_key))
            
            average_recall /= len(iouthresholds)
            average_precision /= len(iouthresholds)

            
        docvisor_dic[ans_key]["metrics"] = {"precison":average_precision, "recall": average_recall}
     
    # docvisor_dic_all_languages[language] = docvisor_dic



# for language in languages:
    easy_dict[language] = {}
    medium_dict[language] = {}
    hard_dict[language] = {}

    for iou in iouthresholds:        
        image_recall_dict[iou].sort(key = lambda x : x[0])
        n = len(image_recall_dict[iou])
        hard_dict[language][iou] = {}
        easy_dict[language][iou] = {}
        medium_dict[language][iou] = {}

        hard_dict[language][iou][0] = image_recall_dict[iou][:K]
        medium_dict[language][iou][0] = image_recall_dict[iou][n//2-K//2:n//2-K//2+K]
        easy_dict[language][iou][0] = image_recall_dict[iou][-K:]
        hard_dict[language][iou][1] = {}
        medium_dict[language][iou][1] = {}
        easy_dict[language][iou][1] = {}
        for ii in range(K):
            ans_key_e = easy_dict[language][iou][0][ii][3] #gets answer key
            ans_key_m = medium_dict[language][iou][0][ii][3]
            ans_key_h = hard_dict[language][iou][0][ii][3]
            easy_dict[language][iou][1][ans_key_e] = {}
            easy_dict[language][iou][1][ans_key_e]["imagePath"] = docvisor_dic[ans_key_e]["imagePath"]
            easy_dict[language][iou][1][ans_key_e]["regions"] = docvisor_dic[ans_key_e]["regions"]
            easy_dict[language][iou][1][ans_key_e]["metrics"] = {"precison":easy_dict[language][iou][0][ii][1], "recall": easy_dict[language][iou][0][ii][0]}
            medium_dict[language][iou][1][ans_key_m] = {}
            medium_dict[language][iou][1][ans_key_m]["imagePath"] = docvisor_dic[ans_key_m]["imagePath"]
            medium_dict[language][iou][1][ans_key_m]["regions"] = docvisor_dic[ans_key_m]["regions"]
            medium_dict[language][iou][1][ans_key_m]["metrics"] = {"precison":medium_dict[language][iou][0][ii][1], "recall": medium_dict[language][iou][0][ii][0]}
            hard_dict[language][iou][1][ans_key_h] = {}
            hard_dict[language][iou][1][ans_key_h]["imagePath"] = docvisor_dic[ans_key_h]["imagePath"]
            hard_dict[language][iou][1][ans_key_h]["regions"] = docvisor_dic[ans_key_h]["regions"]
            hard_dict[language][iou][1][ans_key_h]["metrics"] = {"precison":hard_dict[language][iou][0][ii][1], "recall": hard_dict[language][iou][0][ii][0]}



##############################################################
#Tabulating the data
##############################################################

datatabulate = []
#Not doing language in languages because certain languages might not be taken in current run
for i,language in enumerate(easy_dict.keys()):
    datatabulate.append([language])
    
    avgrecall = 0
    avgprecision = 0
    for imagedata in easy_dict[language][iouthresholds[0]][0]:
        recall, precision, imagePath,ans_key = imagedata
        avgrecall += recall
        avgprecision += precision
    avgrecall /= K
    avgprecision /= K
    
    datatabulate[i].append((round(avgrecall,5),round(avgprecision,5)))
    
    avgrecall = 0
    avgprecision = 0
    for imagedata in medium_dict[language][iouthresholds[0]][0]:
        recall, precision, imagePath,ans_key = imagedata
        avgrecall += recall
        avgprecision += precision
    avgrecall /= K
    avgprecision /= K
    
    datatabulate[i].append((round(avgrecall,5),round(avgprecision,5)))
    
    avgrecall = 0
    avgprecision = 0
    for imagedata in hard_dict[language][iouthresholds[0]][0]:
        recall, precision, imagePath,ans_key = imagedata
        avgrecall += recall
        avgprecision += precision
    avgrecall /= K
    avgprecision /= K
    
    datatabulate[i].append((round(avgrecall,5),round(avgprecision,5)))
    

#define header names
col_names = ["Language", "Easy","Medium","Hard"]
print()
#display table
print(tabulate(datatabulate, headers=col_names))