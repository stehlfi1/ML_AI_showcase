import argparse
from PIL import Image
import numpy as np
import os
import math
import time

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument('-k', type=int, 
                             help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    mutex_group.add_argument("-b", 
                             help="run Naive Bayes classifier", action="store_true")
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def find_files(path, files):
    dsv = None

    for filename in os.listdir(path):
        
        if ".png" in filename:
            files.append(filename)
        elif "dsv" in filename:
            dsv = filename
    
    return dsv

def files_to_dic(path, dsv, dic, counter):
    f = open(path+"/"+dsv, "r")

    for x in f: 
        x = str.strip(x)   
        x = x.split(":")   
        dic[x[0]] = x[1]   
        if dic[x[0]] not in counter:  
            counter[dic[x[0]]] = 1    
        else:               
            counter[dic[x[0]]] += 1 

    f.close()

def KNN(train_path, test_path, o, k):
    train_files = []
    dsv = find_files(train_path, train_files)

    counter = dict()
    dic = dict()
    
    files_to_dic(train_path, dsv, dic, counter)

    test_files = []
    find_files(test_path, test_files)
    open(o, "w").close()
    f = open(o, "a")

    for x in test_files:
        test_image_path = test_path + '/' + x
        im2 = np.array(Image.open(test_image_path)).astype(int).flatten()
        closest = [math.inf, 0]
        for image in train_files:
            train_image_path = train_path + '/' + image
            im1 = np.array(Image.open(train_image_path)).astype(int).flatten()
            diff = im1 - im2
            d1 = np.sqrt(np.sum(np.square(diff)))
            if  closest[0] > d1:
                closest[0] = d1
                closest[1] = dic[image]
        
        f.write(x + ":" + str(closest[1] + "\n"))
        #print(x)
    f.close()
    
#-b -o cls.dsv ./train_700_28 ./train_700_28

def Bayes(train_path, test_path, o):
    
    train_files = []
    dsv = find_files(train_path, train_files)
    train_data = dict()
    counter = dict()
    Org_data = dict()
    files_to_dic(train_path, dsv, train_data, counter)
    print("training Bayes")
   
    divisor = 64 #how many segments of shade rn 6

    for name in train_data:
        
        test_image_path = train_path + '/' + name
        vector = np.array(Image.open(test_image_path)).flatten()
        idx = 0
        for pixel in vector:
            tmp_pixel = int(pixel/divisor)
            
            if(idx, train_data[name], tmp_pixel) in Org_data:
                Org_data[idx, train_data[name], tmp_pixel] += 1
            else:
                Org_data[idx, train_data[name], tmp_pixel] = 1
            idx += 1
    
    vector_size = np.size(vector)      
    Prob_table = dict()
    segment = int(256/divisor)+1 
    for num in counter:
        for i in range (np.size(vector)):
            for shade in range(segment):
                
                if(i, num, shade) in Org_data:
                    Org_data[i, num, shade] += 1
                else:
                    Org_data[i, num, shade] = 1
                
                if(i, num, shade) in Org_data:
                    Prob_table[i, num, shade] = math.log((Org_data[i, num, shade])/((counter[num]+1)*np.size(vector)+segment))

    print("testing")      
    test_files = []
    succ = 0
    fail = 0
    find_files(test_path, test_files)
    open(o, "w").close()
    f = open(o, "a")

    for x in test_files:
        #print(x)
        test_image_path = test_path + '/' + x
        vector_t = np.array(Image.open(test_image_path)).astype(int).flatten()
        test_nums = dict()
        probs = dict()
        best_match = math.inf
        best_match_num = 0
        idx = 0
        for shade in vector_t:
            if(idx, int(shade/divisor)) in test_nums:
                test_nums[idx, int(shade/divisor)] += 1
            else:
                test_nums[idx, int(shade/divisor)] = 1
            idx += 1
        
        for i in range (vector_size):    
            for shade in range(segment):
                if(i, shade) in test_nums:
                    probs[i, shade] = math.log((test_nums[i, shade])/(vector_size+segment)) 
        for nums in counter:
            sum = 0
            for shade in range (segment):
                for i in range(vector_size):
                    if (i, nums, shade) in Prob_table and (i, shade) in probs:
                        sum += abs((Prob_table[i,nums, shade]-probs[i, shade]))
           
            if best_match > sum:
                best_match_num = nums
                best_match = sum

        '''    
        if(train_data)[x] == best_match_num:
            succ +=1
        else:
            fail +=1
        '''
        f.write(x + ":" + str(best_match_num + "\n"))
    print("succes: ", succ, "fail:", fail)
    f.close()

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    if args.k is not None:
        print(f"Running k-NN classifier with k={args.k}")
        KNN( args.train_path, args.test_path, args.o, args.k)
    elif args.b:
        print("Running Naive Bayes classifier")
        Bayes( args.train_path, args.test_path, args.o)

if __name__ == "__main__":
    main()

    
