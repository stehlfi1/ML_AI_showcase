
import argparse
from operator import ne
import os
from xml.dom import NamespaceErr
from PIL import Image
import numpy as np
import math
  
  
  
# python3 KUI\ukoly\ukol6\classifier.py -k 3 -o classification.dsv KUI\ukoly\ukol6\train_data KUI\ukoly\ukol6\test_data
# python3 KUI\ukoly\ukol6\classifier.py -b -o classification.dsv KUI\ukoly\ukol6\train_data KUI\ukoly\ukol6\test_data
  
  
# finds all files from folder and puts them into an array
def find_all_files(files, path):
    dsv = None
    for fname in os.listdir(path):
        if "dsv" in fname:
            dsv = fname
        elif ".png" in fname:
            files.append(fname)
    return dsv
  
# finds all files from .dsv file and put them to dictionary
def files_to_dictionary(dsv, path, dictionary, count):
    f = open(path+"/"+dsv, "r")
    for x in f:
        x = str.strip(x)
        x = x.split(":")
        dictionary[x[0]] = x[1]
        if dictionary[x[0]] not in count:
            count[dictionary[x[0]]] = 1
        else:
            count[dictionary[x[0]]] += 1
    f.close()
  
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
  
def probabilities_to_folder(count, histsize, probabilities, imsize):            # function used for debuging
    open("confusion.txt", 'w').close()
    f = open("confusion.txt", "a")
    f.write("\t")
    for char in count:
        f.write("\t"*int((255/histsize)/2)+char + "\t"*int((255/histsize)/2+1))
    f.write("\n")
    f.write("\t")
    for char in count:
        for i in range(int(256/histsize)+1):
            f.write(str(i) + "\t")
    f.write("\n")
  
    for pixel in range(imsize):
        f.write(str(pixel) + "\t")
        for char in count:
            for i in range(int(256/histsize)+1):
                if (char, i, pixel) in probabilities:
                    f.write(str(((probabilities[char, i, pixel]))) + "\t")
                else:
                    f.write(str(0) + "\t")
        f.write("\n")
    f.close()
  
def Bayes(train_path, test_path, o):
    #training 
    files = []
    dsv = find_all_files(files, train_path)
  
    dictionary = dict()
    count = dict()
    files_to_dictionary(dsv, train_path, dictionary, count)
    CharactersProbability = dict()
    sum = 0
    for digit in count:
        sum += count[digit]
    for digit in count:
        CharactersProbability[digit] = count[digit]/sum
    histsize = 128
    charactersHistogram = dict()
    for name in dictionary:
        n = 0
        impath = train_path + '/' + name 
        image_vector = np.array(Image.open(impath)).flatten()
        for element in image_vector:
            if (dictionary[name], int(element/histsize), n) not in charactersHistogram:
                charactersHistogram[dictionary[name], int(element/histsize), n] = 1
            else:
                charactersHistogram[dictionary[name], int(element/histsize), n] += 1
            n += 1
  
    probabilities = dict()
    hyperk = 1
    for char in count:
        for k in range(np.size(image_vector)):
            for shade in range(int(256/histsize)+1):
                if (char, shade, k) not in charactersHistogram:
                    charactersHistogram[char, shade, k] = 1
                else:
                    charactersHistogram[char, shade, k] += 1
  
                if (char, shade, k) in charactersHistogram:
                    probabilities[char, shade, k] = math.log((charactersHistogram[char, shade, k]+hyperk)/((count[char]+1)*np.size(image_vector)+hyperk*(int(256/histsize)+1)))
    #probabilities_to_folder(count, histsize, probabilities, np.size(image_vector))
    files = []
    find_all_files(files, test_path)
    open(o, 'w').close()
    f = open(o, "a")
  
    SuccessRate = [0, 0]
    print("start")
    for file in files:
        impath = test_path + '/' + file
        image_vector = np.array(Image.open(impath)).flatten()
        histogram = dict()
        n=0
        for element in image_vector:
            if (int(element/histsize),n) not in histogram:
                histogram[int(element/histsize),n] = 1
            else:
                histogram[int(element/histsize),n] += 1
            n+=1
        probs = dict()
        for element in range(int(256/histsize)+1):
            for k in range(np.size(image_vector)):
                if (element, k) in histogram:
                    probs[element, k] = math.log((histogram[element,k]+hyperk)/(np.size(image_vector)+hyperk*(int(256/histsize)+1)))
        Best = [np.inf,0]
        for char in count:
            sum = 0
            for element in range(int(256/histsize)+1):
                for i in range(np.size(image_vector)):
                    if (element,i) in probs and (char,element,i) in probabilities:
                        sum += abs((probs[element,i]-probabilities[char,element,i]))
            if sum < Best[0]:
                Best[0] = sum
                Best[1] = char
        
        if dictionary[file] == Best[1]:
            SuccessRate[0] +=1
        else:
            SuccessRate[1] +=1

    
        f.write(file + ":" + str(Best[1]) + "\n")
    f.close()
    print(SuccessRate, ":  ", SuccessRate[0]/(SuccessRate[0]+SuccessRate[1]))
  
def KNN(train_path, test_path, o, k):
    files = []
    dsv = find_all_files(files, train_path)
  
    dictionary = dict()
    count = dict()
    files_to_dictionary(dsv, train_path, dictionary, count)
      
  
    files2 = []
    find_all_files(files2, test_path)
    open(o, 'w').close()
    f = open(o, "a") 
    for file in files2:
        nearest = [np.inf, 0]
        impath2 = test_path + '/' + file
        im2 = np.array(Image.open(impath2)).astype(int).flatten()
        for image in files:
            impath = train_path + '/' + image 
            im1 = np.array(Image.open(impath)).astype(int).flatten()
            distance = np.sqrt(np.sum(np.square(im1-im2)))
            if distance < nearest[0]:
                nearest[0] = distance
                nearest[1] = dictionary[image]
        print(file)
        f.write(file + ":" + str(nearest[1]) + "\n")
    f.close()
  
def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
      
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    if args.k is not None:
        print(f"Running k-NN classifier with k={args.k}")
        KNN(args.train_path, args.test_path, args.o, args.k)
    elif args.b:
        print("Running Naive Bayes classifier")
        Bayes(args.train_path, args.test_path, args.o)
          
      
  
if __name__ == "__main__":
    main()