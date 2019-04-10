#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER: Jacob Miller
# DATE CREATED: June 14, 2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Add ability to cycle through models
models = ['resnet', 'alexnet', 'vgg']

for model in models:
    
    # Main program function defined below
    def main():
        # DONE: 1. Define start_time to measure total program runtime by
        # collecting start time
        start_time = time()
        
        # DONE: 2. Define get_input_args() function to create & retrieve command
        # line arguments
        in_arg = get_input_args(model)
        
#        #### DELETE BELOW HERE ####
#        print('---Command line arguments---\n Argument 1: {0}\n Argument 2: {1}\n Argument 3: {2}'.format(in_arg.dir, in_arg.arch, in_arg.dogfile))
#        print()
#        #### DELETE UP TO HERE ####
        
        # DONE: 3. Define get_pet_labels() function to create pet image labels by
        # creating a dictionary with key=filename and value=file label to be used
        # to check the accuracy of the classifier function
        pet_labels_dic = get_pet_labels(in_arg.dir)
        
#        #### DELETE BELOW HERE ####
#        print('---pet_labels_dic---')
#        print(pet_labels_dic)
#        print()
#        #### DELETE UP TO HERE ####
    
        # DONE: 4. Define classify_images() function to create the classifier 
        # labels with the classifier function using in_arg.arch, comparing the 
        # labels, and creating a dictionary of results (results_dic)
        results_dic = classify_images(in_arg.dir, pet_labels_dic, in_arg.arch)
        
#        #### DELETE BELOW HERE ####
#        print('---results dic first---')
#        print(results_dic)
#        print()
#        #### DELETE UP TO HERE ####
        
        # DONE: 5. Define adjust_results4_isadog() function to adjust the results
        # dictionary(results_dic) to determine if classifier correctly classified
        # images as 'a dog' or 'not a dog'. This demonstrates if the model can
        # correctly classify dog images as dogs (regardless of breed)
        adjust_results4_isadog(results_dic, in_arg.dogfile)
        
#        #### DELETE BELOW HERE ####
#        print('---results dic after---')
#        print(results_dic)
#        print()
#        #### DELETE UP TO HERE ####
    
        # DONE: 6. Define calculates_results_stats() function to calculate
        # results of run and puts statistics in a results statistics
        # dictionary (results_stats_dic)
        results_stats = calculates_results_stats(results_dic)
        
#        #### DELETE BELOW HERE ####
#        print('---statistics---')
#        print(results_stats)
#        print()
#        #### DELETE UP TO HERE ####
    
        # DONE: 7. Define print_results() function to print summary results, 
        # incorrect classifications of dogs and breeds if requested.
        print('\n---Print results function---')
        print_results(results_dic, results_stats, in_arg.arch, print_incorrect_dogs = True, print_incorrect_breed = True)
    
    
        # DONE: 1. Define end_time to measure total program runtime
        # by collecting end time
        end_time = time()
    
        # DONE: 1. Define tot_time to computes overall runtime in
        # seconds & prints it in hh:mm:ss format
        tot_time = end_time - start_time
    
        print("\nTotal Elapsed Runtime:", str( f'{int( (tot_time / 3600) ) :02}h' ) + ":" + str( f'{ int (  ( (tot_time % 3600) / 60 ) ) :02}m' ) + ":" + str( f'{int ( ( (tot_time % 3600) % 60 ) ) :02}s' ))
    
    
    
    # DONE: 2.-to-7. Define all the function below. Notice that the input 
    # paramaters and return values have been left in the function's docstrings. 
    # This is to provide guidance for acheiving a solution similar to the 
    # instructor provided solution. Feel free to ignore this guidance as long as 
    # you are able to acheive the desired outcomes with this lab.
    
    def get_input_args(model):
        """
        Retrieves and parses the command line arguments created and defined using
        the argparse module. This function returns these arguments as an
        ArgumentParser object. 
         3 command line arguements are created:
           dir - Path to the pet image files(default- 'pet_images/')
           arch - CNN model architecture to use for image classification(default-
                  pick any of the following vgg, alexnet, resnet)
           dogfile - Text file that contains all labels associated to dogs(default-
                    'dognames.txt'
        Parameters:
         None - simply using argparse module to create & store command line arguments
        Returns:
         parse_args() -data structure that stores the command line arguments object  
        """
        
        # Create Argument Parser object
        parser = argparse.ArgumentParser()
        
        # Argument 1: Create path to folder
        parser.add_argument('--dir', type = str, default = 'pet_images/', help = 'Path to pet_images/')
        
        # Argument 2: Create a CNN model architecture
        parser.add_argument('--arch', type = str, default = model, help = 'Chosen model')
        
        # Argument 3: Create dog text file
        parser.add_argument('--dogfile', type = str, default = 'dognames.txt', help = 'Dog text file')
        
        # Return parser arguments
        return parser.parse_args()
    
    
    def get_pet_labels(image_dir):
        """
        Creates a dictionary of pet labels based upon the filenames of the image 
        files. Reads in pet filenames and extracts the pet image labels from the 
        filenames and returns these label as petlabel_dic. This is used to check 
        the accuracy of the image classifier model.
        Parameters:
         image_dir - The (full) path to the folder of images that are to be
                     classified by pretrained CNN models (string)
        Returns:
         petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                         Labels (as value)  
        """
        
        # Initiate list of filenames in image_dir, initiate dog labels list and pet labels dictionary
        filename_list = listdir(image_dir)
        dog_labels = []
        pet_labels_dic = {}
        
        # Creates pet image labels
        for filename in filename_list:
            last_underscore = 0
            
            for letter in reversed(filename):
                last_underscore += 1
                
                if letter == '_':
                    dog_name = (filename[0:len(filename)-last_underscore]).lower()
                    dog_labels.append((dog_name).replace('_', ' '))
                    break
        
        # Adds filename as key, pet label as index
        for index in range(len(filename_list)):
            pet_labels_dic[filename_list[index]] = dog_labels[index] 
        
        return pet_labels_dic
    
    
    def classify_images(image_dir, pet_labels_dic, model):
        """
        Creates classifier labels with classifier function, compares labels, and 
        creates a dictionary containing both labels and comparison of them to be
        returned.
         PLEASE NOTE: This function uses the classifier() function defined in 
         classifier.py within this function. The proper use of this function is
         in test_classifier.py Please refer to this program prior to using the 
         classifier() function to classify images in this function. 
         Parameters:
          image_dir - The (full) path to the folder of images that are to be
                       classified by pretrained CNN models (string)
          petlabels_dic - Dictionary that contains the pet image(true) labels
                         that classify what's in the image, where its' key is the
                         pet image filename & it's value is pet image label where
                         label is lowercase with space between each word in label 
          model - pretrained CNN whose architecture is indicated by this parameter,
                  values must be: resnet alexnet vgg (string)
         Returns:
          results_dic - Dictionary with key as image filename and value as a List 
                 (index)idx 0 = pet image label (string)
                        idx 1 = classifier label (string)
                        idx 2 = 1/0 (int)   where 1 = match between pet image and 
                        classifer labels and 0 = no match between labels
        """
        
        # Initiate empty dictionary
        results_dic = {}
    
        # Creates image location, classifies image and finds index of starting dog name
        for key in pet_labels_dic.keys():
            image_loc = image_dir + str(key)
            classified_image = classifier(image_loc, model).lower()
            found_index = classified_image.find(pet_labels_dic[key])
            
            # Figures out if match, based on various qualifiers for index/string
            if ( found_index == 0 or classified_image[found_index - 1] == ' ' ) and ( ( ( found_index + len(pet_labels_dic[key]) ) == len(classified_image) ) or ( ( classified_image[found_index + len(pet_labels_dic[key]) + 1] ) == (' ' or ',') ) ):
                match = 1
            else:
                match = 0
            
            results_dic[key] = [pet_labels_dic[key], classified_image, match]
        
        return results_dic
    
    
    def adjust_results4_isadog(results_dic, dogfile):
        """
        Adjusts the results dictionary to determine if classifier correctly 
        classified images 'as a dog' or 'not a dog' especially when not a match. 
        Demonstrates if model architecture correctly classifies dog images even if
        it gets dog breed wrong (not a match).
        Parameters:
          results_dic - Dictionary with key as image filename and value as a List 
                 (index)idx 0 = pet image label (string)
                        idx 1 = classifier label (string)
                        idx 2 = 1/0 (int)  where 1 = match between pet image and 
                                classifer labels and 0 = no match between labels
                        --- where idx 3 & idx 4 are added by this function ---
                        idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                                0 = pet Image 'is-NOT-a' dog. 
                        idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                                'as-a' dog and 0 = Classifier classifies image  
                                'as-NOT-a' dog.
         dogsfile - A text file that contains names of all dogs from ImageNet 
                    1000 labels (used by classifier model) and dog names from
                    the pet image files. This file has one dog name per line
                    dog names are all in lowercase with spaces separating the 
                    distinct words of the dogname. This file should have been
                    passed in as a command line argument. (string - indicates 
                    text file's name)
        Returns:
               None - results_dic is mutable data type so no return needed.
        """           
        
        dognames_dic = {}
        
        # Add dognames to dictionary as key with value '1' (arbitrary number)
        with open('dognames.txt') as file:
            for line in file:
                line = line.strip()
                if line not in dognames_dic:
                    dognames_dic[line] = 1
        
        # Extend results_dic by [0,0], to be changed to 1 if is-a-dog
        for key in results_dic:
            results_dic[key].extend([0,0])
        
        # Change indexes 3 or 4 to '1' if it matches with given dog names
        for key in results_dic:
            for dog in dognames_dic:
                if dog in results_dic[key][0]:
                    results_dic[key][3] = 1
                if dog in results_dic[key][1]:
                    results_dic[key][4] = 1
    
    
    def calculates_results_stats(results_dic):
        """
        Calculates statistics of the results of the run using classifier's model 
        architecture on classifying images. Then puts the results statistics in a 
        dictionary (results_stats) so that it's returned for printing as to help
        the user to determine the 'best' model for classifying images. Note that 
        the statistics calculated as the results are either percentages or counts.
        Parameters:
          results_dic - Dictionary with key as image filename and value as a List 
                 (index)idx 0 = pet image label (string)
                        idx 1 = classifier label (string)
                        idx 2 = 1/0 (int)  where 1 = match between pet image and 
                                classifer labels and 0 = no match between labels
                        idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                                0 = pet Image 'is-NOT-a' dog. 
                        idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                                'as-a' dog and 0 = Classifier classifies image  
                                'as-NOT-a' dog.
        Returns:
         results_stats - Dictionary that contains the results statistics (either a
                         percentage or a count) where the key is the statistic's 
                         name (starting with 'pct' for percentage or 'n' for count)
                         and the value is the statistic's value 
        """
        # Define all variables for statistics
        n_images = len(results_dic)
        n_dogs_img = 0
        n_correct_dogs = 0
        n_notdogs_img = 0
        n_correct_not_dogs = 0
        n_correct_breed = 0
        n_label_matches = 0
        
        # Iterate through results_dic and accumulate statistics
        for key in results_dic:
            
            if results_dic[key][3] == 1:
                n_dogs_img += 1
            
            if results_dic[key][3] == results_dic[key][4] == 1:
                n_correct_dogs += 1
            
            if results_dic[key][3] == 0:
                n_notdogs_img += 1
    
            if results_dic[key][3] == results_dic[key][4] == 0:
                n_correct_not_dogs += 1
            
            if results_dic[key][3] == results_dic[key][2] == 1:
                n_correct_breed += 1
            
            if results_dic[key][2] == 1:
                n_label_matches += 1
        
        # Calculate statistics
        pct_correct_dogs = (n_correct_dogs / n_dogs_img) * 100
        pct_correct_notdogs = (n_correct_not_dogs / n_notdogs_img) * 100
        pct_correct_breed = (n_correct_breed / n_dogs_img) * 100
        pct_label_matches = (n_label_matches / n_images) * 100
        
        results_stats = {'n_images':n_images, 'n_dogs_img':n_dogs_img, 'n_correct_dogs':n_correct_dogs, 'n_notdogs_img':n_notdogs_img, 'n_correct_not_dogs':n_correct_not_dogs, 'n_correct_breed':n_correct_breed, 'n_label_matches':n_label_matches, 'pct_correct_dogs':pct_correct_dogs, 'pct_correct_notdogs':pct_correct_notdogs, 'pct_correct_breed':pct_correct_breed, 'pct_label_matches':pct_label_matches}
        
        return results_stats
        
    
    def print_results(results_dic, results_stats, model, print_incorrect_dogs = False, print_incorrect_breed = False):
        """
        Prints summary results on the classification and then prints incorrectly 
        classified dogs and incorrectly classified dog breeds if user indicates 
        they want those printouts (use non-default values)
        Parameters:
          results_dic - Dictionary with key as image filename and value as a List 
                 (index)idx 0 = pet image label (string)
                        idx 1 = classifier label (string)
                        idx 2 = 1/0 (int)  where 1 = match between pet image and 
                                classifer labels and 0 = no match between labels
                        idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                                0 = pet Image 'is-NOT-a' dog. 
                        idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                                'as-a' dog and 0 = Classifier classifies image  
                                'as-NOT-a' dog.
          results_stats - Dictionary that contains the results statistics (either a
                         percentage or a count) where the key is the statistic's 
                         name (starting with 'pct' for percentage or 'n' for count)
                         and the value is the statistic's value 
          model - pretrained CNN whose architecture is indicated by this parameter,
                  values must be: resnet alexnet vgg (string)
          print_incorrect_dogs - True prints incorrectly classified dog images and 
                                 False doesn't print anything(default) (bool)  
          print_incorrect_breed - True prints incorrectly classified dog breeds and 
                                  False doesn't print anything(default) (bool) 
        Returns:
               None - simply printing results.
        """    
        
        # Print all statistics out
        print('Running classifier using [' + model + '] architecture:\n...')
        print('Number of images: ' + str(results_stats['n_images']))
        print('Number of dog images: ' + str(results_stats['n_dogs_img']))
        print('Number of not dog images: ' + str(results_stats['n_notdogs_img']))
        print('Percent correct dog classification: ' + str(results_stats['pct_correct_dogs']))
        print('Percent correct breed classification: ' + str(results_stats['pct_correct_breed']))
        print('Percent correct not dog classification: ' + str(results_stats['pct_correct_notdogs']))
        print('Percent label matches: ' + str(results_stats['pct_label_matches']))
        
        # Print misclassified dogs if user wants
        if print_incorrect_dogs == True and (results_stats['n_correct_dogs'] + results_stats['n_correct_not_dogs'] != results_stats['n_images']):
            print('\nMisclassified dogs:')
            for key in results_dic:
                if sum(results_dic[key][3:]) == 1:
                    print('Pet image:', results_dic[key][0], '/ Classified image:', results_dic[key][1])
        
        # Print misclassified breeds if user wants
        if print_incorrect_breed == True and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed']):
            print('\nMisclassified breeds:')
            for key in results_dic:
                if sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:
                    print('Pet image:', results_dic[key][0], '/ Classified image:', results_dic[key][1])           
                  
                    
    # Call to main function to run the program
    if __name__ == "__main__":
        main()
