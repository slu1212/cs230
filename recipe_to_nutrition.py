import numpy as np
import pandas as pd


def img_to_recipe(img):
    recipe_idx = img_recipe.loc[img_recipe['image_id'] == img]['recipe_id'].values[0]
    recipe_info = layer1.loc[layer1['id'] == recipe_idx]
    recipe_nutr = nutr_info_p.loc[nutr_info_p['id'] == recipe_idx]
    return recipe_info, recipe_nutr

#Sums up the nutritional information for all the ingredients 
#to get the nutritional information for the dish as a whole
def get_labels(nutr_info):
    sum_info = []
    for r in range(0, len(nutr_info)):
        nutr_sum = dict()
        all_keys = set()
        nutr = nutr_info['nutr_per_ingredient'][r]
        for i in range(len(nutr)):
            all_keys = all_keys.union(set(nutr[i].keys()))
        for k in all_keys:
            nutr_sum[k] = 0
            for idx in range(len(nutr)):
                if k in nutr[idx].keys():
                    nutr_sum[k] += nutr[idx][k]
        sum_info.append(nutr_sum)
    return sum_info[::-1]

#This function gets labels for the recipes based off of their summed nutritional information
def nutritional_labels(nutr_info):
    all_labels = []
    for i in range(nutr_info.shape[0]):
        labels = []
        nutr_sum = nutr_info['sum_nutr'][i]
        if 'fat' in nutr_sum.keys() and nutr_sum['fat'] < 500/9:
            labels.append('<500 Calories From Fat')
        if 'sod' in nutr_sum.keys() and nutr_sum['sod'] < 3000:
            labels.append('Low Sodium Meal')
        if 'pro' in nutr_sum.keys() and nutr_sum['pro'] >= 20 and nutr_sum['pro'] <= 30:
            labels.append('Protein Rich')
        if 'pro' in nutr_sum.keys() and nutr_sum['pro'] > 30:
            labels.append('Protein Heavy')
        ingredients = nutr_info['ingredients'][i]
        vegan = True
        vegetarian = True
        pescatarian = True
        kosher = True
        halal = True
        for j in ingredients:
            if any(x in j['text'] for x in nonvegan_ingr):
                vegan = False
            if any (x in j['text'] for x in nonvegetarian_ingr):
                vegetarian = False
            if any (x in j['text'] for x in nonpescatarian_ingr):
                pescatarian = False
        if vegan == True:
            labels.append('Vegan')
        if vegetarian == True:
            labels.append('Vegetarian')
        if pescatarian == True:
            labels.append('Pescatarian')
        all_labels.append(labels)
    return all_labels

#Displays recipes such that they are readable to the user
def display_recipe(info, nutr):
    print(color.DARKCYAN + color.BOLD + color.UNDERLINE + info['title'].values[0] + color.END)
    print(color.BOLD + "Ingredients:" + color.END)
    ingredients = info['ingredients'].values[0]
    for partial in ingredients:
        print(partial['text'])
    print(color.BOLD + "Instructions:" + color.END)
    instructions = info['instructions'].values[0]
    for inst in instructions:
        print(inst['text'])
    if len(nutr) != 0:
        print(color.BOLD + "Nutritional Labels" + color.END)
        labels = nutr['labels'].values[0]
        for l in labels:
            print(l)
    
