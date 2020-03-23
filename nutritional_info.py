import numpy as np
import pandas as pd

class NutritionalInformation:
    def __init__(self):
        self.layer1 = pd.read_json('./../data/layer1.json')
        self.nutr_info = pd.read_json("./../data/recipes_with_nutritional_info.json")
        self.nutr_info_p = pd.read_csv("./../data/nutr_info_process.csv")
        self.img_recipe = pd.read_csv('./../data/img_to_recipe.csv')
        self.nonvegan = pd.read_json('./../data/nonvegan.json')
        self.nonpescatarian = pd.read_json('./../data/nonpescatarian.json')
        self.nonvegetarian = pd.read_json('./../data/nonvegetarian.json')
        self.recipe1m = pd.read_csv('./../data/recipes_1m.csv')
        self.nonvegan_ingr = list(self.nonvegan[0])
        self.nonpescatarian_ingr = list(self.nonpescatarian[0])
        self.nonvegetarian_ingr = list(self.nonvegetarian[0]) 
        
    #Gets the recipe data from the passed in image
    def img_to_recipe(self, img):
        recipe_idx = self.img_recipe.loc[self.img_recipe['image_id'] == img]['recipe_id'].values[0]
        recipe_info = self.layer1.loc[self.layer1['id'] == recipe_idx]
        recipe_nutr = self.recipe1m.loc[self.recipe1m['id'] == recipe_idx]
        return recipe_info, recipe_nutr
    
    #Sums up the nutritional information for all the ingredients 
    #to get the nutritional information for the dish as a whole
    #USED FOR PREPROCESSING NUTRITIONAL DATASET
    def get_labels(self, nutr_info):
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

    #Add the nutritional summaries (if they exist) to the recipe_1m dataset
    def get_sum_nutr(recipes_1m):
        recipes_with_nutr = list(nutr_info_process['id'])
        for i in range(recipes_1m.shape[0]):
            if i%1000 == 0:
                print(i)
            rid = recipes_1m['id'][i]
            if rid in recipes_with_nutr:
                recipes_1m['sum_nutr'][i] = nutr_info_process.loc[nutr_info_process['id'] == rid]['sum_nutr'].values[0]
        return recipes_1m
    
    #This function gets labels for the recipes based off of their summed nutritional information
    #USED FOR PREPROCESSING NUTRITIONAL DATASET
    #This function gets labels for the recipes based off of their summed nutritional information
    def nutritional_labels(self, nutr_info):
        all_labels = []

        for i in range(nutr_info.shape[0]):
            if i%1000 == 0:
                print(i)
            labels = []
            nutr_sum = nutr_info['sum_nutr'][i]
            if nutr_sum != None:
                nutr_sum = ast.literal_eval(nutr_sum)
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
                if any(x in j['text'].lower() for x in nonvegan_ingr):
                    vegan = False
                if any(x in j['text'].lower() for x in nonvegetarian_ingr):
                    vegetarian = False
                if any(x in j['text'].lower() for x in nonpescatarian_ingr):
                    pescatarian = False
            if vegan == True:
                labels.append('Vegan')
            if vegetarian == True:
                labels.append('Vegetarian')
            if pescatarian == True:
                labels.append('Pescatarian')
            all_labels.append(labels)
        return all_labels

    #Displays recipes such that they are easily readable to the user
    def display_recipe(self, nutr, info):
        print('\033[36m' + '\033[1m' + '\033[4m' + info['title'].values[0] + '\033[0m')
        print('\033[1m' + "Ingredients:" + '\033[0m')
        ingredients = info['ingredients'].values[0]
        for partial in ingredients:
            print(partial['text'])
        print('\033[1m' + "Instructions:" + '\033[0m')
        instructions = info['instructions'].values[0]
        for inst in instructions:
            print(inst['text'])
        if len(nutr['labels']) > 0:
            print('\033[1m' + "Nutritional Labels" + '\033[0m')
            labels = nutr['labels'].values[0]
            print(labels[1:len(nutr['labels'].values[0])-1])
            #for l in labels:
                #print(l)
