import numpy as np
import os
from img_encoder import ModelType

class EncodingLoader:
    
    def __init__(self, model_type=ModelType.ResNet50, num_encodings=97):
        enc_dir = './encodings'
        if model_type != ModelType.ResNet50:
            enc_dir += str(model_type)[-3:]
        self.enc_dir = enc_dir
        self.num_encodings = num_encodings
        self.cur_file = 0
        
    def reset_iter(self):
        self.cur_file = 0
    
    def load_encodings_from_files(self):
        enc_dict = dict()
        for filename in os.listdir(self.enc_dir):
            if filename.endswith(".npy"): 
                file = np.load(self.enc_dir + '/' + filename, allow_pickle=True)
                cur_dict = file[()]
                enc_dict.update(cur_dict)
        return enc_dict
    
    def load_next_encoding(self, as_dict=False):
        if self.cur_file == self.num_encodings:
            self.cur_file = 0
            return None
        filename = 'enc_' + str(self.cur_file) + '.npy'
        file = np.load(self.enc_dir + '/' + filename, allow_pickle=True)
        enc_dict = file[()]
        if as_dict:
            self.cur_file += 1
            return enc_dict
        length = enc_dict[list(enc_dict.keys())[0]].shape[0]
        enc_matrix = np.zeros((len(enc_dict), length))
        index_to_filename = dict()
        idx = 0
        for img_file, encoding in enc_dict.items():
            index_to_filename[idx] = img_file
            enc_matrix[idx] = encoding
            idx += 1
        self.cur_file += 1
        return enc_matrix, index_to_filename
    
    def load_encodings_from_file(self, filename, as_dict=False):
        file = np.load(self.enc_dir + '/' + filename, allow_pickle=True)
        enc_dict = file[()]
        if as_dict:
            return enc_dict
        length = enc_dict[list(enc_dict.keys())[0]].shape[0]
        enc_matrix = np.zeros((len(enc_dict), length))
        index_to_filename = dict()
        idx = 0
        for img_file, encoding in enc_dict.items():
            index_to_filename[idx] = img_file
            enc_matrix[idx] = encoding
            idx += 1
        return enc_matrix, index_to_filename

