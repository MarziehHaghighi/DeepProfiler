import comet_ml
import importlib
import os
import numpy as np
import pandas
import glob
import skimage.transform
import pickle

import tensorflow as tf
#from tensorflow.contrib import slim
from keras import backend as K

import deepprofiler.learning.training
import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping
from deepprofiler.dataset.utils import tic, toc

import keras
from keras.models import Model


class testingmodel(object):
    
    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["prepare"]["images"]["channels"])
        self.crop_generator = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))\
            .GeneratorClass
        self.profile_crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
            .SingleImageGeneratorClass
        self.dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))\
            .ModelClass(config, dset, self.crop_generator, self.profile_crop_generator)
        self.profile_crop_generator = self.profile_crop_generator(config, dset)


    def configure(self):        
        # Main session configuration
        configuration = tf.ConfigProto()
        configuration.gpu_options.visible_device_list = self.config["testingmodel"]["gpus"]
        configuration.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configuration)
        self.profile_crop_generator.start(self.sess)
        K.set_session(self.sess)
        self.config["testingmodel"]["feature_layer"]=self.config["train"]["dset"]["targets"][0]
        # Create feature extractor
        if self.config["train"]["pretrained"]:
            checkpoint = self.config["paths"]["pretrained"]+"/"+self.config["testingmodel"]["checkpoint"]
        else:
            checkpoint = self.config["paths"]["checkpoints"]+"/"+self.config["testingmodel"]["checkpoint"]
        
        self.dpmodel.feature_model.load_weights(checkpoint)
        self.feat_extractor = keras.Model(self.dpmodel.feature_model.inputs, self.dpmodel.feature_model.get_layer(
            self.config["testingmodel"]["feature_layer"]).output)
        self.feat_extractor.summary()


    def check(self, meta):
        output_folder = self.config["paths"]["features"]
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
#             print("Already done but overwriting:", output_file)
#             return False    commented by marzieh
            return False
        else:
            return True
    
    # Function to process a single image
    def extract_features(self, key, image_array, meta):  # key is a placeholder
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        print('Hellowwww')
        batch_size = self.config["testingmodel"]["batch_size"]
        image_key, image_names, outlines = self.dset.getImagePaths(meta)
#         print(meta)
        total_crops = self.profile_crop_generator.prepare_image(
                                   self.sess,
                                   image_array,
                                   meta,
                                   False
                            )
#         print('Hereeeee',total_crops,meta)
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return
        num_features = self.config["train"]["model"]["params"]["feature_dim"]
        repeats = "channel_repeats" in self.config["prepare"]["images"].keys()
#         print('here2',self.config["prepare"]["images"].keys())
        # Extract features
        crops = next(self.profile_crop_generator.generate(self.sess))[0]  # single image crop generator yields one batch
#         print('crop shape',crops.shape)
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)
        if repeats:
            feats = np.reshape(feats, (self.num_channels, total_crops, num_features))
            feats = np.concatenate(feats, axis=-1)
            
        # Save features
        np.savez_compressed(output_file, f=feats)
        toc(image_key + " (" + str(total_crops) + " cells)", start)

        
def testmodel(config, dset):
    profile = testingmodel(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="test", check=profile.check)
    print("Generate Test Set Scores: done")
