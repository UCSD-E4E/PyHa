#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *


# # Microfaune + PyHa Experiments

# In[3]:


#path_to_audio_files = "/home/jacob/acoustic-species-id/Mixed_Bird/"
#path_to_ground_truth = "MDD_Xeno_Canto_DSC180_Labels_uniform_3s_binary.csv"
path_to_audio_files = "./mixed_bird/Mixed_Bird/"
path_to_ground_truth = "./mixed_bird/mixed_bird_manual.csv"

# Parameters to define isolation behavior
isolation_parameters_micro = {
    "model" : "microfaune",
    "technique" : "chunk",
    "threshold_type" : "median",
    "threshold_const" : 4.0,
    "threshold_min" : 0.25,
    "window_size" : 2.0,
    "chunk_size" : 3.0
}


# In[4]:


automated_df_micro, local_scores_micro = generate_automated_labels(path_to_audio_files,isolation_parameters_micro);


# In[5]:


#automated_df_micro.to_csv("MDD_Xeno_Canto_Microfaune_Chunk_median_025_400.csv",index=False)
automated_df_micro.to_csv("Piha_Microfaune_Chunk_median_025_400.csv",index=False)


# In[6]:


manual_df = pd.read_csv(path_to_ground_truth)
# Removing the segments that are not 3s long (the ones at the end of clips)
automated_df_micro = automated_df_micro[automated_df_micro["DURATION"]==3.0]
manual_df


# In[7]:


statistics_df_micro = automated_labeling_statistics(automated_df_micro,manual_df,stats_type = "general");
#statistics_df.to_csv("MDD_Xeno_Canto_DSC180_Stats_9.csv",index=False)


# In[8]:


global_dataset_statistics(statistics_df_micro)


# # BirdNET-Lite Experiments

# In[9]:


# loading in the manual annotations again in case someone wants to play with BirdNET without playing with Microfaune
#path_to_ground_truth = "MDD_Xeno_Canto_DSC180_Labels_uniform_3s_binary.csv"
#path_to_ground_truth = "ScreamingPiha_Manual_Labels.csv"
manual_df = pd.read_csv(path_to_ground_truth)

isolation_parameters_birdnet = {
   "model" : "birdnet",
   "output_path" : "outputs",
   "filetype" : "wav", 
   "num_predictions" : 1,
   "write_to_csv" : True
}


#birdnet_labels = pd.read_csv("birdnet_experiments/experiment1.csv")


# In[1]:


automated_df_birdnet, local_scores_birdnet = generate_automated_labels(path_to_audio_files,isolation_parameters_birdnet);


# In[ ]:


#automated_df_micro.to_csv("MDD_Xeno_Canto_Microfaune_Chunk_median_025_400.csv",index=False)
birdnet_labels = automated_df_birdnet
automated_df_birdnet.to_csv("Piha_BirdNET_Chunk.csv",index=False)


# In[ ]:


statistics_df_birdnet = automated_labeling_statistics(birdnet_labels,manual_df,stats_type = "general");


# In[ ]:


statistics_df_birdnet


# In[ ]:


global_dataset_statistics(statistics_df_birdnet)


# In[ ]:


isolation_parameters_birdnet_conf = {
   "model" : "birdnet",
   "output_path" : "outputs",
   "filetype" : "wav", 
   "num_predictions" : 1,
   "write_to_csv" : True
}
automated_df_birdnet_conf = generate_automated_labels(path_to_audio_files,isolation_parameters_birdnet_conf);
statistics_df_conf = automated_labeling_statistics(automated_df_birdnet_conf,manual_df,stats_type = "general");


# In[ ]:


global_dataset_statistics(statistics_df_conf)


# # Tweetynet Experiments

# In[ ]:


# loading in the manual annotations again in case someone wants to play with TweetyNET without playing with Microfaune
#path_to_ground_truth = "MDD_Xeno_Canto_DSC180_Labels_uniform_3s_binary.csv"
#path_to_ground_truth = "ScreamingPiha_Manual_Labels.csv"
manual_df = pd.read_csv(path_to_ground_truth)

isolation_parameters_tweety = {
   "model" : "tweetynet",
    "tweety_output" : True,
    "chunk_size" : 3.0
}


# In[ ]:


automated_df_tweety, local_scores_tweety = generate_automated_labels(path_to_audio_files,isolation_parameters_tweety);


# In[ ]:


#automated_df_micro.to_csv("MDD_Xeno_Canto_Microfaune_Chunk_median_025_400.csv",index=False)
automated_df_tweety.to_csv("Piha_TweetyNET_Chunk.csv",index=False)


# In[ ]:


statistics_df_tweety = automated_labeling_statistics(birdnet_labels,manual_df,stats_type = "general");


# In[ ]:


statistics_df_tweety


# In[ ]:


global_dataset_statistics(statistics_df_tweety)


# # Model Comparison

# In[ ]:


manual_df.style.set_caption("Manual Annotations");
automated_df_micro.style.set_caption("Microfaune Annotations");
automated_df_birdnet.style.set_caption("BirdNET Annotations");
automated_df_tweety.style.set_caption("TweetyNET Annotations");


# In[ ]:


display(global_dataset_statistics(statistics_df_micro).style.set_caption("Microfaune Annotations"))
display(global_dataset_statistics(statistics_df_birdnet).style.set_caption("BirdNET Annotations"))
display(global_dataset_statistics(statistics_df_tweety).style.set_caption("TweetyNET Annotations"))


# In[ ]:


display(annotation_duration_statistics(manual_df).style.set_caption("Manual Annotations"))
display(annotation_duration_statistics(automated_df_micro).style.set_caption("Microfaune Annotations"))
display(annotation_duration_statistics(automated_df_birdnet).style.set_caption("BirdNET Annotations"))
display(annotation_duration_statistics(automated_df_tweety).style.set_caption("TweetyNET Annotations"))


# In[ ]:


generate_ROC_curves(automated_df_micro, manual_df, local_scores_micro, chunk_length = 3)
generate_ROC_curves(automated_df_birdnet, manual_df, local_scores_birdnet, chunk_length = 3)
generate_ROC_curves(automated_df_tweety, manual_df, local_scores_tweety, chunk_length = 3)


# In[ ]:




