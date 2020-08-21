# Import required packages

from tkinter import filedialog
import tkinter as tk
import pickle
import pandas as pd
pd.options.display.max_colwidth = 100
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()



root = tk.Tk()
#
fileN = filedialog.askopenfilename(parent=root, title='Select Pickle file',
                                   filetypes=(("Pickle files", "*.pickle"), ("all files", "*.*")))
root.withdraw()

#This code section allows you to select the piokle file
if fileN is not None:
    with open(fileN, 'rb') as f:
        # Pickling is a method of converting python objects into a byte stream, making saving large amounts of data much faster
        # This is the data I have already processed and stored in the pickle
        all_data = pickle.load(f)
        EMG = all_data['EMG']
        all_EMGs = all_data['EMG_Cycles']
        peak_onset_EMGs = all_data['EMG_Peak_Onset']
        # FK_processed = all_data['Processed_Forward_Kinematics']
        # FK_range = all_data['Forward_Kinematic_Ranges']


#This is an example of how to use a boolean array to index a part of your dataframe.
bool = (all_EMGs.loc[:,'Side'] == all_EMGs.loc[:,'CutSide'])
graph_EMGS = all_EMGs.loc[bool.values]

boolP = (peak_onset_EMGs.loc[:,'Side'] == peak_onset_EMGs.loc[:,'CutSide'])
graph_EMGSP = peak_onset_EMGs.loc[boolP.values]
graph_EMGSP['Peak'] = graph_EMGSP['Peak'].astype('float64')
graph_EMGSP['Onset'] = graph_EMGSP['Onset'].astype('float64')

#Here is where you can try out different seaborn plots with the data that I have. Continous data is contained in the
#"graph_EMGS" dataframe, and discrete data is contained in the "graph_EMGSP" dataframe


# sns.relplot(y='Normalized EMG', x='x', hue='Trial', style='Muscle', col='Side', data=graph_EMGS, kind='line')

# sns.catplot(x = 'Trial', y = 'Peak', col='Muscle', row='Side', data=graph_EMGSP, kind='violin')