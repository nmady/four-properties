import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os

def plot_heatmap(data,cmap="afmhot",title=None,vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, display="Show"):
  ## Code adapted from the charts tutorial to generate the heatmap
  # afmhot, bone, gray, RdBu are good colour map options
  plt.figure(dpi=200)
  ax = sns.heatmap(data,cmap=cmap,vmin=vmin,vmax=vmax)
  if target != None:
    rect = patches.Rectangle((target[1],target[0]),1,1,linewidth=2,ls="--",edgecolor='#333333',facecolor='none')  
    ax.add_patch(rect)
  if spawn != None:
    rect = patches.Rectangle((spawn[1],spawn[0]),1,1,linewidth=2,edgecolor='#666666',facecolor='none')  
    ax.add_patch(rect)
  if start != None:
    rect = patches.Rectangle((start[1],start[0]),1,1,linewidth=2,ls="--",edgecolor='#666666',facecolor='none')  
    ax.add_patch(rect) 
  if agent != None:
    rect = patches.Rectangle((agent[1]+0.3,agent[0]+0.3),0.4,0.4,linewidth=1,edgecolor='#333333',facecolor='#999999')  
    ax.add_patch(rect)       
  if title:    
    plt.title(title)
  if display == "Show":
  	plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("output/"+title+".png")
