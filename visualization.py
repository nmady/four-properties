import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import os

def plot_heatmap(data,cmap="afmhot",title=None,vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, figsize=None, display="Show", savepostfix=""):
  ## Code adapted from the charts tutorial to generate the heatmap
  # afmhot, bone, gray, RdBu are good colour map options

  if figsize is not None:
    fig = plt.figure(figsize=figsize,dpi=200, tight_layout=True)
  else:
    fig = plt.figure(dpi=200)
  ax = sns.heatmap(data,cmap=cmap,vmin=vmin,vmax=vmax,square=True, center=0)
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
  return fig, ax


def plot_final_heatmap(data,cmap="afmhot",title=None,vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, figsize=None, display="Show", savepostfix=""):
  fig, ax = plot_heatmap(data, cmap=cmap,title=title,vmin=vmin,vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, display=display, savepostfix=savepostfix)
  
  if display == "Show":
    plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Note that using the replace() method turns spaces into underscores, which
    # appears to be more LaTeX-import friendly.
    plt.savefig("output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+".png")
    plt.close()

def plot_interim_heatmap(data, stepnum, cmap="afmhot",title=None,vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, figsize=None, savepostfix="vid"):
  fig, ax = plot_heatmap(data, cmap=cmap, title=title,vmin=vmin,vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, savepostfix=savepostfix)

  ax.text(9, 0, "t="+str(stepnum))

  dirpath = "output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+"/"
  subdirectory = os.path.dirname("./" + dirpath)
  if not os.path.exists(subdirectory):
    os.makedirs(subdirectory)
  plt.savefig(dirpath+str(stepnum)+".png")
  plt.close()
  

    

def plot_lineplot_data(data, xlabel=None, ylabel=None, title=None, display="Show",savepostfix=""):
  plt.figure(dpi=200)
  ax = sns.lineplot(data=data, y="Value", x="Time", hue="Type")
  sns.despine(ax=ax, offset=1, trim=True)
  if title:    
    plt.title(title)
  if xlabel:
    ax.set_xlabel(xlabel)
  if ylabel:
    ax.set_ylabel(ylabel)
  if display == "Show":
    plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Note that using the replace() method turns spaces into underscores, which
    # appears to be more LaTeX-import friendly.
    plt.savefig("output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+".png")
    plt.close()

def plot_lineplot(x, y, xlabel=None, ylabel=None, title=None, display="Show",savepostfix=""):
  plt.figure(dpi=200)
  ax = sns.lineplot(x=x, y=y)
  sns.despine(ax=ax, offset=1, trim=True)
  if title:    
    plt.title(title)
  if xlabel:
    ax.set_xlabel(xlabel)
  if ylabel:
    ax.set_ylabel(ylabel)
  sns.despine(ax=ax)
  if display == "Show":
    plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Note that using the replace() method turns spaces into underscores, which
    # appears to be more LaTeX-import friendly.
    plt.savefig("output/"+title.replace(" ", "_")
      +"_"+savepostfix.replace(" ", "_")+".png")
    plt.close()

