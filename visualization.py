import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import os
import sys

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
dpi = 200

def plot_heatmap(data,
  cmap="afmhot",
  title=None,
  vmin=None, vmax=None,
  target=None, spawn=None, start=None, agent=None, 
  figsize=None, 
  linewidths=None, linecolor=None, cbar=True, 
  xticklabels="auto", yticklabels="auto",
  scaling_constant=2,
  display="Show", savepostfix=""):
  ## Code adapted from the charts tutorial to generate the heatmap
  # afmhot, bone, gray, RdBu are good colour map options

  matrix_height_pt = plt.rcParams["font.size"] * scaling_constant * data.shape[0]
  matrix_height_in = matrix_height_pt / dpi
  matrix_width_pt = plt.rcParams["font.size"] * scaling_constant * data.shape[1]
  matrix_width_in = matrix_width_pt / dpi

  print(matrix_width_in, matrix_height_in)

  fig_bar, ax_bar = plt.subplots(figsize=(0.2,matrix_height_in))
  fig, ax1 = plt.subplots(
    figsize=(matrix_width_in, matrix_height_in),
    gridspec_kw=dict(top=1, bottom=0))
  
  fig.dpi = dpi

  ax = sns.heatmap(data,
    ax=ax1,
    cbar_ax=ax_bar,
    cmap=cmap,
    vmin=vmin,vmax=vmax,
    square=True, 
    linewidths=linewidths, linecolor=linecolor, 
    cbar=cbar,
    xticklabels=xticklabels, yticklabels=yticklabels)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 

  # Expand the axis slightly so that the outer grid lines aren't trimmed
  ax.set_ylim(ymin=float(data.shape[0])+0.1, ymax=-0.1)
  ax.set_xlim(xmin=-0.1, xmax=float(data.shape[1])+0.1)

  # Outline the heatmap
  # ax.axhline(y=0, color='k',linewidth=1)
  # ax.axhline(y=data.shape[0], color='k',linewidth=1)
  # ax.axvline(x=0, color='k',linewidth=1)
  # ax.axvline(x=data.shape[1], color='k',linewidth=1)
  ax.hlines(y=0, color='k', xmin=0, xmax=data.shape[1], linewidth=1)
  ax.hlines(y=data.shape[0], color='k', xmin=0, xmax=data.shape[1], linewidth=1)
  ax.vlines(x=0, color='k', ymin=0, ymax=data.shape[0], linewidth=1)
  ax.vlines(x=data.shape[1], color='k', ymin=0, ymax=data.shape[0], linewidth=1)
  
  if target != None:
    rect = patches.Rectangle((target[1],target[0]),1,1,linewidth=2,ls="--",edgecolor='#333333',facecolor='none')  
    ax.add_patch(rect)
  if spawn != None:
    rect = patches.Rectangle((spawn[1],spawn[0]),1,1,linewidth=2,edgecolor='#666666',facecolor='none')  
    ax.add_patch(rect)
  if start != None:
    rect = patches.Rectangle((start[1],start[0]),1,1,linewidth=1,ls="--",edgecolor='#666666',facecolor='none')  
    ax.add_patch(rect) 
  if agent != None:
    rect = patches.Rectangle((agent[1]+0.3,agent[0]+0.3),0.4,0.4,linewidth=1,edgecolor='#333333',facecolor='#999999')  
    ax.add_patch(rect)       
  if title:    
    plt.title(title)
  return fig, ax, fig_bar


def plot_final_heatmap(data,
  cmap="afmhot",
  title="",
  vmin=None, vmax=None,
  target=None,
  spawn=None,
  start=None, 
  agent=None, 
  figsize=None, 
  linewidths=None, linecolor=None, savebar=True, 
  xticklabels="auto", yticklabels="auto",
  tick_location='bottom',
  scaling_constant=2,
  display="Show", 
  savepostfix=""):
  fig, ax, fig_bar = plot_heatmap(data, 
    cmap=cmap,
    title=title,
    vmin=vmin,vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, 
    figsize=figsize, 
    linewidths=linewidths, linecolor=linecolor, 
    xticklabels=xticklabels, yticklabels=yticklabels,
    scaling_constant=scaling_constant,
    display=display, savepostfix=savepostfix)
  
  if display == "Show":
    plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Note that using the replace() method turns spaces into underscores, which
    # appears to be more LaTeX-import friendly.
    plt.savefig("output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+".png", bbox_inches='tight', dpi=200, transparent=True)
    if savebar:
      fig_bar.savefig("output/bar"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+".png", bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)
    plt.close(fig_bar)

def plot_interim_heatmap(data, stepnum, cmap="afmhot", title=None, 
    vmin=None, vmax=None, target=None, spawn=None,start=None, agent=None, 
    figsize=None, savepostfix="vid"):
  fig, ax = plot_heatmap(data, cmap=cmap, title=title, vmin=vmin, vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, savepostfix=savepostfix)

  ax.text(9, 0, "t="+str(stepnum))

  dirpath = "output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+"/"
  subdirectory = os.path.dirname("./" + dirpath)
  if not os.path.exists(subdirectory):
    os.makedirs(subdirectory)
  plt.savefig(dirpath+str(stepnum)+".png")
  plt.close()
  
def plot_both_value_heatmaps(data1, data2, stepnum, cmap="bwr_r", 
    title=None,vmin=None,vmax=None,
    target=None,spawn=None,start=None, agent=None, 
    figsize=None, savepostfix="vid"):
  if data1.shape[0] > data1.shape[1]:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
  else: 
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

  cbar_ax = fig.add_axes([.91, .3, .03, .4])

  plot_heatmap_to_ax(data1, ax=ax1, cmap=cmap, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, savepostfix=savepostfix)
  plot_heatmap_to_ax(data2, ax=ax2, cmap=cmap, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, savepostfix=savepostfix)

  ax1.text(9, 0, "t="+str(stepnum))

  if title is None:
    title = ""
  else: 
    try:
      ax1.set_title(title[0])
      ax2.set_title(title[1])
      title = title[0] + title[1]
    except TypeError:
      ax1.set_title(title)

  dirpath = "output/"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+"/"
  subdirectory = os.path.dirname("./" + dirpath)
  if not os.path.exists(subdirectory):
    os.makedirs(subdirectory)
  plt.savefig(dirpath+str(stepnum)+".png")
  plt.close()

def plot_heatmap_to_ax(data, ax=None, cmap="afmhot", cbar_ax=True, vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, figsize=None, display="Show", savepostfix=""):
  ## Code adapted from the charts tutorial to generate the heatmap
  # afmhot, bone, gray, RdBu are good colour map options

  if cbar_ax is True:
    ax = sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, square=True, center=0)
  else:
    ax = sns.heatmap(data, ax=ax, cmap=cmap, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, square=True, center=0)
  
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

  return ax
    

def save_lineplot(ax, xlabel=None, ylabel=None, title=None, display="Show",savepostfix=""):
  ax.legend(frameon=False).set_title(None)
  sns.despine(ax=ax, offset=1, trim=True)
  if title:    
    plt.title(title)
  if xlabel:
    ax.set_xlabel(xlabel)
  if ylabel:
    ax.set_ylabel(ylabel)
  ax.figure.tight_layout()
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



