import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import os
import sys
import matplotlib.ticker

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 28
dpi = 200

def plot_heatmap(data,
  cmap="afmhot",
  title=None,
  vmin=None, vmax=None,
  target=None, spawn=None, start=None, agent=None, 
  figsize=None, 
  linewidths=None, linecolor=None, 
  outline=False,
  cbar=True, 
  cbar_kws={},
  norm=None,
  xticklabels="auto", yticklabels="auto",
  scaling_constant=2,
  display="Show", savepostfix=""):
  ## Code adapted from the charts tutorial to generate the heatmap
  # afmhot, bone, gray, RdBu are good colour map options

  if figsize is None:

    matrix_height_pt = plt.rcParams["font.size"] * scaling_constant * data.shape[0]
    matrix_height_in = matrix_height_pt / dpi
    matrix_width_pt = plt.rcParams["font.size"] * scaling_constant * data.shape[1]
    matrix_width_in = matrix_width_pt / dpi

    fig_bar, ax_bar = plt.subplots(figsize=(0.2,matrix_height_in))
    fig, ax1 = plt.subplots(
      figsize=(matrix_width_in, matrix_height_in),
      gridspec_kw=dict(top=1, bottom=0))

  else:
    fig_bar, ax_bar = plt.subplots(figsize=(0.2, figsize[1]))
    fig, ax1 = plt.subplots(figsize=figsize, gridspec_kw=dict(top=1, bottom=0))
  
  fig.dpi = dpi
  
  if norm is None:
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

  ax = sns.heatmap(data,
    ax=ax1,
    cbar_ax=ax_bar,
    cmap=cmap,
    vmin=vmin,vmax=vmax,
    square=True, 
    linewidths=linewidths, linecolor=linecolor, 
    cbar=cbar,
    cbar_kws=cbar_kws,
    norm=norm,
    xticklabels=xticklabels, yticklabels=yticklabels
    )

  ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 

  # Expand the axis slightly so that the outer grid lines aren't trimmed
  ax.set_ylim(ymin=float(data.shape[0])+0.1, ymax=-0.1)
  ax.set_xlim(xmin=-0.1, xmax=float(data.shape[1])+0.1)

  # Outline the heatmap
  if outline:
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
  return fig, ax, fig_bar, ax_bar


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
  count=False,
  logscale=False,
  xticklabels="auto", yticklabels="auto",
  tick_location='bottom',
  scaling_constant=2,
  display="Show", 
  savepostfix=""):
  '''
    Args: 
      data (rectangular dataset): from seaborn.heatmap
        2D dataset that can be coerced into an ndarray. 
        If a Pandas DataFrame is provided, the index/column information will be 
        used to label the columns and rows.

      count (boolean): set to True to ensure the colorbar only shows integer
        labels, e.g. if the values are counts they shouldn't be fractional
  '''

  norm = None
  if logscale: 
    assert(vmin is None or vmin > 0)
    if cmap != 'bone':
      cmap = sns.light_palette(sns.color_palette(cmap)[-1], as_cmap=True)
      background = 'white'
    else:
      vmin = 0.5
      background = 'k'
    if vmax is not None:
      norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

  if count:
    if vmax:
      n = int(vmax) + 1
    else:
      tempax = sns.heatmap(data, cmap=cmap, vmax=vmax)
      # https://stackoverflow.com/questions/19816820/how-to-retrieve-colorbar-instance-from-figure-in-matplotlib
      colorbar = tempax.collections[-1].colorbar
      n = int(colorbar.vmax)
    cmap = sns.color_palette(cmap, n)

  fig, ax, fig_bar, ax_bar = plot_heatmap(data, 
    cmap=cmap,
    title=title,
    vmin=vmin,vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, 
    figsize=figsize, 
    linewidths=linewidths, linecolor=linecolor, 
    xticklabels=xticklabels, yticklabels=yticklabels,
    norm=norm,
    scaling_constant=scaling_constant,
    display=display, savepostfix=savepostfix)

  if logscale:
    ax.set_facecolor(background)
    fig.patch.set_facecolor('#FFFFFF00')
    if background == 'k':
      ax_bar.set_yticks([1,10,100,1000,5000], labels=[1,10,100,1000,5000])

  # Discretize the colormap if heatmap represents counts
  if count and not logscale:
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    # The list comprehension calculates the positions to evenly distribute the 
    # labels across the colorbar
    tick_positions = [colorbar.vmin + r * (i + 0.5) / (n) for i in range(n)]
    tick_labels = [str(num) for num in range(n)]
    colorbar.set_ticks(tick_positions, labels=tick_labels)
    for number, tick in zip(range(n-1), ax_bar.yaxis.get_major_ticks()):
      if n > 5 and number%(n//5) != 0:
        tick.set_visible(False)
  
  if display == "Show":
    plt.show()
  elif display == "Save":
    file_path = "./output/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Note that using the replace() method turns spaces into underscores, which
    # appears to be more LaTeX-import friendly.
    plt.savefig("output/" + title.replace(" ", "_") + "_" 
      + savepostfix.replace(" ", "_") + ".png", bbox_inches='tight', dpi=200)#, transparent=True)
    if savebar:
      fig_bar.savefig("output/bar"+title.replace(" ", "_")+"_"+savepostfix.replace(" ", "_")+".png", bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)
    plt.close(fig_bar)

def plot_interim_heatmap(
    data, 
    stepnum, 
    cmap="afmhot", 
    title=None, 
    vmin=None, vmax=None, 
    target=None, spawn=None,
    start=None, agent=None, 
    figsize=None, savepostfix="vid"):

  if vmax:
    if not vmin:
      vmin = -vmax

  fig, ax = plt.subplots(1, 1, figsize=figsize)
  cbar_ax = fig.add_axes([.91, .3, .03, .4])
  plot_heatmap_to_ax(data, ax=ax, cmap=cmap, cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, target=target, spawn=spawn, start=start, agent=agent, figsize=figsize, savepostfix=savepostfix)  
  ax.text(9, 0, "t="+str(stepnum))
  if title is None:
    title = ""
  else: 
    ax.set_title(title)

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

  if vmax:
    if not vmin:
      vmin = -vmax

  if data1.shape[0] >= data1.shape[1]:
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

def plot_heatmap_to_ax(data, ax=None, cmap="afmhot", cbar_ax=True,
  vmin=None,vmax=None,target=None,spawn=None,start=None, agent=None, 
  figsize=None, display="Show", savepostfix=""):
  '''
    Args:
      data (rectangular dataset): from seaborn.heatmap
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is
        provided, the index/column information will be used to label the columns
        and rows.
      ...
      cbar_ax (cbar_axmatplotlib Axes): from seaborn.heatmap
        Axes in which to draw the colorbar, otherwise take space from the main 
        Axes.


  '''
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
    

def save_lineplot(ax, xlabel=None, ylabel=None, title=None, show_title=True, display="Show",savepostfix=""):
  ax.legend(frameon=False).set_title(None)
  sns.despine(ax=ax, offset=1, trim=True)
  if title and show_title:    
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



