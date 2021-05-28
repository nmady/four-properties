import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import typer
import os

def plot_bars(steps, width, height, 
        figsize, filepath, outfilepath, y_logscale=False):

    csv_dict = {}
    length_dict = {}
    longest = 0

    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            label = row.pop(0)

            #Note last item always empty because of trailing comma
            row.pop(-1)

            split_label = label.split('_')
            if (split_label[0] == str(width)
                    and split_label[1] == str(height)
                    and split_label[2] == "steps" + str(steps)):

                longest = len(row) if len(row) > longest else longest
                
                if label in csv_dict:
                    if len(row) > len(csv_dict[label]):
                        csv_dict[label] = row
                        length_dict[label] = len(row)
                else:
                    csv_dict[label] = row
                    length_dict[label] = len(row)
    
    #We have to pad the lists if they are of different lengths for pandas
    for val in csv_dict.values():
        val += [np.nan] * (longest - len(val))


    df = pd.DataFrame(csv_dict)
    df = df.astype(float)


    fig, ax = plt.subplots()
    if figsize is not None:
        fig = plt.figure(figsize=figsize,dpi=200, tight_layout=True)
    else:
        fig = plt.figure(dpi=200)
    ax = sns.barplot(data=df)
    sns.despine()
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    plt.xticks(rotation=45, ha='right')
    if y_logscale:
        plt.yscale('log')
    ax.set_ylabel("Count of visits to targets")
    fig.tight_layout()
    plt.savefig(outfilepath)
    plt.close()


def main(
        steps: int = typer.Option(5000, help="Number of steps in each trial."),
        width: int = typer.Option(11, help="Width of gridworld."),
        height: int = typer.Option(11, help="Height of gridworl."),
        figwidth: float = typer.Option(None, 
            help="Width of heatmap figures in inches"),
        figheight: float = typer.Option(None, 
            help="Height of heatmap figure in inches"),
        inpath: str = typer.Option("./output/num_target_visits.csv",
            help="Path to the csv to build the barplot from."),
        y_logscale: bool = typer.Option(False,
            help="Scale the y-axis on a log scale."),
        outpath: str = typer.Option(None,
            help="Output will be saved as a file to this path."),
        ):
    """
    Let's make a nice barplot from the csv file.
    """

    if ((figwidth is not None and figheight is None) or 
        (figheight is not None and figwidth is None)):
        raise ValueError(
            "We need both figwidth and figheight to set figsize."
            )
    if figwidth is None and figheight is None:
        figsize = None
    else:
        figsize = (figwidth, figheight)

    assert os.path.exists(inpath)

    if outpath is None:
        outpath = "./output/barplot"+str(width) + "_" + str(height) + "_" + str(steps) +".png"

    plot_bars(steps, width, height, figsize, 
        inpath, outpath, y_logscale=y_logscale)


if __name__ == "__main__":
    typer.run(main)