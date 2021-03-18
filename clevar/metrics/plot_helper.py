def add_grid(ax):
    ax.xaxis.grid(True, which='major', lw=.5)
    ax.yaxis.grid(True, which='major', lw=.5)
    ax.xaxis.grid(True, which='minor', lw=.1)
    ax.yaxis.grid(True, which='minor', lw=.1)
