import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import xarray as xr
import cartopy.crs as ccrs



def plot_corr(results, Z, time, cp=0, method='ICA', save_plot=True) :
    plt.plot(time, Z, label='NAO', color='red')

    colors = {'PCA':'burlywood', 'ICA':'darkblue', 'PLS':'darkgreen'}

    confounder = results[method]['confounder'].T[cp]
    corr = np.corrcoef(Z, confounder)[0, 1]
    plt.plot(time, confounder*np.sign(corr), label=r'${}-PCF$'.format(method), color=colors[method])
    plt.xlabel('years')
    plt.legend()
    plt.title('Correlation: {}'.format(results[method]['corr'][cp]))
    if save_plot :
        plt.savefig('../Results/common_driver/Corr_Plot_Final_{}.pdf'.format(method), format='pdf')
    plt.show()
    
def plot_weights(results, latitudes, longitudes, original_shape, cp=0, method='ICA', mode_box='NAO', pos_x=-10, pos_y=50, title='', title_fontsize=5, save_plot=True, show_station=True):
    if method == 'PCA' :
        weights = results[method]['weights'].T[cp]
    else :
        weights = results[method]['weights'][cp]
        
    reshaped_array = weights.reshape(original_shape[1], original_shape[2])

    # Create a DataArray with the reshaped array and coordinate information
    new_xarray = xr.DataArray(reshaped_array, coords={'lat': latitudes, 'lon': longitudes}, dims=['lat', 'lon'])

    # Create a figure and axis for the plot with Orthographic projection centered on (-30, 35)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Orthographic(pos_x, pos_y)})
    
    #ax.add_patch(mpatches.Rectangle(xy=(30, 0), width=360, height=180, facecolor='gray', alpha=0.5, transform=ccrs.PlateCarree()))


    # Plot the data on the map using Orthographic projection
    new_xarray.plot(ax=ax, cmap='coolwarm', transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.4, 'location': 'left', 'pad': 0.05})

    # Add coastlines, gridlines, and a title with a specified font size
    ax.coastlines(linewidth=0.3)
    ax.set_global()
    ax.set_title(title, fontsize=title_fontsize)

    if show_station :
        # Icelandic Low (65°N, 20°W) - Empty Square in Orthographic coordinates
        ax.scatter(-20, 65, s=25, marker='s', edgecolor='black', facecolor='none', label='Reykjavík', transform=ccrs.PlateCarree())

        # Azores High (40°N, 45°W) - Empty Triangle in Orthographic coordinates
        ax.scatter(-25.68, 37.7, s=25, marker='^', edgecolor='black', facecolor='none', label='Ponta Delgada', transform=ccrs.PlateCarree())


    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    if save_plot:
        plt.savefig('../Results/common_driver/Weights_Plot_Final_{}.pdf'.format(method), format='pdf')
    # Show the plot
    plt.show()
    
def plot_graph(results, coef_MED, coef_DK, method='ICA', NAO=None, save_plot=True) :
    # extract confounders
    confounders = results[method]['confounder']

    # Number of conounders
    n_confounders = confounders.shape[1]


    #################################
    ##### Creating the plot #########
    #################################
    if NAO is not None :
        correlations = np.corrcoef(np.hstack((NAO[:,None], confounders)).T)[0, 1:].round(2)
        z_columns = ['Z_' + str(i) + '\n' + str(correlations[i]) for i in range(n_confounders)]
    else :
        z_columns = ['Z_'+str(i)for i in range(n_confounders)]

    # Create a directed graph
    G = nx.DiGraph()

    # Define node positions for Z, MED, and DK with grey color
    pos = {
        'MED': (-0.5, -1),
        'DK': (0.5, -1),
    }
    node_colors = {
        'MED': 'grey',
        'DK': 'grey',
    }

    # Add Z nodes in a horizontal line with grey color
    z_positions = [((-1+i), 1) for i in np.linspace(0, 1, n_confounders)]  # Adjust positions as needed
    for i, z_col in enumerate(z_columns):
        pos[z_col] = z_positions[i]
        node_colors[z_col] = 'grey'

    # Add directed edges based on regression weights with red arrows for positive and blue arrows for negative weights
    for i, z_col in enumerate(z_columns):
        weight = coef_MED[i]
        edge_color = 'red' if weight >= 0 else 'blue'

        # Adjust edge positions manually
        edge_pos = [(pos[z_col][0] + 0.1, pos[z_col][1] + 0.1),
                    (pos['MED'][0] - 0.1, pos['MED'][1] + 0.1)]

        G.add_edge(z_col, 'MED', weight=weight, color=edge_color, arrows=True)

        weight = coef_DK[i]
        edge_color = 'red' if weight >= 0 else 'blue'

        # Adjust edge positions manually
        edge_pos = [(pos[z_col][0] + 0.1, pos[z_col][1] - 5),
                    (pos['DK'][0] - 0.1, pos['DK'][1] + 0.1)]

        G.add_edge(z_col, 'DK', weight=weight, color=edge_color, arrows=True)

    # Calculate edge widths based on weights (scaled by a factor for better visualization)
    edge_widths = [abs(G[u][v]['weight']) * 3 for u, v in G.edges]  # Adjust the scaling factor as needed

    # Draw the causal graph with straight edges
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=[node_colors[node] for node in G.nodes], font_size=10, font_color='black', font_weight='bold', width=edge_widths, edge_color=[G[u][v]['color'] for u, v in G.edges], connectionstyle="arc3,rad=0.1", edge_cmap=plt.cm.RdBu_r)
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges}

    for (u, v), label in edge_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        label_x = x - 0.05 
        label_y = y + 0.1  if v == 'MED' else y - 0.1  # Adjust the x-coordinate for MED and DK edges
        plt.text(label_x, label_y, label, fontsize=8, ha="center", va="center", color='black')




    # Show the plot
    plt.title("")
    plt.axis('off')
    if save_plot:
        plt.savefig('../Results/common_driver/CAusal_Graph_Plot_Final_{}.pdf'.format(method), format='pdf')

    plt.show()
