import networkx as nx
import numpy as np
import matplotlib.pyplot as plot
import random

####################################################################################
#
# Test functions that might be helpful while implementing your models
#
####################################################################################

def get_alpha(G, k_min):
    ''' Returns the best fitting power law exponent for the in degree distribution '''
    return get_alpha_from_data([d for n, d in G.in_degree()], k_min)

def get_alpha_from_data(data, min_val):
    ''' Returns the best fitting power law exponent for the data '''
    data_sorted = sorted(data)
    min_idx = np.searchsorted(data_sorted, min_val)
    data2 = data_sorted[min_idx:]
    denom = np.log(min_val - 1 / 2)
    log_data = [np.log(x) - denom for x in data2]
    alpha = 1 + len(log_data) / sum(log_data)

    print('fit alpha on # points', len(data2))

    return alpha

def plot_degrees(graph, scale='log', colour='#40a6d1', alpha=.8):
    '''Plots the log-log degree distribution of the graph'''
    plot.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in graph.nodes():
        if graph.degree(n) > max_degree:
            max_degree = graph.degree(n)
    # X-axis and y-axis values
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree+1):
        x.append(i)
        y_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_tmp[i] += 1
        y = [i/num_nodes for i in y_tmp]
    # Plot the graph
    deg, = plot.plot(x, y,label='Degree distribution',linewidth=0, marker='o',markersize=8, color=colour, alpha=alpha)
    # Check for the lin / log parameter and set axes scale
    if scale == 'log':
        plot.xscale('log')
        plot.yscale('log')
        plot.title('Degree distribution (log-log scale)')
    else:
        plot.title('Degree distribution (linear scale)')


    plot.ylabel('P(k)')
    plot.xlabel('k')
    plot.show()

##########################################################################
#
# Functions that your final implementation will call
#
##########################################################################

def get_seed_multidigraph(c):
    ''' Returns a complete digraph on c+1 vertices'''
    graph = nx.gnp_random_graph(c+1,p=1,directed=True)
    seed_graph = nx.MultiDiGraph()
    seed_graph.add_nodes_from(graph)
    seed_graph.add_edges_from(graph.edges())

    return seed_graph

def get_fitness(sigma):
    ''' Samples from the standard lognormal distribution with std dev = sigma'''
    return np.random.lognormal(0, sigma)


def run_vertex_copy(out_dir_name):
    ''' Creates the four vertex copy model networks and exports them in gexf format'''
    c = 1
    num = 100
    outfile_template = out_dir_name + '/vertex-copy-{0}-{1}-{2}.gexf'

    for x in [1,3,6,8]:
        gamma = x/9
        G = vertex_copy_model(c, gamma, num)

        outfile = outfile_template.format(num,c,gamma)
        print('vertex copy model writing to ' + outfile)
        nx.write_gexf(G, outfile)

def run_lnfa(out_dir_name):
    ''' Creates the four LNFA model networks and exports them in gexf format'''
    c = 1
    num=100
    outfile_template = out_dir_name + '/lnfa-{0}-{1}-{2}.gexf'


    for sigma in [1/4, 1, 4, 16]:
        G = lnfa_model(c, sigma, num)

        outfile = outfile_template.format(num, c, sigma)
        print('lnfa model writing to ' + outfile)
        nx.write_gexf(G, outfile)

##########################################################################
#
# Your implementation starts here
#
##########################################################################
# def get_random_node(graph):
#     # nodes = graph.nodes()
#     # print(nodes)
#     # random_num = random.randint(0, len(nodes) - 1)
#     # return nodes[random_num]

def vertex_copy_model(c, gamma, num_steps):

    G = get_seed_multidigraph(c)

    for n_plus_one in range(num_steps):
        node_i = random.randint(0, len(G.nodes()) - 1)
        # print(node_i)
        node_i_neighbours = G.neighbors(node_i)
        for node_j in node_i_neighbours:
            probability = random.random()
            if probability <= gamma:
                new_neighbour = node_j
            else:
                new_neighbour = random.randint(0, len(G.nodes()) - 1)
        G.add_node(n_plus_one)
        G.add_edge(n_plus_one, new_neighbour)

    return G




    ############
    #
    # You provide the implementation that adds num_steps vertices
    # according to the Vertex Copy model
    #
    ###########

    return G



def lnfa_model(c, sigma, num_steps):

    G = get_seed_multidigraph(c)

    ############
    #
    # You provide the implementation that adds num_steps vertices
    # according to the LNFA model. Don't forget to assign fitnes
    # values to the c+1 vertrices in the seed graph G.
    #
    ###########


    return G


###############################################################################
#
# MAIN EXECUTION
#
###############################################################################


#### replace this code with your own code to test your models

G1 = vertex_copy_model(1, 1/2, 10000)
plot_degrees(G1)

G2 = lnfa_model(1, 2, 100)


#### once your code is working, use these two methods to generate the 8 networks
#### that will be part of your written report
#run_vertex_copy('/output_dir_name')
#run_lnfa('/output_dir_name')



