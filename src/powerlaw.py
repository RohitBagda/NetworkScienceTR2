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
    G = get_seed_multidigraph(c) #create the graph
    for num_step in range(num_steps):
        #the n+1th node has index n (which is the number of nodes in G)
        #because the nodes are indexed starting with 0
        n_plus_one = len(G.nodes())

        #get a random node i from the graph
        node_i = random.randint(0, len(G.nodes()) - 1)

        #get neighbours of i
        node_i_neighbours = G.neighbors(node_i)
        for node_j in node_i_neighbours:
            #generate a value between 0 and 1, if that value is less than or equal to Gamma,
            #then set the target to node j
            #otherwise, the target is any random node
            probability = random.random()
            if probability <= gamma:
                new_neighbour = node_j
            else:
                new_neighbour = random.randint(0, len(G.nodes()) - 1)
        #add the node to the graph and add the arc from the new node to the new neighbour
        G.add_node(n_plus_one)
        G.add_edge(n_plus_one, new_neighbour)

    return G


def lnfa_model(c, sigma, num_steps):
    G = get_seed_multidigraph(c) #create the graph

    #create an array whose values are the "fitness" values for the node
    #i.e, the ith value of the array is the "fitness" of node i
    node_fitness_values = [get_fitness(sigma) for node in G.nodes()]

    for num_step in range(num_steps):
        # the n+1th node has index n (which is the number of nodes in G)
        # because the nodes are indexed starting with 0
        node_n_plus_one = len(G.nodes())

        #get fitness for new node
        random_fitness = get_fitness(sigma)
        #sum up all the fitness values currently in the graph
        #thus, the probability of selecting a node
        #is the fitness value of that node divided by the sum
        fitness_j_sum = 0
        for j in range(len(G.nodes())):
            fitness_j_sum += node_fitness_values[j]

        #create an array which contains the probabilities of each node,
        #the ith element in this array is the probability of selecting node i
        fitness_dist = [(value / fitness_j_sum) for value in node_fitness_values]

        #add node to graph
        G.add_node(node_n_plus_one)

        #create c links
        #we select a random node from the array based on the given probability distribution
        #in the fitness_dist and create an arc from new node to selected
        for c_link in range(c):
            #we exclude the nth node because we are selecting from all the other
            #nodes in the graph excluding the new one we just added
            new_neighbour = np.random.choice(a=[node_index for node_index in range(len(G.nodes()) - 1)], p=fitness_dist)

            #add the edge from new node to selected neighbour
            G.add_edge(node_n_plus_one, new_neighbour)

        #add the fitness value to our array of fitness values
        node_fitness_values.append(random_fitness)

    return G


###############################################################################
#
# MAIN EXECUTION
#
###############################################################################


#### replace this code with your own code to test your models

G1 = vertex_copy_model(1, 1/2, 10000)
plot_degrees(G1)
print(get_alpha(G1, 8))

G2 = lnfa_model(1, 2, 10000)
plot_degrees(G2)
print(get_alpha(G2, 8))



#### once your code is working, use these two methods to generate the 8 networks
#### that will be part of your written report
#run_vertex_copy('/output_dir_name')
#run_lnfa('/output_dir_name')



