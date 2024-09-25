from typing import List

import numpy as np
import copy

from anaconda_navigator.api.external_apps.validation_utils import catch_exception

from config import parse_settings_file
from libs.tsp import two_opt, path_distance_for_circular
from libs.grafo import COPS
from collections import OrderedDict
import time


def my_print(msg, index_print=0):
    if index_print == 1:
        print(msg)
    elif index_print == 2:
        print(msg)

class Cluster:
    """
    Class container to clusters
                [c] -> index of the cluster
                [x] -> [0] -> Vertex list belonging to this cluster
                       [1] -> list of sets who this cluster belongs it
                       [2] -> LONG_TERM_MEMORY ( > 0 in solution) (<= 0 not in solution)
                       [3] -> 0 if only belong to start or end sets
                       [4] -> Aspiration level (the best profit obtained all times this cluster was in solution path)
                       [5] -> 1 this cluster can be inserted in solution 0 otherwise
                       [6] -> index (profit / number_of_cluster)
    :ivar vertices_of_cluster: List of vertices belonging to this cluster
    :ivar subgroups: List of subgroups that belongs to this cluster
    :ivar long_term_memory: LONG_TERM_MEMORY ( > 0 in solution) (<= 0 not in solution)
    :ivar belong_start_end: 0 if only belong to start or end sets
    :ivar aspiration_level: Aspiration level (the best profit obtained all times this cluster was in solution path)
    :ivar can_be_inserted: 1 this cluster can be inserted in solution 0 otherwise
    :ivar index_profit: index (profit / number_of_cluster)
    """
    def __init__(self, list_vertex: list = None, list_subgroup: list = None, indexes_profit: list = None):
        """
        Initialize the class to store the cluster data
        :param list_vertex: Vertex list belonging to this cluster
        :param list_subgroup: list of sets who this cluster belongs it
        :param indexes_profit: index (profit / number_of_cluster)
        """
        if list_vertex is not None:
            self.vertices_of_cluster:list = list_vertex.copy()
        else:
            self.vertices_of_cluster:list = []
        if list_subgroup is not None:
            self.subgroups: list = list_subgroup.copy()
        else:
            self.subgroups: list = []
        self.long_term_memory: int = 0
        self.belong_start_end: bool = False
        self.aspiration_level: float = 0.0
        self.can_be_inserted: bool = True
        if indexes_profit is not None:
            self.index_profit: List[float] = indexes_profit
        else:
            self.index_profit: List[float] = []

class Vertex:
    """
    Class container to vertex

    :ivar position: [X, Y, Z] The position of the vertex
    :ivar visited: if vertex is being visited (1 visited, 0 otherwise)
    :ivar clusters_belongs: List of clusters who this vertex it belongs
    """
    def __init__(self, position: np.ndarray = np.empty(0), visited: bool = False, clusters_belongs: list = None):
        """
        Class initializer
        :param position: [X, Y, Z] or [X, Y] The position of the vertex
        :param visited: if vertex is being visited (1 visited, 0 otherwise)
        :param clusters_belongs: List of clusters who this vertex it belongs
        """
        self.position: np.ndarray = position
        self.visited: bool = visited
        if clusters_belongs is not None:
            self.clusters_belongs: list = clusters_belongs
        else:
            self.clusters_belongs: list = []

class Subgroup:
    def __init__(self,vertices_i: List[int] = None, clusters_i: List[int] = None):
        if vertices_i is not None:
            self.vertices_subgroup: List[int] = vertices_i.copy()
        else:
            self.vertices_subgroup: List[int] = []
        if clusters_i is not None:
            self.clusters_subgroup = clusters_i.copy()
        else:
            self.clusters_subgroup: List[int] = []

class TabuSearchCOPS(COPS):
    """
    General COPS class
    :ivar settings: Store all settings to the configuration parameters
    :ivar tabu_alpha: Minimum value to the long-term memory value to include the subgroup to be removed of the solution
    :ivar beta: Minimum value to consider a point be considered to be randomly removed.
    :ivar max_initial_solution_attempts: Maximum attempts to generate an initial solution without improvements.
    :ivar iterations_without_improvement: Maximum number of attempts to generate a solution without an improvement to stop the optimization iterations.
    :ivar absolute_max_iterations: Maximum number of iterations to find a final solution even if no improve is reached.
    :ivar cops: Instance of COPS class defined in the grafo.py file
    :ivar array_profit: Array of profit to each subgroup
    :ivar matrix_dist: Matrix distance of node i to j
    :ivar vertices: List of vertices. Each element of vertices is an instance of the Vertex class.
    :ivar subgroups: Subgroups list. Each element is an instance of Subgroup class
    :ivar clusters: Clusters list. Each element is an instance of Cluster class
    :ivar visited_subgroups: Indexes list for the subgroups in the list of subgroups in the solution list.
    :ivar clusters_can_be_updated: List of clusters that do not belong to start or end vertex
    :ivar solution: Dictionary with the solution subgroups, profit, and vertices
    :ivar best_solution: Dictionary with the best solution.
    :ivar start_cluster: The cluster that starts the path. The start cluster is configured in the .cops file.
    :ivar end_cluster: The cluster that ends the path. The end cluster is configured in the .cops file.
    :ivar start_subgroup: The first subgroup of the first cluster.
    :ivar end_subgroup: The subgroup that belongs to the end cluster.
    :ivar num_clusters: The total number of clusters.
    :ivar best_end_cluster: The cluster of the end for the path with better profit
    """
    def __init__(self, cops_class):
        """
        Initialization function
        :param cops_class: Class with COPS data read from .cops file
        """

        super().__init__()
        self.settings = parse_settings_file("configCOPS.yaml")
        self.tabu_alpha: float = self.settings["tabu alfa"]
        self.beta: float = self.settings["beta"]
        self.max_initial_solution_attempts: int = self.settings["max initial solution attempts"]
        self.iterations_without_improvement: int = 0
        self.max_iterations_without_improvement: int = self.settings["max iteration without improvements"]
        self.absolute_max_iterations: int = self.settings["absolute max iterations"]
        self.iterations_to_change_final_set: int = 0

        self.cops: cops_class = cops_class
        self.array_profit: np.ndarray = np.array(cops_class.profit)
        self.matrix_dist: np.ndarray = cops_class.matrix_dist

        self.start_subgroup: int = -1
        self.end_subgroup: int = -1

        # Load the vertices from .cops file by the COPS class
        self.vertices: List[Vertex] = [Vertex(np.array(x), False, []) for x in cops_class.list_vertex]

        # Load the subgroups from .cops file by the COPS class. Each subgroup is a set of vertex
        self.subgroups: List[Subgroup] = [Subgroup(vertices_i=x) for x in cops_class.list_subgroups]

        # Load the clusters from .cops file by the COPS class. Each cluster is a set of subgroups
        self.clusters: List[Cluster] = [Cluster([], cops_class.list_clusters[x], []) for x in range(len(cops_class.list_clusters))]

        self.num_clusters = len(self.clusters)

        # Note that the same cluster can have more than one subgroup.
        #            This loop says about what cluster contain the sets.
        #            Ex: self.array_clusters[c][1] = [2,3] means that cluster c contains sets 2 and 3 """
        for c in range(self.num_clusters): #The subgroups are sequential then will not exist subgroup 10 without the nine groups before.Clusters are sequential too.
            for v in self.clusters[c].subgroups:
                self.clusters[c].vertices_of_cluster.extend(self.subgroups[v].vertices_subgroup.copy())
                self.subgroups[v].clusters_subgroup.append(c)

        # Compute the profit to each subgroup of each cluster. One cluster is a list of subgroups the profit index stores the normalized profit to each subgroup of the cluster
        for c in range(self.num_clusters):
            for s in self.clusters[c].subgroups:
                self.clusters[c].index_profit.extend([self.array_profit[s]/len(self.clusters[c].vertices_of_cluster)])

        # Note that the same vertex can belong to more than one subgroup.
        #            This loop says about each vertex belongs to which clusters.
        for c in range(self.num_clusters):
            for v in self.clusters[c].vertices_of_cluster:
                self.vertices[v].clusters_belongs.append(c)

        # Read start and end cluster from .cops file
        self.start_cluster = cops_class.start_cluster
        self.end_cluster = cops_class.end_cluster
        self.best_end_cluster = copy.deepcopy(self.end_cluster)

        self.clusters[self.start_cluster].long_term_memory = 1  # long-term memory (>=1 in solution)
        self.clusters[self.start_cluster].belong_start_end = True

        """ Note that: It's possible that end set contain more than one cluster who finishes the path,
                                but each end cluster must contain only one cluster."""

        self.end_subgroup = np.random.choice(self.clusters[self.end_cluster].subgroups)
        best_end_subgroup = copy.deepcopy(self.end_subgroup)
        self.start_subgroup = self.clusters[self.start_cluster].subgroups[0]

        self.clusters[self.end_cluster].long_term_memory = 1  # long-term memory (>=1 in solution)
        self.clusters[self.end_cluster].belong_start_end = True

        """ This dictionary contain:
            the clusters who belongs to any set except the start and end sets """
        # all_clusters contain all clusters except the clusters who belong only to the initial or final sets.
        # Note: the cluster who belong to initial or final sets could belong to another set

        # (in this case this cluster will be in the all_clusters variable)

        # all_clusters = [self.clusters[c] for c in range(num_clusters) if c != start_cluster and c != end_cluster]

        self.clusters_can_be_updated: np.ndarray = np.array([i for i in range(self.num_clusters) if not self.clusters[i].belong_start_end])

        """ These variable indicates whether a cluster or set is being visited
            0 -> unvisited  1 -> visited"""
        self.visited_subgroups: np.ndarray = np.zeros(len(self.subgroups))
        # self.vertex_visited = np.zeros(len(self.array_vertex))

        self.solution: dict[str: int,str: int,str: list,str: list] = {"profit": 0,
                         "distance": 0,
                         "route": [],
                         "subgroups_visited": [],
                         # "sets_visited": [],
                         }

        self.best_solution: dict[str: int,str: int,str: list,str: list] = {"profit": 0,
                              "distance": 0,
                              "route": [],
                              "subgroups_visited": [],
                              # "sets_visited": [],
                              }
        return

    # def __init__(self, cops_class):
    #     super().__init__()
    #     settings =  parse_settings_file("configCOPS.yaml")
    #     self.tabu_alfa = settings["tabu alfa"]
    #     self.beta = settings["beta"]
    #     self.max_initial_solution_attempts = settings["max initial solution attempts"]
    #     self.iterations_without_improvement = 0
    #     self.max_iterations_without_improvement = settings["max iteration without improvements"]
    #     self.absolute_max_iterations = settings["absolute max iterations"]
    #     self.iterations_to_change_final_set = 0
    #
    #     self.cops = cops_class
    #     self.array_profit = np.array(cops_class.profit)
    #     self.matrix_dist = cops_class.matrix_dist
    #
    #     # VERTICES
    #     """ self.array_vertex[v][x]
    #         [v] -> index of the vertex
    #         [x] -> [0] -> [x, y]
    #                [1] -> if vertex is being visited (1 visited, 0 otherwise)
    #                [2] -> list of clusters who this vertex belongs it
    #     """
    #     self.array_vertex = np.array([np.array([np.array(x), 0, []], dtype=object) for x in cops_class.list_vertex], dtype=object)
    #
    #
    #     # SETS
    #     """  """
    #     self.array_sets = np.array([np.array(x) for x in cops_class.list_clusters], dtype=object)
    #     self.num_sets = len(self.array_sets)
    #
    #     # CLUSTERS
    #     """ self.array_clusters[c][x]
    #         [c] -> index of the cluster
    #         [x] -> [0] -> list of vertex who belongs to this cluster
    #                [1] -> list of sets who this cluster belongs it
    #                [2] -> LONG_TERM_MEMORY ( > 0 in solution) (<= 0 not in solution)
    #                [3] -> 0 if only belong to start or end sets
    #                [4] -> Aspiration level (the best profit obtained all times this cluster was in solution path)
    #                [5] -> 1 this cluster can be inserted in solution 0 otherwise
    #                [6] -> index (profit / number_of_cluster)
    #     """
    #     self.array_clusters = np.array(
    #         [np.array([np.array(cops_class.list_subgroups[x]), [], 0, 0, 0, 1, self.array_profit[x] / len(cops_class.list_subgroups[x])], dtype=object) for x in range(len(cops_class.list_subgroups))], dtype=object)
    #     #my_print(f"array_cluster {self.array_clusters}", index_print=1)
    #     self.num_clusters = len(self.array_clusters)
    #     # Note that the same cluster can belong to more than one set.
    #     #            This loop says about each cluster which sets it belongs to.
    #     #            Ex: self.array_clusters[c][1] = [2,3] means that cluster c belongs to sets 2 and 3 """
    #     for s in range(self.num_sets):
    #         for c in self.array_sets[s]:
    #             self.array_clusters[c][1].append(s)
    #             # self.cluster_match_sets[c].append(s)
    #
    #     # Note that the same vertex can belong to more than one set.
    #     #            This loop says about each vertex which clusters it belongs to.
    #     for c in range(self.num_clusters):
    #         for v in self.array_clusters[c][0]:
    #             self.array_vertex[v][2].append(c)
    #
    #     self.start_set = cops_class.start_cluster
    #     self.end_set = cops_class.end_cluster
    #
    #     self.start_cluster = self.array_sets[self.start_set][0]  # The start cluster is inside start set
    #     self.array_clusters[self.start_cluster][2] = 1  # long term memory (>=1 in solution)
    #
    #     """ Note that: It's possible that end set contain more than one cluster who finishes the path,
    #                             but each end cluster must contain only one cluster."""
    #     self.end_cluster = np.random.choice(self.array_sets[self.end_set])
    #     self.best_end_cluster = copy.deepcopy(self.end_cluster)
    #     self.array_clusters[self.end_cluster][2] = 1  # long term memory (>=1 in solution)
    #
    #     """ This dictionary contain:
    #         the clusters who belongs to any set except the start and end sets """
    #     # all_clusters contain all clusters except the clusters who belong only to the initial or final sets.
    #     # Note: the cluster who belong to initial or final sets could belong to another set
    #     # (in this case this cluster will be in the all_clusters variable)
    #     all_clusters = [c for s in range(self.num_sets) for c in self.array_sets[s] if
    #                     s != self.start_set and s != self.end_set]
    #     # the dictionary will eliminate the repeated clusters in all_clusters variable
    #     for c in all_clusters:
    #         self.array_clusters[c][3] = 1
    #     self.clusters_can_be_updated = np.array(
    #         [i for i in range(self.num_clusters) if self.array_clusters[i][3] == 1])
    #
    #     """ These variable indicates whether a cluster or set is being visited
    #         0 -> unvisited  1 -> visited"""
    #     self.sets_visited = np.zeros(len(self.array_sets))
    #     # self.vertex_visited = np.zeros(len(self.array_vertex))
    #
    #     self.solution = {"profit": 0,
    #                      "distance": 0,
    #                      "route": [],
    #                      "subgroups_visited": [],
    #                      # "sets_visited": [],
    #                      }
    #
    #     self.best_solution = {"profit": 0,
    #                           "distance": 0,
    #                           "route": [],
    #                           "subgroups_visited": [],
    #                           # "sets_visited": [],
    #                           }

    def insertion_neighborhood(self, neighbor, inserted_cluster, inserted_subgroup):
        """
        Insert a new cluster to the path of the solution. a) Non-Tabu Insertion: Choose randomly from S_client a subgroup that is not in the tabu list and that does not belong to a cluster that is in the current
        solution P, and insert it into the neighbor P′.
        :param inserted_subgroup:
        :param neighbor: Set of subgroups of the computed path until now
        :param inserted_cluster: Cluster to be inserted in the neighbor
        :return: Need to update the solution
        """
        need_a_solution = True

        n_tour, n_distance = self.tour_generation(neighbor)

        """ verify if this neighborhood is feasible """
        if n_distance < self.cops.t_max:
            """ verify if this neighborhood has the best profit or
                if the neighborhood has the same profit but less distance 
                NOTE: It will choose better paths when the profit is equal and the distance is less """
            n_profit = self.solution["profit"] + self.array_profit[inserted_subgroup]

            if n_profit > self.solution["profit"] or (
                    n_profit == self.solution["profit"] and n_distance < self.solution["distance"]):

                """ update solution """
                self.solution["subgroups_visited"].append(inserted_subgroup)
                self.solution["route"] = n_tour
                self.solution["distance"] = n_distance
                self.solution["profit"] = n_profit

                """ update subgroups visited"""
                for s in self.subgroups[inserted_subgroup].clusters_subgroup:  # self.cluster_match_sets[rand_cluster]:
                    self.visited_subgroups[s] = 1

                """ update long_term_memory """
                for c in range(self.num_clusters):
                    if self.clusters[c].long_term_memory > 0:
                        self.clusters[c].long_term_memory += 1
                    else:
                        self.clusters[c].long_term_memory -= 1
                    self.clusters[inserted_cluster].long_term_memory = 1  # Inserted cluster should be a value 1

                """ update vertex inserted """
                for v in self.clusters[inserted_cluster].vertices_of_cluster:
                    self.vertices[v].visited = 1

                """ update Aspiration level """
                for s in self.solution["subgroups_visited"]:
                    for c in self.subgroups[s].clusters_subgroup:
                        profit_idx = self.clusters[c].subgroups.index(s)
                        if n_profit > self.clusters[c].index_profit[profit_idx]:
                            self.clusters[c].index_profit[profit_idx] = n_profit

                need_a_solution = False
                my_print(f"CHANGE THE SOLUTION NEIGHBORHOOD {self.solution}")
        return need_a_solution

    def tour_update_remove_cluster(self, removed_cluster):
        """
        Remove the cluster deleting relations of the cluster
        :param removed_cluster: Cluster to be removed
        :return:
        """
        n_tour = []
        n_distance = 0

        """ eliminate a vertex if it belongs ONLY to the cluster who will be removed """
        eliminated_vertex = []
        for v in self.clusters[removed_cluster].vertices_of_cluster:  # vertex in removed cluster
            can_eliminate_this_vertex = True
            for c in self.vertices[v].clusters_belongs:  # clusters who this vertex belongs it (remember a vertex could belong to more than one cluster)
                if self.clusters[c].long_term_memory > 0:
                    if c != removed_cluster:
                        can_eliminate_this_vertex = False
                        break
            if can_eliminate_this_vertex:
                eliminated_vertex.append(v)
        """ eliminate edges """
        new_edge_init = -1
        for t in range(1, len(self.solution["route"])):
            if any(self.solution["route"][t][0] == x for x in eliminated_vertex):
                eliminated_vertex.remove(self.solution["route"][t][0])
                if new_edge_init == -1:
                    new_edge_init = self.solution["route"][t-1][0]
            else:
                if new_edge_init == -1:
                    init = self.solution["route"][t - 1][0]
                    end = self.solution["route"][t - 1][1]
                    n_tour.append((init, end))
                    n_distance += self.cops.matrix_dist[init][end]
                else:
                    init = new_edge_init
                    new_edge_init = -1
                    end = self.solution["route"][t][0]
                    n_tour.append((init, end))
                    n_distance += self.cops.matrix_dist[init][end]
        # treatment for the last edge
        t = len(self.solution["route"]) - 1
        if new_edge_init == -1:
            init = self.solution["route"][t][0]
            end = self.solution["route"][t][1]
            n_tour.append((init, end))
            n_distance += self.cops.matrix_dist[init][end]
        else:
            init = new_edge_init
            end = self.solution["route"][t][1]
            n_tour.append((init, end))
            n_distance += self.cops.matrix_dist[init][end]

        return n_tour, n_distance

    def removal_neighborhood(self, neighbor, removed_cluster):
        """
        The removal of a subgroup which implies the removal of the cluster to the subgroup belongs by just removing their vertices from the solution and join its predecessor to the successor.
        :param neighbor: Subgroup list that will be removed the subgroup
        :param removed_cluster: The cluster that has the subgroup that will be removed
        :return:
        """
        #need_a_solution = True

        #n_tour, n_distance = self.tour_generation(neighbor)
        n_tour, n_distance = self.tour_update_remove_cluster(removed_cluster)

        """ verify if this neighborhood is feasible """
        #if n_distance < self.cops.t_max:
        """ update the aspiration level"""
        for i in range(len(self.clusters[removed_cluster].index_profit)):
            self.clusters[removed_cluster].index_profit[i] = self.solution["profit"]

        #n_profit = self.solution["profit"] - self.array_profit[removed_cluster]



        """ update subgroups visited"""
        for s in self.clusters[removed_cluster].subgroups:  # self.cluster_match_sets[rand_cluster]:
            self.visited_subgroups[s] = 0
            self.solution["profit"] -= self.array_profit[s]
            try:
                neighbor.remove(s)
            except:
                pass

        """ update solution """
        self.solution["subgroups_visited"] = neighbor  # .remove(removed_cluster)
        self.solution["route"] = n_tour
        self.solution["distance"] = n_distance

        """ update long_term_memory """
        for c in range(self.num_clusters):
            if self.clusters[c].long_term_memory > 0:
                self.clusters[c].long_term_memory += 1
            else:
                self.clusters[c].long_term_memory -= 1
            self.clusters[removed_cluster].long_term_memory = 0  # removed cluster should be a value 0

        """ update vertex removed """
        for v in self.clusters[removed_cluster].vertices_of_cluster:
            self.vertices[v].visited = False

        #need_a_solution = False
        my_print(f"CHANGE THE SOLUTION NEIGHBORHOOD {self.solution}" )
        """ Always generates a feasible solution"""
        return False  #need_a_solution

    def insertion_criterion(self, index):
        criterion = self.clusters[index].index_profit * self.clusters[index].long_term_memory
        # remember: if we want to insert than the cluster are not in solution (long-term <= 0)
        min_value = np.argmin(criterion)
        chosen_cluster = index[min_value]
        #my_print(f"{chosen_cluster} - {self.array_clusters[index, 6]} - {self.array_clusters[index, 2]} - {criterion}", index_print=1)
        return chosen_cluster

    def generate_neighborhood(self):
        """
        Each iteration will generate a group of neighbors of the current solution P in a predefined sequence. In order to avoid solving many TSP instances and then choose the best move, the first plausible
        solution found should be considered as the neighbor P′ for each iteration. Next, this neighbor should be compared with the final solution P∗. The final solution will be exchanged for this neighbor
        if it is more profitable as the final solution or has the same reward but with a lower traversal cost. Note that it will choose more efficient paths when the reward is the same but the cost is lower.
        The rules to produce the neighbors follow two lines: (i) an insertion of some feasible subgroup in the current solution; or (ii) a removal of a subgroup from the solution. The insertion is made by solving
        the classic local search strategy 2-opt [16] for all chosen vertices. The removal of a subgroup is further time optimized by just removing their vertices from the solution and join its predecessor to the
        successor. However, it is necessary to take care not to remove a vertex which is shared by any other subgroup in the current solution P. Neighbors are generated in the following sequence:
        a) Non-Tabu Insertion: Choose randomly from S_client a subgroup that is not in the tabu list and that does not belong to a cluster that is in the current solution P, and insert it into the neighbor P′.
        b) Old Removal: Remove from P a randomly chosen subgroup from Sclient for which ηi > β, where β is a constant.
        c) Tabu Insertion: Insert a subgroup from tabu insert list that does not belong to some cluster served in the current solution. The chosen subgroup is the one with the highest aspiration level.
        Similar to [2], the aspiration level is the highest reward value obtained on any solution that contained this subgroup.
        d) Random Insertion: Randomly chose and insert any subgroup from Sclient that does not belong to some cluster served in the current solution.
        e) Non-Tabu Removal: Remove from P a randomly chosen subgroup from Sclient that is non-tabu to remove.
        f) Random Removal: Chose randomly a client subgroup from the current solution and remove it.
        5) Non-circular paths: If the starting point differs from the ending point of the path, then there will be an end cluster, and all end subgroups must contain only one vertex, and the solver will choose
        which subgroup will be the end depot. In this case, the endpoint will be changed after T iterations without improvement. For this, we choose the subgroup with the minimum long-term memory, that is, the
        subgroup that has been outside the solution for the longest time. T is defined as β divided by the number of final subgroups. The idea is that the neighborhoods of all endpoints will be tested before the
        end of the run.
        6) Stop condition: The algorithm stops after β iterations without increasing the reward of the final solution, or decreasing the distance traveled, keeping the same reward. The value of β obtained after
        preliminary tests was 300.
        :return:
        """
        need_a_neighborhood = True

        visited_clusters = []
        non_tabu_remove = []
        old_visited = []

        unvisited_clusters = []
        non_tabu_insertion = []
        tabu_insertion = []

        for i in self.clusters_can_be_updated:
            long_term_memory = self.clusters[i].long_term_memory
            # update the lists for remove
            if long_term_memory > 0:
                visited_clusters.append(i)
                if long_term_memory > self.tabu_alpha: #Tabu list: A subgroup will be a tabu (forbidden) to insert or remove if, respectively, ηi > −α and ηi < α ,where α is the tabu constant.
                    non_tabu_remove.append(i)
                elif long_term_memory > self.beta: #Old Removal: Remove from P a randomly chosen subgroup from S_client for which long-term memory > β ,where β is a constant.
                    old_visited.append(i)
            else:
                """ it's a possible insertion if this cluster don't belong to a subgroup in the solution """
                a = self.clusters[i].subgroups  #
                none_set_was_visited = True
                for aa in a:
                    if self.visited_subgroups[aa] == 1:
                        none_set_was_visited = False
                        break
                # update the lists for insertion
                if none_set_was_visited:
                    unvisited_clusters.append(i)
                    if long_term_memory < -self.tabu_alpha:
                        non_tabu_insertion.append(i)
                    else:
                        tabu_insertion.append(i)

        """ Non-Tabu Insertion """
        if any(non_tabu_insertion):
            """ Neighborhoods will be generated by a small modification of the current solution """
            neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
            chosen_cluster = np.random.choice(non_tabu_insertion)
            chosen_subgroup = np.random.choice(self.clusters[chosen_cluster].subgroups)
            neighborhood.append(chosen_subgroup)
            need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster, chosen_subgroup)
            if not need_a_neighborhood:
                my_print(f"non_tabu_insertion {chosen_cluster} {non_tabu_insertion}")
            else:
                my_print(f"discarded Non-Tabu Insertion {chosen_cluster} {non_tabu_insertion}")

        """ Old Removal """
        if need_a_neighborhood:
            if any(old_visited):
                """ Neighborhoods will be generated by a small modification of the current solution """
                neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
                chosen_cluster = np.random.choice(old_visited)
                need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
                if not need_a_neighborhood:
                    my_print(f"old_Removal {chosen_cluster} {old_visited}")
                else:
                    my_print(f"discarded old_Removal {chosen_cluster} {old_visited}")

        """ Tabu Insertion """
        if need_a_neighborhood:
            if any(tabu_insertion):
                neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
                chosen_cluster = tabu_insertion[np.argmax([self.clusters[i].aspiration_level for i in tabu_insertion])]
                chosen_subgroup = np.random.choice(self.clusters[chosen_cluster].subgroups)
                neighborhood.append(chosen_subgroup)
                need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster, chosen_subgroup)
                if not need_a_neighborhood:
                    my_print(f"tabu_insertion {chosen_cluster} {tabu_insertion}")
                else:
                    my_print(f"discarded tabu_insertion {chosen_cluster} {tabu_insertion}")

        """ Non-Tabu Removal """
        if need_a_neighborhood:
            if any(non_tabu_remove):
                """ Neighborhoods will be generated by a small modification of the current solution """
                neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
                chosen_cluster = np.random.choice(non_tabu_remove)
                # neighborhood.remove(chosen_cluster)
                need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
                if not need_a_neighborhood:
                    my_print(f"Non-Tabu Removal {chosen_cluster} {non_tabu_remove}")
                else:
                    my_print(f"discarded Non-Tabu Removal {chosen_cluster} {non_tabu_remove}")

        """ Random Removal """
        if need_a_neighborhood:
            if any(visited_clusters):
                """ Neighborhoods will be generated by a small modification of the current solution """
                neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
                chosen_cluster = np.random.choice(visited_clusters)
                need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
                if not need_a_neighborhood:
                    my_print(f"Random Removal {chosen_cluster} {visited_clusters}")
                else:
                    my_print(f"discarded Random Removal {chosen_cluster} {visited_clusters}")

        """ Random Insertion """
        if need_a_neighborhood:
            if any(unvisited_clusters):
                """ Neighborhoods will be generated by a small modification of the current solution """
                neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
                chosen_cluster = np.random.choice(unvisited_clusters)
                neighborhood.append(chosen_cluster)
                chosen_subgroup = np.random.choice(self.clusters[chosen_cluster].subgroups)
                neighborhood.append(chosen_subgroup)
                need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster, chosen_subgroup)
                if not need_a_neighborhood:
                    my_print(f"Random Insertion {chosen_cluster} {unvisited_clusters}")
                else:
                    my_print(f"discarded Random Insertion {chosen_cluster} {unvisited_clusters}")

        if need_a_neighborhood:
            for c in range(self.num_clusters):
                long_term_memory = self.clusters[c].long_term_memory
                if long_term_memory > 0:
                    long_term_memory += 1
                else:
                    long_term_memory -= 1
        else:
            self.choose_best_solution()


    # def generate_neighborhood(self):
    #     need_a_neighborhood = True
    #
    #     visited_clusters = []
    #     non_tabu_remove = []
    #     old_visited = []
    #
    #     unvisited_clusters = []
    #     non_tabu_insertion = []
    #     tabu_insertion = []
    #
    #     for i in self.clusters_can_be_updated:
    #         long_term_memory = self.array_clusters[i][2]
    #         # update the lists for remove
    #         if long_term_memory > 0:
    #             visited_clusters.append(i)
    #             if long_term_memory > self.tabu_alpha:
    #                 non_tabu_remove.append(i)
    #             elif long_term_memory > self.beta:
    #                 old_visited.append(i)
    #         else:
    #             """ it's a possible insertion if this cluster don't belong to a set who is in the solution """
    #             a = self.array_clusters[i][1]  # self.cluster_match_sets[i]
    #             none_set_was_visited = True
    #             for aa in a:
    #                 if self.sets_visited[aa] == 1:
    #                     none_set_was_visited = False
    #                     break
    #             # update the lists for insertion
    #             if none_set_was_visited:
    #                 unvisited_clusters.append(i)
    #                 if long_term_memory < -self.tabu_alpha:
    #                     non_tabu_insertion.append(i)
    #                 else:
    #                     tabu_insertion.append(i)
    #
    #     """ Non-Tabu Insertion """
    #     if any(non_tabu_insertion):
    #         """ Neighborhoods will be generated by a small modification of the current solution """
    #         neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #         chosen_cluster = np.random.choice(non_tabu_insertion)
    #         #chosen_cluster = self.insertion_criterion(non_tabu_insertion)
    #         #chosen_cluster = non_tabu_insertion[np.argmax([self.array_clusters[i][4] for i in non_tabu_insertion])]
    #         neighborhood.append(chosen_cluster)
    #         need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster)
    #         if not need_a_neighborhood:
    #             my_print(f"non_tabu_insertion {chosen_cluster} {non_tabu_insertion}")
    #         else:
    #             my_print(f"discarded Non-Tabu Insertion {chosen_cluster} {non_tabu_insertion}")
    #
    #     """ Old Removal """
    #     if need_a_neighborhood:
    #         if any(old_visited):
    #             """ Neighborhoods will be generated by a small modification of the current solution """
    #             neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #             chosen_cluster = np.random.choice(old_visited)
    #             neighborhood.remove(chosen_cluster)
    #             need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
    #             if not need_a_neighborhood:
    #                 my_print(f"old_Removal {chosen_cluster} {old_visited}")
    #             else:
    #                 my_print(f"discarded old_Removal {chosen_cluster} {old_visited}")
    #
    #     """ Tabu Insertion """
    #     if need_a_neighborhood:
    #         if any(tabu_insertion):
    #             neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #             #choose the cluster with the aspiration level criterion
    #             #chosen_cluster = np.random.choice(tabu_insertion)
    #             #chosen_cluster = self.insertion_criterion(tabu_insertion)
    #             chosen_cluster = tabu_insertion[np.argmax([self.array_clusters[i][4] for i in tabu_insertion])]
    #             neighborhood.append(chosen_cluster)
    #             need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster)
    #             if not need_a_neighborhood:
    #                 my_print(f"tabu_insertion {chosen_cluster} {tabu_insertion}")
    #             else:
    #                 my_print(f"discarded tabu_insertion {chosen_cluster} {tabu_insertion}")
    #
    #     """ Non-Tabu Removal """
    #     if need_a_neighborhood:
    #         if any(non_tabu_remove):
    #             """ Neighborhoods will be generated by a small modification of the current solution """
    #             neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #             chosen_cluster = np.random.choice(non_tabu_remove)
    #             neighborhood.remove(chosen_cluster)
    #             need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
    #             if not need_a_neighborhood:
    #                 my_print(f"Non-Tabu Removal {chosen_cluster} {non_tabu_remove}")
    #             else:
    #                 my_print(f"discarded Non-Tabu Removal {chosen_cluster} {non_tabu_remove}")
    #
    #     """ Random Removal """
    #     if need_a_neighborhood:
    #         if any(visited_clusters):
    #             """ Neighborhoods will be generated by a small modification of the current solution """
    #             neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #             chosen_cluster = np.random.choice(visited_clusters)
    #             neighborhood.remove(chosen_cluster)
    #             need_a_neighborhood = self.removal_neighborhood(neighborhood, chosen_cluster)
    #             if not need_a_neighborhood:
    #                 my_print(f"Random Removal {chosen_cluster} {visited_clusters}")
    #             else:
    #                 my_print(f"discarded Random Removal {chosen_cluster} {visited_clusters}")
    #
    #     """ Random Insertion """
    #     if need_a_neighborhood:
    #         if any(unvisited_clusters):
    #             """ Neighborhoods will be generated by a small modification of the current solution """
    #             neighborhood = copy.deepcopy(self.solution["subgroups_visited"])
    #             chosen_cluster = np.random.choice(unvisited_clusters)
    #             neighborhood.append(chosen_cluster)
    #             need_a_neighborhood = self.insertion_neighborhood(neighborhood, chosen_cluster)
    #             if not need_a_neighborhood:
    #                 my_print(f"Random Insertion {chosen_cluster} {unvisited_clusters}")
    #             else:
    #                 my_print(f"discarded Random Insertion {chosen_cluster} {unvisited_clusters}")
    #
    #     if need_a_neighborhood:
    #         for c in range(self.num_clusters):
    #             long_term_memory = self.array_clusters[i][2]
    #             if long_term_memory > 0:
    #                 long_term_memory += 1
    #             else:
    #                 long_term_memory -= 1
    #     else:
    #         self.choose_best_solution()

    def main(self):
        """ Compute an initial solution p0 """
        print("------ Initial Solution ---------------")
        self.initial_solution( )
        print(f"solution {self.solution}")
        self.choose_best_solution()

        print("------ Generating Neighborhoods ---------------")
        cont = 0
        while self.iterations_without_improvement < self.max_iterations_without_improvement and cont < self.absolute_max_iterations:
            self.generate_neighborhood()
            cont += 1
            if not self.cops.circular_path:
                if self.iterations_to_change_final_set > self.max_iterations_without_improvement:
                    self.change_end_cluster()
        print(' ------- Neighbors generated -------- ')
        """ add start end end to solution clusters_visited"""
        self.best_solution["subgroups_visited"].insert(0, self.start_cluster)
        # self.best_solution["subgroups_visited"].append([c for c in self.array_sets[self.end_cluster] if self.array_clusters[c][2] > 0][0])
        self.best_solution["subgroups_visited"].append(self.best_end_cluster)
        #########################################
        return self.best_solution


    # def main(self):
    #     """ Compute an initial solution p0 """
    #     my_print("------ Initial Solution ---------------")
    #     self.initial_solution()
    #     my_print(f"solution {self.solution}")
    #     self.choose_best_solution()
    #
    #     my_print("------ Generated Neighborhoods ---------------")
    #     cont = 0
    #     #   while cont < 200:
    #     while self.iterations_without_improvement < 300:
    #         self.generate_neighborhood()
    #         cont += 1
    #         if not self.cops.circular_path:
    #             if self.iterations_to_change_final_set > self.max_iterations_without_improvement:
    #                 self.change_end_cluster()
    #
    #     """ add start end end to solution clusters_visited"""
    #     self.best_solution["subgroups_visited"].insert(0, self.start_cluster)
    #     #self.best_solution["subgroups_visited"].append([c for c in self.array_sets[self.end_cluster] if self.array_clusters[c][2] > 0][0])
    #     self.best_solution["subgroups_visited"].append(self.best_end_cluster)
    #     #########################################
    #     return self.best_solution

    def initial_solution(self):
        """
        Generate an initial solution.Initial solution p0: For the initial solution p0, we randomly choose one cluster at a time,and for each cluster,we select the subgroup with the highest profitability level.
        The profitability level is defined here as the profit achieved by visiting the subgroup divided by the number of vertices of this subgroup. Then, this subgroup is inserted into p0, and a new tour is
        generated using the 2-opt strategy [16]. The procedure stops when all clusters have been considered or the tour surpasses the budget for Λ consecutive iterations.
        :return:
        """
        t1 = time.time()
        p0 = [self.start_cluster]
        visited_subgroups = []
        # clusters_visited = []
        """ Order randomly the cluster array, without the initial and final cluster """
        index_clusters = [i for i in range(len(self.clusters)) if i != self.start_cluster and i != self.end_cluster]
        np.random.shuffle(index_clusters)

        """ Make the start subgroup as visited """
        visited_subgroups.append(self.start_subgroup)
        # clusters_visited.append(self.start_cluster)
        profit = 0
        tour = []
        distance = 0
        """ For each cluster chose a subgroup with best profit and try to find a plausible path """
        early_stop = self.max_initial_solution_attempts
        
        while any(index_clusters) and early_stop > 0:
            """ Chose the cluster randomly """
            c = np.random.choice(index_clusters)
            max_subgroup_profit_idx = np.argmax(self.clusters[c].index_profit)

            """ Chose the subgroup with best profit in the cluster """
            choose_subgroup = self.clusters[c].subgroups[max_subgroup_profit_idx]

            """ If cluster is already visited to initial solution it can't be visited again """
            if self.clusters[c].can_be_inserted:
                p0.append(choose_subgroup)

                initial_tour, initial_distance = self.tour_generation(p0, improvement_threshold=self.settings["improvement threshold"]) #Generate a route between vertices

                """ test if the solution is plausible """
                if initial_distance > self.cops.t_max:
                    early_stop -= 1
                    p0.pop(-1)
                    try:
                        index_clusters.remove(c)
                    except ValueError:
                        pass
                else:
                    early_stop = self.max_initial_solution_attempts
                    tour = initial_tour
                    distance = initial_distance
                    profit += self.array_profit[choose_subgroup]
                    # clusters_visited.append(a)
                    for visited in self.clusters[c].subgroups:
                        try:
                            index_clusters.remove(visited)
                        except ValueError:
                            pass
                    visited = self.clusters[c].subgroups[max_subgroup_profit_idx]
                    visited_subgroups.append(visited)
                    """ Set all clusters with this subgroup as visited """
                    for c in self.subgroups[visited].clusters_subgroup:
                        self.clusters[c].can_be_inserted = False


                    my_print(f"clusters in solution {p0} ")
                # my_print("-------------")

        """ For non_circular_path add the final cluster and set in their respective array """
        if not self.cops.circular_path:
            visited_subgroups.append(self.end_subgroup)
            # clusters_visited.append(self.end_cluster)

        self.solution["route"] = tour
        self.solution["subgroups_visited"] = p0
        # self.solution["subgroups_visited"].sort()
        # self.solution["visited_subgroups"] = visited_subgroups
        self.solution["distance"] = distance
        self.solution["profit"] = profit

        for i in p0:
            for c in self.subgroups[i].clusters_subgroup:
                self.clusters[c].long_term_memory = 1
            for v in self.subgroups[i].vertices_subgroup:
                self.vertices[v].visited = True  # indicates if a vertex is being visited
        self.vertices[self.clusters[self.end_cluster].vertices_of_cluster[0]].visited = True  # all end and start cluster have only one vertex
        self.vertices[self.clusters[self.start_cluster].vertices_of_cluster[0]].visited = True
        self.visited_subgroups[[i for i in visited_subgroups]] = 1

        """ update Aspiration level """
        for s in self.solution["subgroups_visited"]:
            for c in self.subgroups[s].clusters_subgroup:
                profit_index = self.clusters[c].subgroups.index(s)
                if profit > self.clusters[c].index_profit[profit_index]:
                    self.clusters[c].index_profit[profit_index] = profit

        my_print(f"LONG_TERM_MEMORY {[c.long_term_memory for c in self.clusters]}")
        my_print(f"visited_subgroups {self.visited_subgroups}")
        my_print(f"vertex_visited {[i.visited for i in self.vertices]}")

        execution_time = time.time() - t1
        print("Runtime Init Solution: {} seconds".format(execution_time))


    # def initial_solution(self):
    #     t1 = time.time()
    #     cv0 = []
    #     sets_visited = []
    #     # clusters_visited = []
    #     """ Order randomly the set array, without the initial and final sets """
    #     index_set = [i for i in range(len(self.array_sets)) if i != self.start_set and i != self.end_set]
    #     np.random.shuffle(index_set)
    #
    #     sets_visited.append(self.start_set)
    #     # clusters_visited.append(self.start_cluster)
    #     profit = 0
    #     tour = []
    #     distance = 0
    #     """ For each set chose a cluster and try to find a plausible path """
    #     early_stop = self.max_initial_solution_attempts
    #     while any(index_set) and early_stop > 0:
    #         s = np.random.choice(index_set)
    #         #a = np.random.choice(self.array_sets[s])  # Chose randomly a cluster from set
    #         indexMaxProfitPerNunCluster = np.argmax([self.array_clusters[c][6] for c in self.array_sets[s]])  # Chose the cluster from set with the highest profit
    #         a = self.array_sets[s][indexMaxProfitPerNunCluster]
    #         if self.array_clusters[a][5]:  # if this cluster can be inserted
    #             cv0.append(a)
    #
    #             initial_tour, initial_distance = self.tour_generation(cv0)
    #
    #             """ test if the solution is plausible """
    #             if initial_distance > self.cops.t_max:
    #                 early_stop -= 1
    #                 cv0.pop(-1)
    #                 try:
    #                     index_set.remove(s)
    #                 except ValueError:
    #                     pass
    #             else:
    #                 early_stop = self.max_initial_solution_attempts
    #                 tour = initial_tour
    #                 distance = initial_distance
    #                 profit += self.array_profit[a]
    #                 # clusters_visited.append(a)
    #                 for visited in self.array_clusters[a][1]:  # list of sets who this cluster belongs it
    #                     try:
    #                         index_set.remove(visited)
    #                         sets_visited.append(visited)
    #                         for c in self.array_sets[visited]:
    #                             self.array_clusters[c][5] = 0  # can't be visited
    #                     except ValueError:
    #                         pass
    #                 my_print(f"clusters in solution {cv0} ")
    #             # my_print("-------------")
    #
    #     """ For non_circular_path add the final cluster and set in their respective array """
    #     if not self.cops.circular_path:
    #         sets_visited.append(self.end_set)
    #         # clusters_visited.append(self.end_cluster)
    #
    #     self.solution["route"] = tour
    #     self.solution["subgroups_visited"] = cv0
    #     # self.solution["subgroups_visited"].sort()
    #     # self.solution["sets_visited"] = sets_visited
    #     self.solution["distance"] = distance
    #     self.solution["profit"] = profit
    #
    #     for i in cv0:
    #         self.array_clusters[i][2] = 1  # update LONG_TERM_MEMORY
    #         for v in self.array_clusters[i][0]:
    #             self.array_vertex[v][1] = 1  # indicates if a vertex is being visited
    #     self.array_vertex[self.array_clusters[self.end_cluster][0][0]][1] = 1  # all end and start cluster have only one vertex
    #     self.array_vertex[self.array_clusters[self.start_cluster][0][0]][1] = 1
    #     self.sets_visited[[i for i in sets_visited]] = 1
    #
    #     """ update Aspiration level """
    #     for c in self.solution["subgroups_visited"]:
    #         if profit > self.array_clusters[c, 4]:
    #             self.array_clusters[c, 4] = profit
    #
    #     my_print(f"LONG_TERM_MEMORY {[c[2] for c in self.array_clusters]}")
    #     my_print(f"sets_visited {self.sets_visited}")
    #     my_print(f"vertex_visited {[i[1] for i in self.array_vertex]}")
    #
    #     tempoExec = time.time() - t1
    #     print("Runtime Init Solution: {} seconds".format(tempoExec))

    def tour_generation(self,cluster_tg: List[int], improvement_threshold=0.001):
        """
        A tour is generated with 2-opt technic
        :param cluster_tg: Subgroups list .The vertex in this list will be optimized.
        :param improvement_threshold: Minimum improvement to consider a solution to 2-opt stable. This value can be configured in config.yaml
        :return: edges: The indexes of the vertex
        :return: d: Total distance
        """
        print(' ****** Starting tour generation ******* ')
        t_tg1 = time.time()
        cv0 = cluster_tg.copy()

        """ Note: the 2-opt solver needs a different treatment for a path that ends at the same vertex it started """
        if not self.cops.circular_path:
            cv0.append(self.end_cluster)

        """ select only wanted vertex """
        #selected_index = [v for i in cv0 for v in self.array_clusters[i][0]]
        selected_index = list(OrderedDict.fromkeys(cv0))  # will eliminate repeated elements
        # selected_vertex = np.array([vertices_tg[i].position for i in subgroup_tg[g].vertices_subgroup for g in selected_index])
        selected_vertex_l = []
        index_vertices = []
        for g in selected_index:
            for i in self.subgroups[g].vertices_subgroup:
                selected_vertex_l.append(self.vertices[i].position)
                index_vertices.append(i)
        selected_vertex = np.array(selected_vertex_l)
        """ calc the route from two-opt algorithm """
        route = two_opt(selected_vertex, improvement_threshold, is_a_circular_path=self.cops.circular_path)
        real_route = [index_vertices[i] for i in route]  # the real route is mapped from route

        """ define the edges from the found route """
        edges = [(real_route[i], real_route[i + 1]) for i in range(len(real_route) - 1)]
        if self.cops.circular_path:  # increase the last edge for a circular path
            edges.append((real_route[-1], real_route[0]))
        distance = path_distance_for_circular(route, selected_vertex)  # only for conference
        """ calculate the path distance from edge distances matrix """
        d = sum([self.matrix_dist[edge[0]][edge[1]] for edge in edges])
        print('======= Ending tour generation ======== ')
        total_time = time.time() - t_tg1
        print(f'Total time tour generation: {total_time}')
        return edges, d


    # def tour_generation(self, clusters, improvement_threshold=0.001):
    #     """ tour is generated with 2-opt technic """
    #
    #     cv0 = clusters.copy()
    #     cv0.insert(0, self.start_cluster)
    #
    #     """ Note: the 2-opt solver needs a different treatment for a path that
    #                     ends at the same vertex it started """
    #     if not self.cops.circular_path:
    #         cv0.append(self.end_cluster)
    #
    #     """ select only wanted vertex """
    #     #selected_index = [v for i in cv0 for v in self.array_clusters[i][0]]
    #     selected_index = list(OrderedDict.fromkeys([v for i in cv0 for v in self.array_clusters[i][0]]))  # will eliminate repeated elements
    #     selected_vertex = np.array([self.array_vertex[i][0] for i in selected_index])
    #
    #     """ calc the route from two-opt algorithm """
    #     route = two_opt(selected_vertex, improvement_threshold, is_a_circular_path=self.cops.circular_path)
    #     real_route = [selected_index[i] for i in route]  # the real route is mapped from route
    #
    #     """ define the edges from the found route """
    #     edges = [(real_route[i], real_route[i + 1]) for i in range(len(real_route) - 1)]
    #     if self.cops.circular_path:  # increase the last edge for a circular path
    #         edges.append((real_route[-1], real_route[0]))
    #     distance = path_distance_for_circular(route, selected_vertex)  # only for conference
    #     """ calculate the path distance from edge distances matrix """
    #     d = sum([self.matrix_dist[edge[0]][edge[1]] for edge in edges])
    #
    #     return edges, d

    def choose_best_solution(self):
        """
        Verify if the current solution is better or worse than the best solution. If the solution profit is bigger than profit of best solution make it best solution.
        If it is equal, verify if the distance is smaller if it is set the solution as best solution.
        :return:
        """
        if self.solution["profit"] > self.best_solution["profit"] or \
                (self.solution["profit"] == self.best_solution["profit"] and
                 self.solution["distance"] < self.best_solution["distance"]):
            self.best_solution = copy.deepcopy(self.solution)
            self.iterations_without_improvement = 0
            self.iterations_to_change_final_set = 0
            self.best_end_cluster = self.end_cluster
        else:
            self.iterations_without_improvement += 1
            self.iterations_to_change_final_set += 1

    def change_end_cluster(self):
        """
        Change the cluster that ends the path
        :return:
        """
        #my_print(f"-----------CHANGE THE END POINT {self.end_cluster}", index_print=2)

        # index of the min long-term-memory for the end clusters
        a = np.argmin([self.clusters[c].long_term_memory for c in self.subgroups[self.end_subgroup].clusters_subgroup])

        # old end cluster long-term-memory update
        self.clusters[self.end_cluster].long_term_memory = 0  # long term memory (=0 removed from solution)
        self.vertices[self.clusters[self.end_cluster].vertices_of_cluster[0]].visited = False  # vertex removed from solution

        # new end cluster will be the oldest it was in solution
        self.end_cluster = self.subgroups[self.end_subgroup].clusters_subgroup[a]
        self.clusters[self.end_cluster].long_term_memory = 1  # long term memory (>=1 in solution)
        self.vertices[self.clusters[self.end_cluster].vertices_of_cluster[0]].visited = True

        #my_print(f" changed to {self.end_cluster}", index_print=2)

        # reboot iterations counter
        self.iterations_to_change_final_set = 0
