import random

import numpy as np
from overrides import overrides

from planners.baseSampler import Sampler
from env import Node
from planners.randomPolicySampler import RandomPolicySampler
from planners.rrtPlanner import RRTPlanner
from helpers import Colour


class MultiRRTSampler(Sampler):
    @overrides
    def init(self, **kwargs):
        super().init(**kwargs)
        self.randomSampler = RandomPolicySampler()
        self.randomSampler.init(**kwargs)

    @overrides
    def get_next_pos(self):
        # Random path
        while True:
            if random.random() < self.args.goalBias:
                # init/goal bias
                p = self.goal_pos
            else:
                p = self.randomSampler.get_next_pos()[0]
            return p, self.report_success, self.report_fail

class Tree():
    def __init__(self, root_node, max_num, is_root_tree=False):
        self.poses = np.empty((max_num + 50, 2)) # +50 to prevent overflowing
        self.nodes = [root_node]
        self.is_root_tree = is_root_tree

    def get_nodes_poses(self):
        return self.nodes, self.poses

    def clear(self):
        del self.nodes
        del self.poses

class MultiRRTPlanner(RRTPlanner):
    @overrides
    def init(self, *argv, **kwargs):
        super().init(*argv, **kwargs)
        self.N = 10
        self.found_solution = False
        self.trees = []

        # create root tree
        root_tree = Tree(None, max_num=self.args.max_number_nodes,
                               is_root_tree=True)
        root_tree.poses = self.poses
        root_tree.nodes = self.nodes
        self.trees.append(root_tree)

        # List_of_clutter_coords = [
        #     (-1, -1),
        #
        # ]
        for i in range(self.N):
            if i == 0:
                tree = Tree(self.args.env.goalPt, self.args.max_number_nodes)
                tree.poses[0] = self.args.env.goalPt.pos
                self.trees.append(tree)
            else:
                rand_pos, _, _ = self.args.sampler.get_valid_next_pos()

                tree = Tree(Node(rand_pos), self.args.max_number_nodes)
                tree.poses[0] = rand_pos
                self.trees.append(tree)


    def join_trees(self, tree1, node1, tree2, node2):
        """Given two trees and two nodes, such that:
            node1 from tree1,
            node2 from tree2, and
            node1 can be connect to node 2,
        tree2 will be joined to tree1 and tree2 will then be discarded."""

        nn = node1
        cur_node = node2
        parent = cur_node.parent

        while parent is not None:
            parent = cur_node.parent

            cur_node, nn = self.choose_least_cost_parent(
                cur_node, nn=nn, nodes=tree1.nodes)
            self.rewire(cur_node, nodes=tree1.nodes)

            tree1.poses[len(tree1.nodes)] = cur_node.pos
            tree1.nodes.append(cur_node)

            nn = cur_node
            cur_node = parent

        tree2.clear()

    def check_found_solution(self):
        if not self.found_solution:
            # check for if the goal's parents can be chained to start pt
            nn = self.goalPt.parent
            while nn is not None and nn != self.startPt:
                nn = nn.parent
            if nn == self.startPt:
                self.found_solution = True
        return self.found_solution

    @overrides
    def run_once(self):
        # randomise the tree to be add to.
        rand_idx = np.random.randint(0, len(self.trees))
        tree = self.trees[rand_idx]
        nodes, poses = tree.get_nodes_poses()

        ###################################################################

        rand_pos, _, _ = self.args.sampler.get_valid_next_pos()
        # Found a node that is not in X_obs
        idx = self.find_nearest_neighbour_idx(rand_pos, poses[:len(nodes)])
        nn = nodes[idx]
        # get an intermediate node according to step-size
        newpos = self.args.env.step_from_to(nn.pos, rand_pos)
        # check if it has a free path to nn or not
        if not self.args.env.cc.path_is_free(nn.pos, newpos):
            self.args.env.stats.add_invalid(obs=False)
        else:
            newnode = Node(newpos)
            self.args.env.stats.add_free()

            ######################
            newnode, nn = self.choose_least_cost_parent(
                newnode, nn, nodes=nodes)
            poses[len(nodes)] = newnode.pos

            nodes.append(newnode)
            # rewire to see what the newly added node can do for us
            self.rewire(newnode, nodes)
            self.args.env.draw_path(nn, newnode)

            ###################################################################
            # check if two tree joins
            for other_tree in self.trees:
                other_nodes, other_poses = other_tree.get_nodes_poses()

                if nodes is other_nodes:
                    # skip comparing to itself
                    continue

                distances = np.linalg.norm(
                    other_poses[:len(self.nodes)] - newpos, axis=1)
                if min(distances) < self.args.epsilon:
                    idx = np.argmin(distances)
                    if self.args.env.cc.path_is_free(other_poses[idx], newpos):
                        # the two tree is able to join!

                        # check to see either of them is root tree
                        # if newnode in self.nodes or other_nodes[idx] in self.nodes:

                        if newnode in self.nodes:
                            node1 = newnode
                            tree1 = tree
                            node2 = other_nodes[idx]
                            tree2 = other_tree
                        else:
                            node1 = other_nodes[idx]
                            tree1 = other_tree
                            node2 = newnode
                            tree2 = tree

                        # Yes! add everything to root tree
                        self.join_trees(tree1, node1, tree2, node2)
                        self.trees.remove(tree2)

                        # assert node1 in self.nodes
                        # assert node2 in other_nodes
                    self.args.env.update_screen(True)
            if self.goalPt.parent is not None:
                # check if this lead to start pt
                if self.check_found_solution():
                    if self.goalPt.parent.cost < self.c_max:
                        self.c_max = self.goalPt.parent.cost
                        self.draw_solution_path()

    @overrides
    def paint(self):
        # return
        drawn_nodes_pairs = set()
        for tree in self.trees:
            # draw root
            self.args.env.draw_circle(
                pos=tree.nodes[0].pos,
                colour=Colour.blue,
                radius=self.args.goal_radius // 4,
                layer=self.args.env.path_layers)
            # draw edges
            for n in tree.nodes:
                if n.parent is not None:
                    new_set = frozenset({n, n.parent})
                    if new_set not in drawn_nodes_pairs:
                        drawn_nodes_pairs.add(new_set)
                        self.args.env.draw_path(n, n.parent)
