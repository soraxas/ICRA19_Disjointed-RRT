import logging

from overrides import overrides

from helpers import *
from planners.randomPolicySampler import RandomPolicySampler
from planners.rrdtPlanner import RRdTPlanner, RRdTSampler, Node

############################################################
##              Disjointed Particles Sampler              ##
############################################################

LOGGER = logging.getLogger(__name__)

MAX_NUMBER_NODES = 20000



class MultiRRTSampler(RRdTSampler):

    @overrides
    def init(self, **kwargs):
        super().init(**kwargs)

        self.randomSampler = RandomPolicySampler()
        self.randomSampler.init(**kwargs)

        self.particle_root_pos = []
        for p in self.p_manager.particles:
            self.particle_root_pos.append(p.pos.copy())

    @overrides
    def get_next_pos(self):
        choice = self.get_random_choice()
        pos = self.randomSampler.get_valid_next_pos()[0]
        return (pos, None,
                None,
                lambda **kwargs: 1,
                lambda **kwargs: 1)

    @overrides
    def paint(self, window):
        if self._last_prob is None:
            return
        for i, p in enumerate(self.particle_root_pos):
            self.particles_layer.fill((255, 128, 255, 0))
            # get a transition from green to red
            self.args.env.draw_circle(pos=p, colour=Colour.blue, radius=3, layer=self.particles_layer)
            window.blit(self.particles_layer, (0, 0))


class MultiRRTPlanner(RRdTPlanner):

    @overrides
    def run_once(self):
        # Get an sample that is free (not in blocked space)
        _tmp = self.args.sampler.get_valid_next_pos()
        if _tmp is None:
            # we have added a new samples when respawning a local sampler
            return
        rand_pos, parent_tree, last_node, report_success, report_fail = _tmp

        # idx = np.random.randint(0, len(self.args.sampler.tree_manager.disjointedTrees))
        # parent_tree = self.args.sampler.tree[idx]
        parent_tree = np.random.choice((self.args.sampler.tree_manager.root, *self.args.sampler.tree_manager.disjointedTrees))


        ###################33
        idx = self.find_nearest_neighbour_idx(
            rand_pos, parent_tree.poses[:len(parent_tree.nodes)])
        nn = parent_tree.nodes[idx]
        # get an intermediate node according to step-size
        newpos = self.args.env.step_from_to(nn.pos, rand_pos)
        ##########################333333333
        # check if it is free or not ofree
        if not self.args.env.cc.path_is_free(nn.pos, newpos):
            self.args.env.stats.add_invalid(obs=False)
            report_fail(pos=rand_pos, free=False)
        else:
            newnode = Node(newpos)
            self.args.env.stats.add_free()
            self.args.sampler.add_tree_node(newnode.pos)
            report_success(newnode=newnode, pos=newnode.pos)
            ######################
            newnode, nn = self.args.sampler.tree_manager.connect_two_nodes(
                newnode, nn, parent_tree)
            # try to add this newnode to existing trees
            self.args.sampler.tree_manager.add_pos_to_existing_tree(
                newnode, parent_tree)
