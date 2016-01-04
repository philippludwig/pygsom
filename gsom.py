# -*- coding: utf-8 -*-
"""
The MIT License (MIT)

Copyright (c) 2015 Philipp Ludwig <git@philippludwig.net>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""@package GSOM

This is an implementation of the growing self-organizing map.

Different possible approaches for the GSOM have been presented in the past
by various researchers. To make things clear, this implementation is based
on the one described in the work of:

Alahakoon, Damminda, S. Halgamuge, and Bala Srinivasan:
"Dynamic self-organizing maps with controlled growth for knowledge discovery."
Neural Networks, IEEE Transactions on 11.3 (2000): 601-614.

Sadly, this article is not as comprehensive as desirable. Therefore this
implementation should not be taken as a reference, but as a best-effort
version. Some details of the algorithm have been assembled based on the
work of Mengxue Cao et. al, who described their approach within their work:

"Growing Self-Organizing Map Approach for Semantic Acquisition Modeling
way within their work"

Refer to both papers for further details.

Additionally, this algorithm picks up some of the aspects proposed in the
work of:

Andreas Nürnberger and Marcin Detyniecki:
"Externally growing self-organizing maps and its application to e-mail
 database visualization and exploration"
"""
from math import log, exp
import itertools
import math
import random
import scipy

class GSOM_Node:
	""" Represents one node in a growing SOM. """
	R = random.Random()


	def __init__(self, dim, x, y):
		""" Initialize this node. """
		# Create a weight vector of the given dimension:
		# Initialize the weight vector with random values between 0 and 1.
		self.weights=scipy.array([self.R.random() for _ in range(dim)])

		# Remember the error occuring at this particular node
		self.error = 0.0

		# Holds the number of the iteration during the node has been inserted.
		self.it = 0

		# Holds the number of the last iteration where the node has won.
		self.last_it = 0

		# Holds the best-matching data.
		self.data = None
		self.last_changed = 0

		# This node has no neighbours yet.
		self.right = None
		self.left  = None
		self.up    = None
		self.down  = None

		# Copy the given coordinates.
		self.x, self.y = x, y


	def adjust_weights(self, target, learn_rate):
		""" Adjust the weights of this node. """
		for w in range(0, len(target)):
			self.weights[w] += learn_rate * (target[w] - self.weights[w])


	def is_boundary(self):
		""" Check if this node is at the boundary of the map. """
		if not self.right: return True
		if not self.left:  return True
		if not self.up:    return True
		if not self.down:  return True
		return False


class GSOM:
	""" Represents a growing self-organizing map. """

	def _distance(self, v1, v2):
		""" Calculate the euclidean distance between two scipy arrays."""
		dist = 0.0
		for v, w in zip(v1, v2):
			dist += pow(v - w,2)
		return dist


	def _find_bmu(self, vec):
		""" Find the best matching unit within the map for the given input
		    vector. """
		dist=float("inf")
		winner = False
		for node in self.nodes:
			d = self._distance(vec, node.weights)
			if(d < dist):
				dist = d
				winner = node

		return winner


	def _find_similar_boundary(self, node):
		""" Find the most similar boundary node to the given node. """
		dist = float("inf")
		winner = False
		for boundary in self.nodes:
			if not boundary.is_boundary(): continue
			if boundary == node: continue

			d = self._distance(node.weights, boundary.weights)
			if d < dist:
				dist = d
				winner = node

		return winner


	def __init__(self, dataset, spread_factor = 0.5):
		""" Initializes this GSOM using the given data. """
		# Assign the data
		self.data = []
		for fn,t in dataset:
			arr = scipy.array(t)
			self.data.append([fn,arr])

		# Determine the dimension of the data.
		self.dim = len(self.data[0][1])

		# Calculate the growing threshold:
		self._GT = -self.dim * math.log(spread_factor, 2)

		# Create the 4 starting Nodes.
		self.nodes = []
		n00 = GSOM_Node(self.dim, 0, 0)
		n01 = GSOM_Node(self.dim, 0, 1)
		n10 = GSOM_Node(self.dim, 1, 0)
		n11 = GSOM_Node(self.dim, 1, 1)
		self.nodes.extend([n00,n01,n10,n11])

		# Create starting topology
		n00.right = n10
		n00.up    = n01
		n01.right = n11
		n01.down  = n00
		n10.up    = n11
		n10.left  = n00
		n11.left  = n01
		n11.down  = n10

		# Set properties
		self.it = 0		       # Current iteration
		self.max_it = len(self.data)
		self.num_it = 1000     # Total iterations
		self.init_lr = 0.1     # Initial value of the learning rate
		self.alpha = 0.1
		self.output = file("gsom.csv","w")


	def train(self):
		# Select the next input.
		input = random.choice(self.data)[1]

		# Calculate the learn rate.
		# Note that the learning rate, according to the original paper,
		# is resetted for every new input.
		learn_rate = self.init_lr * self.alpha * (1 - 1.5/len(self.nodes))

		# We now present the input several times to the network.
		# It is unclear what's a good number here, since no publication
		# took the effort to name a value. However, the implementation
		# provided by Arkadi Kagan presents the input 20 times, so we
		# will copy that here.
		recalc_nodes = []
		for _ in range(20):
			# Find the best matching unit
			BMU = self._find_bmu(input)
			BMU.last_it = self.it

			# Adapt the weights of the direct topological neighbours
			neighbours = []
			neighbours.append(BMU)
			if BMU.left:  neighbours.append(BMU.left)
			if BMU.right: neighbours.append(BMU.right)
			if BMU.up:    neighbours.append(BMU.up)
			if BMU.down:  neighbours.append(BMU.down)

			if BMU not in recalc_nodes: recalc_nodes.append(BMU)

			for node in neighbours:
				node.adjust_weights(input, learn_rate)
				if node not in recalc_nodes: recalc_nodes.append(node)

			# Calculate the error.
			err = self._distance(BMU.weights, input)

			# Add the error to the node.
			growing, nodes = self._node_add_error(BMU, err)
			if growing: recalc_nodes.extend(nodes)

		# Count the iteration
		self.it += 1

		# Re-Calc representative data elements for changed nodes.
		used_data = []
		for node in self.nodes:
			used_data.append(node.data)

		for node in recalc_nodes:
			dist = float("inf")
			winner = False
			winner_fn = False

			for fn,point in self.data:
				if fn in used_data: continue

				d = self._distance(point, node.weights)
				if(d < dist):
					dist = d
					winner = point
					winner_fn = fn

			if node.data != winner_fn:
				node.data = winner_fn
				node.last_changed = self.it
			self.output.write(node.data + "," + str(node.x) + "," + str(node.y)\
					+ ",change\n")
			used_data.append(winner_fn)

		# Remove unused nodes.
		self._remove_unused_nodes()


	def _node_add_error(self, node, error):
		""" Add the given error to the error value of the given node.

		    This will also take care of growing the map (if necessary) and
			distributing the error along the neighbours (if necessary) """
		node.error += error

		# Consider growing
		if node.error > self._GT:
			if not node.is_boundary():
				# Find the boundary node which is most similar to this node.
				node = self._find_similar_boundary(node)
				if not node:
					print("GSOM: Error: No free boundary node found!")

				""" Old method:
				# Distribute the error along the neighbours.
				# Since this is not a boundary node, this node must have
				# 4 neighbours.
				node.error = 0.5 * self._GT
				node.left.error += 0.25 * node.left.error
				node.right.error += 0.25 * node.right.error
				node.up.error += 0.25 * node.up.error
				node.down.error += 0.25 * node.down.error
				"""
			nodes = self._grow(node)
			return True, nodes

		return False, 0


	def _grow(self, node):
		""" Grow this GSOM. """
		# We grow this GSOM at every possible direction.
		nodes = []
		if node.left == None:
			nn = self._insert(node.x - 1, node.y, node)
			nodes.append(nn)
			print("Growing left at: (" + str(node.x) + "," + str(node.y)\
					+ ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

		if node.right == None:
			nn = self._insert(node.x + 1, node.y, node)
			nodes.append(nn)
			print("Growing right at: (" + str(node.x) + "," + str(node.y)\
					+ ") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

		if node.up == None:
			nn = self._insert(node.x, node.y + 1, node)
			nodes.append(nn)
			print("Growing up at: (" + str(node.x) + "," + str(node.y) +\
					") -> (" + str(nn.x) + ", " + str(nn.y) + ")")

		if node.down == None:
			nn = self._insert(node.x, node.y - 1, node)
			nodes.append(nn)
			print("Growing down at: (" + str(node.x) + "," + str(node.y) +\
					") -> (" + str(nn.x) + ", " + str(nn.y) + ")")
		return nodes


	def _insert(self, x, y, init_node):
		# Create new node
		new_node = GSOM_Node(self.dim, x, y)
		self.nodes.append(new_node)

		# Save the number of the current iteration. We need this to prune
		# this node later (if neccessary).
		new_node.it = new_node.last_it = self.it

		# Create the connections to possible neighbouring nodes.
		for node in self.nodes:
			# Left, Right, Up, Down
			if node.x == x - 1 and node.y == y:
				new_node.left = node
				node.right = new_node
			if node.x == x + 1 and node.y == y:
				new_node.right = node
				node.left = new_node
			if node.x == x and node.y == y + 1:
				new_node.up = node
				node.down = new_node
			if node.x == x and node.y == y - 1:
				new_node.down = node
				node.up = new_node

		# Calculate new weights, look for a neighbour.
		neigh = new_node.left
		if neigh == None: neigh = new_node.right
		if neigh == None: neigh = new_node.up
		if neigh == None: neigh = new_node.down
		if neigh == None: print("_insert: No neighbour found!")

		for i in range(0, len(new_node.weights)):
			new_node.weights[i] = 2 * init_node.weights[i] - neigh.weights[i]

		return new_node


	def _remove_unused_nodes(self):
		""" Remove all nodes from the GSOM that have not been used. """
		to_remove = []

		# Iterate over all nodes.
		for node in self.nodes:
			# Different rules for nodes that have been used or not.
			iterations_not_won = self.it - node.last_it

			# If we have 50 nodes, every node is allowed not to win 50 times
			# in a row. This means every node must be picked at least once.
			if iterations_not_won < len(self.nodes) * 4.0 * (1 + self.it/len(self.data)) : continue


			# First, remove the connections to the neighbouring nodes.
			if node.left:  node.left.right = None
			if node.up:    node.up.down    = None
			if node.down:  node.down.up    = None
			if node.right: node.right.left = None

			# Save this node for removing.
			to_remove.append(node)

		# Now remove all marked nodes.
		for node in to_remove:
			print("Removing node @ " + str(node.x) + ", " + str(node.y) + \
			      " - Current it: " + str(self.it) + " - Last time won: " +\
				  str(node.last_it))
			if node.data:
				self.output.write(node.data + "," + str(node.x)+","+str(node.y)\
					+ ",remove\n")
			self.nodes.remove(node)

