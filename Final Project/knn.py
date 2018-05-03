import numpy as np
import math
import matplotlib.pyplot as plt
import surprise
from matplotlib.collections import LineCollection


import os
from surprise import Dataset
from surprise import KNNBaseline
from surprise import Reader



# defines the reward/connection graph
r = np.array([[0, 0, 0, 0,  0, 0],
              [0, 0, 0, 0,  0, 0],
              [0, 0, 0, 0,  0, 0],
              [0, 0, 0, 0,  0, 0],
              [0, 0, 0, 0,  0, 0],
			  [0, 0, 0, 0,  0, 0]]).astype("float32")
q = np.zeros_like(r)


def update_q(state, next_state, action, alpha, gamma):
    rsa = r[state, action]
    qsa = q[state, action]
    new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]


def show_traverse():
    # show all the greedy traversals
    for i in range(len(q)):
        current_state = i
        traverse = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_state = np.argmax(q[current_state])
            current_state = next_state
            traverse += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        traverse = traverse[:-4]
        print("Greedy traversal for starting state %i" % i)
        print(traverse)
        print("")


def show_q():
    # show all the valid/used transitions
    coords = np.array([[2, 2],
                       [4, 2],
                       [5, 3],
                       [4, 4],
                       [2, 4],
                       [5, 2]])
    # invert y axis for display
    coords[:, 1] = max(coords[:, 1]) - coords[:, 1]

    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    plt.scatter(coords[:, 0], coords[:, 1], c='r')

    start_idx, end_idx = np.where(q > 0)
    segments = [[coords[start], coords[stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.array(q[q > 0])
    # bump up values for viz
    values = values
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r)
    lc.set_array(values)
    ax.add_collection(lc)

    verticalalignment = 'top'
    horizontalalignment = 'left'
    for i in range(len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        name = str(i)
        if i == 1:
            y = y - .05
            x = x + .05
        elif i == 3:
            y = y - .05
            x = x + .05
        elif i == 4:
            y = y - .05
            x = x + .05
        else:
            y = y + .05
            x = x + .05

        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(float(len(coords))),
                           alpha=.6))
    plt.show()

reader = Reader(line_format='user item rating', sep=' ', skip_lines=3, rating_scale=(1, 5))

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

sim_options = {
    'name': 'cosine',
    'user_based': True # compute  similarities between items
}

algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)

w, h = 1683, 944;
matrix = [[0 for x in range(w)] for y in range(h)]

for i in range (1,944):
    for j in range (1,1683):
        pred = algo.predict(uid=str(i), iid=str(j), r_ui=0, verbose=False)
        matrix[i][j] = pred.est
        #print (matrix[i][j])


#a = np.random.choice([0,1,2,3,4,5], (943,1682))
#print(a.shape)
#print(a[:])

a = np.zeros_like(matrix)
f = open('u.txt','r')
message = f.read()
m = []
m = message.split("\n")
y = 0
for i in range(len(m)) :
	p = m[i].split("\t")
	for j in range(3) :
		y = y + int(p[2])
		a[int(p[0])][int(p[1])] = y
		y = 0
f.close()

i=0
l = []
while i < 944:
    j=0
    list1 = [] 
    while j < 1683:
        if a[i][j] != 0:
            list1.append(j)
        j += 1 
    l.append(list1)
    i += 1
#for i in range(len(l)):
#    print(l[i])

# Core algorithm
gamma = 0.8
alpha = 1.0
n_episodes = 100
n_states = 6
n_actions = 6
epsilon = 0.05
random_state = np.random.RandomState(1999)
for e in range(int(n_episodes)):
	states = list(range(n_states))
	random_state.shuffle(states)
	for i in range(943) :
		if len(l[i])>0:
			current_state = a[i][l[i][0]]
			for j in range(len(l[i])-2) :
				# epsilon greedy
				'''valid_moves = r[current_state] >= 0
				if random_state.rand() < epsilon:
				actions = np.array(list(range(n_actions)))
				actions = actions[valid_moves == True]
				if type(actions) is int:
					actions = [actions]
				random_state.shuffle(actions)
				action = actions[0]
				next_state = action
				else:
				if np.sum(q[current_state]) > 0:
					action = np.argmax(q[current_state])
				else:
					# Don't allow invalid moves at the start
					# Just take a random move
					actions = np.array(list(range(n_actions)))
					actions = actions[valid_moves == True]
					random_state.shuffle(actions)
					action = actions[0]
				next_state = action'''
				next_state = a[i][l[i][j+1]]
				action = next_state
				r[int(a[i][l[i][j]])][int(a[i][l[i][j+1]])] =  a[i][l[i][j+2]] - matrix[i][j]   
				q[int(current_state)][int(action)] = update_q(int(current_state), int(next_state), int(action),alpha=alpha, gamma=gamma)
				current_state = next_state

#print(q)
#show_traverse()
#show_q()
b = np.zeros_like(matrix)

for i in range(943) :
	for j in range(2,1682) :
		b[i][j] = matrix[i][j] + q[int(a[i][j-2])][int(a[i][j-1])]

for i in range(943) : 
	print(b[i])
surprise.evaluate(algo, data, measures=['RMSE'])
cn = 0
x = 0

for i in range(943) :
	for j in range(1682) :
		if int(a[i][j]) != 0 :
			x = x+(int(a[i][j])-b[i][j])*(int(a[i][j])-b[i][j])
			cn += 1 
print("************")			
ans = x/cn
ans = math.sqrt(ans)
print(ans)
