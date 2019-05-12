import multiprocessing as MP
import numpy as NP
import time
import random
from time import sleep


class treeHandler(MP.Process):
	"""Performs MCTS starting from some input state for a given game

	Arguments:
	Pipe_treeHandler : This is the pipe used to communicate with this treeHandler instance
	Pipe_nnRequest: This is the pipe used to request NN inferences for unexplored states
	game: Class object that is the game being played
	config: Sets a variety of internal parameters

	Pipe_treeHandler packets:
	packet structure: [command,arg]
	Input Commands:
		0:Requests the NNRequest pipe used by this process
			arg = NA
		1:Loads the root state and nnName used for inference
			arg = [nnName, State]
		2:Adds some amount of noise to the root state
			arg = [noise]
		3:Begins to search the root state
			arg = [playerNum, num_simulations, explore]
		4:Terminates current search and returns latest state values
		5:Terminates and does not return
		6:Resets the internal tree of explored states
	Return Commands:
		0:Task failure
		1:Task success (includes arg if appropriate)
	Tree object:
		Dict object that accept game state as key, stores leafs as
		{'P':P,'Q':Q,'N':N}
	"""

	def __init__(self,Pipe_treeSearcher,Pipe_nnRequester, game,config):
		MP.Process.__init__(self)
		self.Game=game
		self.Pipe_treeSearcher=Pipe_treeSearcher
		self.Pipe_nnRequester = Pipe_nnRequester
		self.loadConfig(config)
		self.search_state = []
		self.search_playernum = 0
		self.nnName=''
		self.tree = {}
		self.stayAlive=True
	def loadConfig(self,config):
		"""Handles parsing of config dictionary, if keys are missing revert to a default value"""
	def run(self):
		starttime_total = time.time()
		"""Sets up all child tree_searcher processes, then listens to Pipe_treeHandler for commands"""
		print(self.name)

		while(self.stayAlive):
			if(self.Pipe_treeSearcher.poll()):
				command,arg = self.Pipe_treeSearcher.recv()
				response = {
				0:self.requestNNPipes,
				1:self.loadRootState,
				2:self.addRootNoise,
				3:self.startRootSearch,
				4:self.terminate_and_return,
				5:self.terminate,
				6:self.resetTree,
				}[command](arg)
				if (response==None):
					continue
				self.Pipe_treeSearcher.send(response)
		print(str(self.name)+" Exiting")

	def requestNNPipes(self,arg):
		"""Responds with the nnRequest pipe used by this process"""
		return [1,[self.Pipe_nnRequester]]

	def resetTree(self,arg):
		"""Returns internal state tree to blank values"""
		self.tree={}
		return None

	def terminate(self,arg):
		"""Stops the treeHandler process"""
		self.stayAlive = False
		return None

	def terminate_and_return(self,arg):
		"""Terminates process and returns latest root state esimate"""
		self.stayAlive = False
		return self.shareFinalState()

	def updateState(self,state,Q,a):
		"""modifies leaf keyed by state, replace a'th element of Q, increments a'th element of N """
		leaf=self.tree[state]
		leaf['Q'][a]=Q
		leaf['N'][a]+=1
		return None

	def loadRootState(self,arg):
		#print("Loading root state")
		#print(arg)
		self.nnName, self.search_state, self.search_playernum = arg
		return None

	def addRootNoise(self,arg):
		"""Adds noise to the root state prior estimates. If the root state has not been initialized yet, first request the NN inference then add the noise"""
		noise = arg
		if self.search_state.tostring() not in self.tree:
			state_nnPOV = self.Game.state_getNNPOV(self.search_state,self.search_playernum)
			self.Pipe_nnRequester.send([0,[self.nnName,state_nnPOV]])
			nn_command,nn_arg = self.Pipe_nnRequester.recv()
			P,rewards=nn_arg
			dim = P.shape
			P += noise
			self.tree[self.search_state.tostring()]={'P':P[:],'Q':NP.zeros(dim),'N':NP.zeros(dim)}
		else:
			self.tree[self.search_state.tostring()]['P'] += noise
		return None

	def startRootSearch(self,arg):
		"""Loop through the list of all expanders processes and send each a packet indicating: beginSearch,[state,(num_Sims/num_procs),explore]"""
		num_Sims,explore = arg
		for _ in range(0,num_Sims):
			self.searchTree(self.search_state, explore, self.search_playernum)
		return self.shareFinalState()

	def shareFinalState(self):
		"""Sends the latest estimate of the root state through the treeHandler pipe"""

		return [1,self.tree[self.search_state.tostring()]]

	def searchTree(self, state, explore, playernum):
		"""Recursive function that searches a tree made of game actions, utilizes Pipe_treeExpander to request and record states, uses Pipe_nnHandler to request state predictions"""

		if state.tostring() in self.tree:
			#print("state found")
			#sleep(2)
			leaf = self.tree[state.tostring()]
			Q = leaf['Q']
			P = leaf['P']
			N = leaf['N']
			#print(leaf)
			available_actions = self.Game.actions_getAvailable(state)

			best_a, best_u = -1 , -float("inf")
			for a in available_actions:
				u=Q[a]+(explore*P[a]*NP.sqrt(sum(N))/(N[a]+1))
				if u > best_u:
					best_u = u
					best_a = a
			#print("Choosing move:"+str(best_a))
			state_new,next_playernum = self.Game.state_getNext(state,best_a,playernum)
			#print([state_new,next_playernum])
			#endstate, 1 = prevstate, 0
			#reward [1,-1]
			gameover,rewards=self.Game.state_isTerminal(state_new,best_a)
			#print([gameover,rewards])
			if not(gameover):
				#print("Entering new state")
				rewards = self.searchTree(state_new, explore, next_playernum)
			#print("returnted from state w/ reward:"+str(rewards)+" Playernum:"+str(playernum))
			v = rewards[playernum]
			new_Q = (N[best_a]*Q[best_a] + v)/(N[best_a]+1)
			Q[best_a]=new_Q
			N[best_a]+=1
			#print("leaf is updated:"+str(leaf))
			return rewards

		else:
			#print("state NOT found")
			#print(state)
			state_nnPOV = self.Game.state_getNNPOV(state,playernum)
			self.Pipe_nnRequester.send([0,[self.nnName,state_nnPOV]])
			nn_command,nn_arg = self.Pipe_nnRequester.recv()
			P,rewards=nn_arg
			#print("Expected reward:"+str(rewards))
			dim = P.shape
			self.tree[state.tostring()]={'P':P[:],'Q':NP.zeros(dim),'N':NP.zeros(dim)}
			#print(self.tree[state.tostring()])
			#sleep(2)
			return rewards























