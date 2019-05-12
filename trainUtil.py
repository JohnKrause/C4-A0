import multiprocessing as MP
import numpy as NP
from itertools import cycle

class gameMemory():
	"""Stores memory of each turn made in a training game: State, P, V, Playernum

	addState accepts State,P,Playernum as single elements
	getMemory_NNPOV returns State,P,V lists after tranforming to NN expected POV
	convertToNNPOV takes internal memory and applies game.state_getNNPOV function to each entry, returns results
	getMemory_AmplifiedNNPOV transforms memory into additional states according to game rules then returns new states
	getMemory returns contents of memory without modification
	"""
	def __init__(self,game):
		self.memory={'State':[],'P':[],'V':[], 'Player':[]}
		self.memory_NNPOV={'State':[],'P':[],'V':[], 'Player':[]}
		self.game = game
	def reset(self):
		"""resets internal memory to blank values"""
		self.memory={'State':[],'P':[],'V':[], 'Player':[]}
		self.memory_NNPOV={'State':[],'P':[],'V':[], 'Player':[]}

	def addState(self,State,P,Player):
		"""Appends state information to internal memory"""
		#print(State)
		self.memory['State'].append(State[:])
		self.memory['P'].append(P[:])
		self.memory['V'].append(0)
		self.memory['Player'].append(Player)

	def updateV(self,rewards):
		"""Modifies the internal V values in memory according to game outcome"""
		self.memory['V']=[rewards for _ in self.memory['V']]

	def getMemory_NNPOV(self):
		"""Converts internal memory to NNPOV then returns memory"""
		self.memory_NNPOV = self.game.convertTo_NNPOV(self.memory)
		return self.memory_NNPOV['State'], self.memory_NNPOV['P'], self.memory_NNPOV['V']

	def getMemory_AmplifiedNNPOV(self):
		"""Generates new board state according to game symmetry rules, then converst to NNPOV, then returns memory"""
		self.memory_NNPOV = self.game.convertTo_AmplifiedNNPOV(self.memory)
		return self.memory_NNPOV['State'], self.memory_NNPOV['P'], self.memory_NNPOV['V']

	def getMemory(self):
		"""Returns unmodified memory"""
		return self.memory['State'], self.memory['P'], self.memory['V']

class trainingSamples():
	"""Used for storing training memory samples, keeps track of when enough samples are available

	Reset: Empties memory
	addEntry: Appends entry to internal memeory
	isReady: Returns True if total number of entries is larger than batch_size, else returns False
	"""
	def __init__(self,batch_size):
		self.batch_size = batch_size
		self.states=[]
		self.VOutputs=[]
		self.POutputs=[]
	def reset(self):
		"""Resets internal memory"""
		self.states=[]
		self.VOutputs=[]
		self.POutputs=[]
	def addEntry(self,state,P,V):
		"""Appends entry to internal memory"""
		self.states+=state
		self.VOutputs+=V
		self.POutputs+=P
		excess = len(self.states) - self.batch_size
		if(excess>0):
			for _ in range(0,excess):
				del(self.states[0])
				del(self.VOutputs[0])
				del(self.POutputs[0])
	def isReady(self):
		"""Returns True if number of stored examples is large enough"""
		if(len(self.states)>=self.batch_size):
			return True
		return False
	def getBatch(self):
		"""Returns the batch memory"""
		return [NP.asarray(self.states), NP.asarray(self.POutputs), NP.asarray(self.VOutputs)]






