import treeHandler as TH
import trainUtil as TU 
import numpy as NP 
import multiprocessing as MP

class gamePlayer():
	"""Abstracts playing of games versus self and generating of test data sets

	Arguments:
	game: Class representing rules of chosen game
	config: Sets various internal parameters for how game is played

	
	"""
	def __init__(self, game, config):
		self.game = game
		self.treeHandler_Pipe, treeHandler_Pipe_s = MP.Pipe()
		self.nnHandler_Pipe, nnHandler_Pipe_s = MP.Pipe()
		self.treeHandler = TH.treeHandler(treeHandler_Pipe_s,nnHandler_Pipe_s,self.game,{})
		self.treeHandler.start()
		self.loadConfig(config)
		self.gameMemory = TU.gameMemory(self.game)
		
		self.gameRecord=[]
	def loadConfig(self,config):
		"""Handles parsing of config dictionary, if keys are missing revert to a default value"""
		self.nnName = config['val_nnname']
		self.numSims = config['val_numsims']
		self.exploreInit = config['val_exploreinit']
		self.exploreStart_Game = self.exploreInit
		self.exploreDecay_Move = config['val_exploredecay_move']
		self.exploreDecay_Game = config['val_exploredecay_game']
		self.priorNoise = config['val_priornoise']

	def startGame(self):
		"""Initializes required game states then triggers first MCTS in treeHandler """
		self.gameOver=False
		self.restart = False
		self.state=NP.zeros((6,7))
		self.gameRecord=[]
		self.gameMemory.reset()
		self.playerNum=0
		self.explore = self.exploreStart_Game
		self.exploreStart_Game *= self.exploreDecay_Game

		loadRootState_packet = [self.nnName, self.state, self.playerNum]
		addRootNoise_packet = [self.priorNoise]
		startRootSearch_packet = [self.numSims,self.explore]

		self.treeHandler_Pipe.send([1,loadRootState_packet])
		self.treeHandler_Pipe.send([2,addRootNoise_packet])
		self.treeHandler_Pipe.send([3,startRootSearch_packet])

	def updateGame(self):
		"""Checks to see if MCTS is finished, if so advance game state by choosing moves and updating internal memory"""
		if(self.treeHandler_Pipe.poll()):
			command,updated_leaf = self.treeHandler_Pipe.recv()
			if(self.restart):
				self.startGame()
				return False
			#print('P'+str(updated_leaf['P']))
			#print('N'+str(updated_leaf['N']))
			#print('Q'+str(updated_leaf['Q']))
			#print("--------------------")
			updated_P = [n/sum(updated_leaf['N']) for n in updated_leaf['N']]
			updated_P = NP.asarray(updated_P)
			
			#Either get random move or choose best move from updated leaf
			action = NP.random.choice(len(updated_P),p=updated_P)
			#print(action)
			#Advance state with new action
			self.state, self.playerNum = self.game.state_getNext(self.state,action,self.playerNum)
			#Add state the memory
			self.gameMemory.addState(self.state,updated_P,self.playerNum)
			#Check if game has ended
			self.gameOver, rewards = self.game.state_isTerminal(self.state,action)
			if(self.gameOver):
				#Game is over, update game memory with final outcome
				#print("Gameover")
				self.gameMemory.updateV(rewards)
				self.gameRecord = self.gameMemory.getMemory_AmplifiedNNPOV()
				return True
			self.explore *= (self.exploreDecay_Move)
			
			loadRootState_packet = [self.nnName, self.state, self.playerNum]
			addRootNoise_packet = [self.priorNoise]
			startRootSearch_packet = [self.numSims,self.explore]

			self.treeHandler_Pipe.send([1,loadRootState_packet])
			self.treeHandler_Pipe.send([2,addRootNoise_packet])
			self.treeHandler_Pipe.send([3,startRootSearch_packet])
		return False
				
	def resetSimulation(self):
		"""Resets the MCTS memory to blank"""
		self.exploreStart_Game = self.exploreInit
		self.treeHandler_Pipe.send([6,0])

	def getGameRecord(self):
		"""Returns the record of moves made by this game player"""
		return self.gameRecord

	def restartGame(self):
		"""Ends current game then starts a new one"""
		self.restart = True

	def getNNHandlerPipe(self):
		"""Returns the NNpipe used by the MCTS process (treeHandler Process)"""
		return [self.nnHandler_Pipe]

	def getTreeHandlerPipe(self):
		"""Returns pipe used to communicate with the MCTS process (treeHandler process)"""
		return self.treeHandler_Pipe

	def getState(self):
		"""Returns the current game state"""
		return self.state

	def setExplore(self,explore):
		"""Changes the value of the explore variable used by the MCTS process (treeHandler process)"""
		self.exploreStart = explore

					