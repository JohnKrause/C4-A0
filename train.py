import nnHandler as NH 
import trainUtil as TU 
import c4game as game
import gamePlayer as GP
import treeHandler as TH 

import multiprocessing as MP 
import numpy as NP 
import time
import random


def train(config):
	
	#Load Config
	#---------------------------------------
	num_generations = config['num_generations']
	num_games_generation = config['num_games_generation']
	num_games_trial = config['num_games_trial']
	exploreInit = config['exploreInit']
	exploreDecay_move = config['exploreDecay_Move']
	exploreDecay_game = config['exploreDecay_Game']
	num_gamePlayers = config['num_gamePlayers']
	num_sims = config['num_sim']
	num_trainingSamples = config['num_trainingSamples']
	val_priorNoise = config['val_priorNoise']

	theGame = game.Game()
	nnPipe_List=[]
	gamePlayers_List=[]
	winnerFileName = 'LatestV3.h5'
	importFileName = 'BackupV5.h5'
	backupFileName = 'BackupV6.h5'

	#Create all GamePlayer instances
	#---------------------------------------
	gamePlayerConfig = {
			'val_nnname':'bill',
			'val_numsims':num_sims,
			'val_exploreinit':exploreInit,
			'val_exploredecay_move':exploreDecay_move,
			'val_exploredecay_game':exploreDecay_game,
			'val_priornoise':val_priorNoise
			}
	for index in range(num_gamePlayers):
		gamePlayer = GP.gamePlayer(theGame,gamePlayerConfig)
		nnPipe_List += gamePlayer.getNNHandlerPipe()
		gamePlayers_List += [gamePlayer]
	#Create an extra gameplayer instance for later usage
	testPlayer = GP.gamePlayer(theGame,gamePlayerConfig)
	nnPipe_List += testPlayer.getNNHandlerPipe()
	treeHandler_Pipe_m = gamePlayers_List[0].getTreeHandlerPipe()

	#Create NN handler instance
	#---------------------------------------
	NNConfig = {
			'val_learningRate':0.02,
			'val_nninputShape':[6,7,3],
			'val_vOutputShape':2,
			'val_pOutputShape':7,
			'val_minPredBatch':1
			}
	#Create one extra NN handler pipe for our usage later
	nnPipe_m,nnPipe_s = MP.Pipe()
	nnPipe_List += [nnPipe_s]
	nnHandler = NH.nnHandler(nnPipe_List,NNConfig)
	nnHandler.start()

	#Setup all generational metrics
	#---------------------------------------
	lossArray = []
	fitArray = []
	lastGames = []
	fitAttempt = 0
	totalGamesFinished = 0
	exploreStart = 1
	exploreDecay = 0.95
	startTime = time.time()

	nnPipe_m.send([3,['bill',backupFileName]])
	nnPipe_m.send([5,['adam',backupFileName]])
	trainingMemory = TU.trainingSamples(num_trainingSamples)

	time.sleep(3)
	for generationID in range(0,num_generations):
		print("Beginning Generation: "+str(generationID))
		for Player in gamePlayers_List:
			Player.resetSimulation()
			Player.startGame()
		gamesFinished = 0
		trialCounter = 0
		while(gamesFinished<num_games_generation):

			#Update Each instance of the running games
			for Player in gamePlayers_List:
				gameOver = Player.updateGame()
				if(gameOver):
					gamesFinished += 1
					totalGamesFinished += 1
					trialCounter += 1
					states, ps, vs = Player.getGameRecord()
					trainingMemory.addEntry(states, ps, vs)
					lastGames = Player.getState()
					Player.startGame()

			#Check to see if enough training data is available to refit the NN
			if(trainingMemory.isReady()):
				#Send the training data to NN handler for training
				nnPipe_m.send([1,['adam']+trainingMemory.getBatch()+[num_trainingSamples]])
				trainingMemory.reset()
				command,loss = nnPipe_m.recv()
				#Display our loss metrics
				lossArray += loss
				fitAttempt += 1
				print(str(max(loss[0]))+" "+str(min(loss[0]))+" "+str(fitAttempt)+" "+str(totalGamesFinished))
				print("Elapsed time:"+str((time.time()-startTime)/(60*60)))
				print("Play Rate:"+str(totalGamesFinished / ((time.time()-startTime)/(60*60))) )
				print(lastGames)
				nnPipe_m.send([3,['adam',backupFileName]])
				#Tell all game players to start fresh games
				for Player in gamePlayers_List:
					Player.restartGame()
			if(trialCounter > num_games_trial):
				trialCounter = 0
				#if((adamWins>billWins): #Overwrite bestnn with newnn
				#	print("Overwriting Bill with Adam")
				nnPipe_m.send([4,['adam','bill']])
				#	nnPipe_m.send([3,['adam',winnerFileName]])
				#else:
				#	print("Keeping Bill")
					#nnPipe_m.send([4,['bill','adam']])
				#	nnPipe_m.send([3,['bill',winnerFileName]])
				print("Win% , Turns/Game="+str(play_Vs_Random(treeHandler_Pipe_m,'bill')))

def play_Vs_Random(treeHandler_Pipe_m,nnName):
	thegame = game.Game()
	Games = 0
	RandWins=0
	AIwins = 0
	targetnnName=nnName
	num_sims = 50
	explore=0.5
	numTurns=0
	noise=0.14


	for gamenum in range(0,20):
		Games+=1
		playernum=0
		state = NP.zeros((6,7))
		gameOver = False

		while not(gameOver):
			numTurns+=1
			#Initiate search of game state
			
			loadRootState_packet = [targetnnName, state, playernum]
			addRootNoise_packet = [noise]
			startRootSearch_packet = [num_sims,explore]
			treeHandler_Pipe_m.send([1,loadRootState_packet])
			treeHandler_Pipe_m.send([2,addRootNoise_packet])
			treeHandler_Pipe_m.send([3,startRootSearch_packet])

			#Get updated state information from tree handler
			command,updated_leaf = treeHandler_Pipe_m.recv()
			updated_P = [n/sum(updated_leaf['N']) for n in updated_leaf['N']]
			updated_P = NP.asarray(updated_P)
			#Either get random move or choose best move from updated leaf
			action = NP.argmax(updated_P)
			#Advance state with new action
			state, playernum = thegame.state_getNext(state,action,playernum)
			#Add state the memory
			gameOver, rewards = thegame.state_isTerminal(state,action)
			if(gameOver):
				#AI won
				AIwins+=1
				break
			#Make random move...
			numTurns+=1
			action = random.sample(thegame.actions_getAvailable(state),1)[0]
			state, playernum = thegame.state_getNext(state,action,playernum)
			gameOver, rewards = thegame.state_isTerminal(state,action)
			if(gameOver):
				#Random won
				RandWins+=1
				break
	return [AIwins/Games, numTurns/Games]

#def play_Vs_nn(treeHandler_Pipe_m,nn1_name,nn2_name):
	

#def play_Vs_Human(config):
	


