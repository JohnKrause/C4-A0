import numpy as NP

class Game():
	def __init__(self,player1_icon=1.,player2_icon=-1.,neutral_icon=0.):
		self.player1_icon = player1_icon
		self.player2_icon = player2_icon
		self.neutral_icon = neutral_icon
	def state_isTerminal(self,state,action):
		winner = checkForWin(state,action,self.player1_icon,self.player2_icon)
		if(winner==self.player1_icon):
			reward=[1,-1]
			gameover = True
		elif(winner==self.player2_icon):
			reward=[-1,1]
			gameover = True
		else:
			if(self.actions_getAvailable(state)==False):
				reward=[-0.5,-0.5]
				gameover=True
			else:
				reward=[0,0]
				gameover=False
		return gameover,reward
	def actions_getAvailable(self,state):
		availmoves = []
		for index,val in enumerate(state[0]):
			if val==self.neutral_icon:
				availmoves.append(index)
		if (len(availmoves)):
			return availmoves
		else:
			return False
			
	def state_getNNPOV(self,state,playernum):
		num_rows,num_cols = state.shape
		p1board = NP.copy(state)
		p1board[p1board<0]=0.0 
		p2board = NP.copy(state) 
		p2board[p2board>0]=0.0
		p2board *= -1.0

		amboard = NP.copy(state)
		amboard[amboard>0]=-1
		for colindex,col in enumerate(state.T): #through each column of board
			for rowindex,val in enumerate(col): #Run through each valu in column, starting from bottom
				if val:
					if(rowindex):
						amboard[rowindex-1][colindex]=1
						break
				if (rowindex==(num_rows-1)):
					amboard[num_rows-1][colindex]=1
		amboard[amboard<0]=0
		if (playernum==0):
			return [p1board,p2board,amboard]
		else:
			return [p2board,p1board,amboard]
	def reward_getNNPOV(self,reward,playernum):
		if(playernum==0): #By default all states of this game are from POV of player 1 (playernum 0)
			return reward
		else: #Game needs to be modified to match POV of player 2 (playernum 1), multiply all values by -1
			return [reward[1],reward[0]]
	def convertTo_NNPOV(self,memory):
		NNPOV = {'State':[],'P':[],'V':[], 'Player':[]}
		for state,P,V,playernum in zip(memory['State'],memory['P'],memory['V'],memory['Player']):
			NNPOV['State'] += self.state_getNNPOV(state,playernum)
			NNPOV['P'] += [P]
			NNPOV['V'] += self.reward_getNNPOV(V,playernum)
			NNPOV['Player'] += [playernum]
		return NNPOV


	def convertTo_AmplifiedNNPOV(self,memory):
		NNPOV = {'State':[],'P':[],'V':[], 'Player':[]}
		for state,P,V,playernum in zip(memory['State'],memory['P'],memory['V'],memory['Player']):

			state_NNPOV = self.state_getNNPOV(state,playernum)
			

			NNPOV['State'] += [state_NNPOV]
			NNPOV['P'] += [P]
			NNPOV['V'] += [self.reward_getNNPOV(V,playernum)]
			NNPOV['Player'] += [playernum]

			NNPOV['State'] += [[NP.fliplr(state_NNPOV[0]),NP.fliplr(state_NNPOV[1]),NP.fliplr(state_NNPOV[2])]]
			NNPOV['V'] += [self.reward_getNNPOV(V,playernum)]
			NNPOV['P'] += [NP.flip(P)]
			NNPOV['Player'] += [self.reward_getNNPOV(V,playernum)]
		return NNPOV


	def state_getNext(self,state,action,playernum):
		state_new=NP.copy(state)
		col = [row[action] for row in state_new]
		for index,icon in reversed(list(enumerate(col))):
			if (icon == self.neutral_icon):
				if(playernum==0):
					state_new[index][action]=self.player1_icon
				else:
					state_new[index][action]=self.player2_icon
				break

		playernum += 1
		if(playernum>1):
			playernum=0
		return state_new, playernum

def checkForWin(board,action,player1_icon,player2_icon):
	#search directions in order x,y
	numrow, numcol = board.shape
	#Check to see if the column is won:
	row_played = 0

	col = board[:,[action]]
	col = col.flatten()
	last_chip = 0
	count=1
	for place in col:
		if(place):
			if(place==last_chip):
				count+=1
			else:
				last_chip = place
				count=1
		else:
			row_played +=1
			count = 1
			last_chip = 0
		if(count>3):
			return last_chip

	row = board[[row_played],:][0]
	last_chip = 0
	count=1
	for place in row:
		if(place):
			if(place==last_chip):
				count+=1
			else:
				last_chip = place
				count=1
		else:
			count = 1
			last_chip = 0
		if(count>3):
			return last_chip


	diag1 = NP.diagonal(board,(action-row_played))
	last_chip = 0
	count=1
	for place in diag1:
		if(place):
			if(place==last_chip):
				count+=1
			else:
				last_chip = place
				count=1
		else:
			count = 1
			last_chip = 0
		if(count>3):
			return last_chip
	
	diag2 = NP.diagonal(NP.flip(board,1),((numcol-action-1)-row_played))
	last_chip = 0
	count=1
	for place in diag2:
		if(place):
			if(place==last_chip):
				count+=1
			else:
				last_chip = place
				count=1
		else:
			count = 1
			last_chip = 0
		if(count>3):
			return last_chip
	return False
