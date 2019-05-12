import multiprocessing as MP
import numpy as NP
import keras.backend as K
import keras.models as km
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation, Add, Concatenate, GaussianNoise, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, ReLU
from keras.optimizers import SGD,Adam
from keras.initializers import TruncatedNormal
from keras.backend.tensorflow_backend import set_session
import tensorflow as TF


class nnHandler(MP.Process):
	"""Handles inference and training of the NN instance

	Arguments:
	NNPipe_List: A list of Pipe objects through which communications will come

	Pipe_treeHandler packets:
	packet structure: [command,arg]
	Input Commands:
		0:Adds a state to the inference/prediction pipeline.
			arg = [TargetnnName, State]
		1:Fits the NN according to the training samples provided
			arg = [name,inState,outP,outV]
		2:Terminates this process
			arg = NA
		3:Exports weights of the chosen NN to a file with some name
			arg = [nnName,fileName]
		4:Copies NN weights between two NNs
			arg = [sourceNNname, targetNNname]
		5:Loads NN weights from a file
			arg = [nnName, fileName]
		6:Changes the learning rate of a NN
			arg = [nnName, newLearningRate]
	Return Commands:
		0:Task failure
		1:Task success (includes arg if appropriate)
"""
	def __init__(self, NNPipe_List, config):
		MP.Process.__init__(self)
		self.loadConfig(config)
		self.NNPipe_List = NNPipe_List
		self.nnList={}
		self.predictBatch={}
		self.responseBatch=[]
		self.predictBatchSize=0
		self.stayAlive=True
	def loadConfig(self,config):
		"""Handles parsing of config dictionary, if keys are missing revert to a default value"""
		try:
			self.val_learningRate = config['val_learningRate']
		except:
			self.val_learningRate=0.2
		try:
			self.val_minPredBatch = config['val_minPredBatch']
		except:
			self.val_minPredBatch = 1
		try:
			self.val_nnInputShape= config['val_nnInputShape']
		except:
			self.val_nnInputShape=[6,7,3]
		try:
			self.val_vOutputShape= config['val_vOutputShape']
		except:
			self.val_vOutputShape=2
		try:
			self.val_pOutputShape= config['val_pOutputShape']
		except:
			self.val_pOutputShape=7

	def run(self):
		"""Defines neural network and compiles it, then begins polling for messages within the nnPipe_List"""
		config = TF.ConfigProto()
		config.allow_soft_placement = False  # dynamically grow the memory used on the GPU
		sess = TF.Session(config=config)
		set_session(sess)  # set this TensorFlow session as the default session for Keras

		inputs = Input(shape = self.val_nnInputShape, name="MainInput")
		cv1 = Conv2D(81,kernel_size=(3,3), padding="same")(inputs)
		bn1 = BatchNormalization()(cv1)
		av1 = Activation('relu')(bn1)

		cv2 = Conv2D(81,kernel_size=(3,3), padding="same")(av1)
		bn2 = BatchNormalization()(cv2)
		add2 = Add()([av1,bn2])
		av2 = Activation('relu')(add2)

		cv3 = Conv2D(81,kernel_size=(3,3), padding="same")(av2)
		bn3 = BatchNormalization()(cv3)
		add3 = Add()([av2,bn3])
		av3 = Activation('relu')(add3)

		cv4 = Conv2D(81,kernel_size=(2,2), padding="same")(av3)
		bn4 = BatchNormalization()(cv4)
		add4 = Add()([av3,bn4])
		av4 = Activation('relu')(add4)

		cv5 = Conv2D(81,kernel_size=(2,2), padding="same")(av4)
		bn5 = BatchNormalization()(cv5)
		add5 = Add()([av4,bn5])
		av5 = Activation('relu')(add5)

		cv6 = Conv2D(81,kernel_size=(2,2), padding="same")(av5)
		bn6 = BatchNormalization()(cv6)
		add6 = Add()([av5,bn6])
		av6 = Activation('relu')(add6)

		cv7 = Conv2D(81,kernel_size=(2,2), padding="same")(av6)
		bn7 = BatchNormalization()(cv7)
		add7 = Add()([av6,bn7])
		av7 = Activation('relu')(add7)

		#P output branch
		cv8 = Conv2D(64,kernel_size=(3,3), padding="same")(av7)
		av8 = Activation('relu')(cv8)
		cva1 = Conv2D(50,kernel_size=(2,2), padding="same", activation = 'relu')(av8)
		fla1 = Flatten()(cva1)
		d1a = Dense(42,activation='relu')(fla1)
		d2a = Dense(self.val_pOutputShape,activation='softmax',name="POutput")(d1a)
		#V output branch
		cvb1 = Conv2D(50,kernel_size=(2,2), padding="same", activation = 'relu')(add7)
		flb1 = Flatten()(cvb1)
		d1b = Dense(42,activation='relu')(flb1)
		d2b = Dense(self.val_vOutputShape,activation='tanh',name="VOutput")(d1b)

		opti = Adam(lr=self.val_learningRate)
		nn1 = Model(inputs=[inputs],outputs=[d2a,d2b])
		nn1.compile(optimizer=opti,loss={'VOutput':'mean_squared_error','POutput':'categorical_crossentropy'},loss_weights={'VOutput':1,'POutput':1})
		
		self.nnList['bill']=nn1
		#print(nn1.summary())

		while(self.stayAlive):
			nn_requests = MP.connection.wait(self.NNPipe_List,timeout=None)
			for request_pipe in nn_requests:
				command,arg = request_pipe.recv()
				_ = {
					0:self.addTo_predictBatch,
					1:self.fit,
					2:self.terminate,
					3:self.exportWeights,
					4:self.transferWeights,
					5:self.loadNNWeights,
					6:self.changeNNLR
				}[command](arg+[request_pipe])
			if (self.predictBatchSize >= self.val_minPredBatch):
				#print(self.predictBatchSize)
				self.predict_Batch()
			for request_pipe,response in self.responseBatch:
				if response != None:
					request_pipe.send(response)
			self.responseBatch = []
		print(str(self.name)+" Exiting")

	def predict_Batch(self):
		"""performs predictions for all states loaded into the prediction batch list, then replies with the prediction through the corresponding pipe"""
		self.predictBatchSize=0
		for name in self.predictBatch:
			states = self.predictBatch[name]['states']
			reqPipe = self.predictBatch[name]['reqPipes']
			num = len(states)
			predictions=self.nnList[name].predict(NP.reshape(states,[num]+self.val_nnInputShape))
			for request_pipe,P,reward in zip(reqPipe,predictions[0],predictions[1]):
				self.responseBatch.append([request_pipe,[1,[P,reward]] ] )
		self.predictBatch = {}

	def addTo_predictBatch(self,arg):
		"""Adds a state to the list of states to be predicted in a batch"""
		name,state,request_pipe = arg
		self.predictBatchSize += 1

		if name not in self.predictBatch:
			self.predictBatch[name]={'states':[],'reqPipes':[]}
		self.predictBatch[name]['states'].append(state)
		self.predictBatch[name]['reqPipes'].append(request_pipe)		

	def fit(self,arg):
		"""Fits the NN according to the training data provided"""
		name,inState,outP,outV,numSamples,request_pipe = arg
		#view = [[state,p,v] for state,p,v in zip(inState,outP,outV)]
		#print(view)
		hist = self.nnList[name].fit(
								NP.reshape(inState,[numSamples]+self.val_nnInputShape),
									{'POutput':NP.reshape(outP,[numSamples]+[self.val_pOutputShape]),
									'VOutput':NP.reshape(outV,[numSamples]+[self.val_vOutputShape])},
								verbose = 0,
								batch_size=100,
								epochs = 100
							)
		self.responseBatch.append([ request_pipe,[1,[hist.history['loss']]] ] )

	def terminate(self,arg):
		"""Terminates this process"""
		self.stayAlive=False

	def exportWeights(self,arg):
		"""Exports weights from the NN to a file"""
		nnName,file,request_pipe = arg
		print("NN exporting:"+str(arg))
		self.nnList[nnName].save(file)
		self.responseBatch.append([request_pipe,None] )

	def transferWeights(self,arg):
		"""copies weights from one NN to another NN """
		model1,model2,request_pipe = arg
		print("Xfering weights from:"+str(model1)+" to "+str(model2))
		self.nnList[model2].set_weights(self.nnList[model1].get_weights())
		self.responseBatch.append([request_pipe,None] )

	def loadNNWeights(self,arg):
		"""Copies NN weights from a file, overwriting another NN"""
		nnName,file,request_pipe = arg
		print("NN Loading:"+str(arg))
		self.nnList[nnName]=km.load_model(file)
		self.responseBatch.append([request_pipe,None] )

	def changeNNLR(self,arg):
		"""Changes the learning rate of a neural net"""
		nnName,newlr,request_pipe = arg
		print("NN Learning Rate changed:"+str(arg))
		K.set_value(self.nnList[nnName].optimizer.lr, newlr)
		self.responseBatch.append([request_pipe,None] )
