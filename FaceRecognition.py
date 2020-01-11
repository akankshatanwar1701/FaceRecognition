import cv2
import numpy as np
import os

class face_recog():
	def distance(self,v1,v2):
		return np.sqrt(sum((v1-v2)**2)) 

	def knn(self,train,test,k=5):
		self.dist=[]
	    
	    
		for i in range(train.shape[0]):
			self.ix= train[i, :-1] 
			self.iy= train[i,-1] 
	        
	        
			self.d= self.distance(test,self.ix)
			self.dist.append([self.d,self.iy])
		self.dk= sorted(self.dist, key=lambda x:x[0])[:k] 
		self.labels= np.array(self.dk)[:,-1]
		self.output=np.unique(self.labels,return_counts=True)
		self.index= np.argmax(self.output[1])
		return self.output[0][self.index]


	def data_prep_test(self):
		self.cap=cv2.VideoCapture(0)
	#Face detection
		face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

		self.skip=0
		self.dataset_path= './data/'
		self.face_data= [] #training, x

		self.labels= [] #y

		self.class_id=0 #labels
		self.names= {} #mapping b/w id and name

	#Data prep

		for fx in os.listdir(self.dataset_path): #listdir what all files? in data folder
			if fx.endswith('.npy'): #(x,30000)
				self.names[self.class_id]=fx[:-4]#mapping b/w class id and name    #slice .npy
				print("loaded"+fx)
				self.data_item=np.load(self.dataset_path+fx)
				self.face_data.append(self.data_item) # first,second so on faces for a given file

			#Create Labels for the Class
				self.target= self.class_id*np.ones((self.data_item.shape[0]))
				self.class_id+=1
				self.labels.append(self.target)

		self.face_dataset= np.concatenate(self.face_data,axis=0)
		self.face_labels=np.concatenate(self.labels,axis=0).reshape(-1,1)
		print(self.face_dataset.shape)
		print(self.face_labels.shape)

		self.trainset= np.concatenate((self.face_dataset,self.face_labels),axis=1)
		print(self.trainset.shape) # 30000 features +1 labels=30001



	#Testing


		while True:
			self.ret,self.frame=self.cap.read()

			if self.ret == False:
				continue


			self.faces=face_cascade.detectMultiScale(self.frame,1.3,5)


			for face in self.faces:
				x,y,w,h= face

			#get region of interest
				self.offset=10
				self.face_section = self.frame[y-self.offset:y+h+self.offset,x-self.offset:x+w+self.offset]
				self.face_section = cv2.resize(self.face_section,(100,100))

				self.out= self.knn(self.trainset,self.face_section.flatten()) #or reshape

			#Display on the screen the name and rec
				self.pred_name=self.names[int(self.out)]
				cv2.putText(self.frame,self.pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA) #adds text to img,x,y.. is coord,font,size?,color,thick
				cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,255),2)

			cv2.imshow("faces",self.frame)

			key=cv2.waitKey(1) &0xFF
			if key==ord('q'):
				break

		self.cap.release()
		self.cap.destroyAllWindows()

a=face_recog()
a.data_prep_test()


