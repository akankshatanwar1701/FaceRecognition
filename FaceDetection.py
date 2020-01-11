import cv2
import numpy as np

class face_detec():
	def __init__(self):
		self.skip=0
		self.faces= []
		self.offset=10
		self.dataset_path= './data/'
		self.filename="Default"
		self.face_data= []

	def run(self):
		self.cap=cv2.VideoCapture(0)

		face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

		self.file_name= input ("Enter the name of the person: ")
		while True:
			self.ret,self.frame= self.cap.read()

			if self.ret == False:
				continue 

			self.gray_frame=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
	

			self.faces=face_cascade.detectMultiScale(self.frame,1.3,5) 
	
			self.faces=sorted(self.faces,key=lambda f:f[2]*f[3],reverse=True) 														   
			
			for face in self.faces:
				x,y,w,h = face
				cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,255),2)

		
				self.offset = 10
				self.face_section = self.frame[y-self.offset:y+h+self.offset,x-self.offset:x+w+self.offset]
				self.face_section = cv2.resize(self.face_section,(100,100)) 
				cv2.imshow("Face Section",self.face_section)

				self.skip += 1
				if self.skip%10==0:
					self.face_data.append(self.face_section)
					print(len(self.face_data))
		
			self.show()

			key_pressed = cv2.waitKey(1) & 0xFF
			if key_pressed == ord('q'):
				break
						

	def	show(self):
		cv2.imshow("Frame",self.frame)

	def convert(self):
		self.face_data= np.asarray(self.face_data)
		self.face_data=self.face_data.reshape((self.face_data.shape[0],-1)) 


	def save(self):
		np.save(self.dataset_path+self.file_name+'.npy',self.face_data) 
		print("Data successfully saved at"+self.dataset_path+self.file_name+'.npy')
		self.cap.release()
		cv2.destroyAllWindows()



persons=[]
char='y'
while(char=='y'or char=='YES' or char=='yes'):
	persons.append(face_detec())
	persons[-1].run()
	persons[-1].convert()
	persons[-1].save()
	
	char= input ("Do you want to continue?: ")




