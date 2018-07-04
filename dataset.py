from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
import argparse
import time, os

dataset_path = 'd:dataset_test.pz'

'''
#Nice locations
[720.488037109375, -629.4966430664062, 36.21859359741211] 
[398.19970703125, -580.4951171875, 28.84107208251953] 
[2432.731201171875, -176.03179931640625, 87.86923217773438]
[-661.9974975585938, -1932.245849609375, 27.170656204223633]
'''

def reset():
	#Resets position of car to a specific location
	# Same conditions as below | 
	dataset = Dataset(rate=60, frame=[1920,1080], vehicles=True, location=True)
	scenario = Scenario(weather='EXTRASUNNY',vehicle='packer',time=[9,0],drivingMode=[786603,20.0])
	client.sendMessage(Config(scenario=scenario,dataset=dataset))

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
	parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
	parser.add_argument('-d', '--dataset_path', default=dataset_path, help='Place to store the dataset')
	args = parser.parse_args()

    #remove old dataset if exists
	try:
		os.remove(dataset_path)
	except OSError:
		pass

	# Creates a new connection to DeepGTAV using the specified ip and port 
	client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9) 
	# Dataset options
	dataset = Dataset(rate=60, frame=[1920,1080], vehicles=True, location=True)
	# Automatic driving scenario
	scenario = Scenario(weather='EXTRASUNNY',vehicle='packer',time=[9,0],drivingMode=[786603,20.0])#, location=[-661.9974975585938, -1932.245849609375, 27.170656204223633]) 
	client.sendMessage(Start(scenario=scenario,dataset=dataset)) # Start request
	
	count = 0
	old_location = [0, 0, 0]
	
	# Main loop stops when dataset.size < N GB
	while os.path.getsize(dataset_path) < (100e+9): 
		try:
			# Message recieved as a Python dictionary
			message = client.recvMessage()
			# Checks if car is stuck, resets position if it is
			if (count % 250)==0:
				new_location = message['location']
				# Float position converted to ints so it doesn't have to be in the exact same position to be reset
				if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]) and int(new_location[2]) == int(old_location[2]):
					reset()
				old_location = message['location']
				print('At location: ' + str(old_location))
			count += 1

		except KeyboardInterrupt:
			i = input('Paused. Press p to continue and q to exit... ')
			if i == 'p':
				continue
			elif i == 'q':
				break
			
	# DeepGTAV stop message
	client.sendMessage(Stop())
	client.close() 
