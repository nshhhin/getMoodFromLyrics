
class ball:
	
	def __init__(self,name):
		self.name = name
		self.x = 0

	def move(self):
		self.x = self.x + 1

	def show(self):
		print(self.name + ":" + str(self.x))


ball1 = ball("A")
ball1.move()
ball1.show()
