import pygame
import numpy as np
import torch
from createModel import *
pygame.init()
w = 500
h = 450
window = pygame.display.set_mode((w, h))
pygame.display.set_caption("Number Recognition")

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)


class Pixel:
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.colour = white
		self.neighbours = []


	def draw(self, window):
		pygame.draw.rect(window,black,(self.x-1,self.y-1,self.x+self.width+1,self.y + self.height + 1))
		pygame.draw.rect(window, self.colour, (self.x, self.y, self.x + self.width, self.y + self.height))

	def isClicked(self, x_click, y_click):
		if self.x <= x_click <= self.x + self.width:
			if self.y <= y_click <= self.y + self.height:
				return True
	def getColour(self):
		return self.colour
	def clearPixel(self):
		self.colour = white

	def toggleColour(self):
		if self.colour == white:
			self.colour = black

		else:
			self.colour = white







class Grid:
	def __init__(self, rows, cols, width, height):
		self.rows = rows
		self.cols = cols
		self.width = width
		self.height = height
		self.pixels = []

	def clear(self):
		for r in self.pixels:
			for c in r:
				c.clearPixel()


	def drawPixels(self, display):
		for r in self.pixels:
			for c in r:
				c.draw(display)

	def getPixels(self):
		return self.pixels

	def getTensor(self):
		bin_matrix = []
		for row in self.pixels:
			bin_row = []
			for pixel in row:
				if pixel.colour == white:
					bin_row.append(0)
				else:
					bin_row.append(1)
			bin_matrix.append(bin_row)
		np_array = np.array(bin_matrix)
		return torch.Tensor(np_array)

	def predictNum(self):
		tensor = self.getTensor().view(1,1,28,28)
		model = ConvNet()
		model.load_state_dict(torch.load('NumberGuesser.pt'))
		model.eval()
		with torch.no_grad():
			prediction = model(tensor.view(1,1,28,28)).argmax()
		print("I predict that this is a: {}".format(prediction.item()))

	def createPixels(self):
		px_width = self.width // self.cols
		px_height = self.height // self.rows
		for r in range(self.rows):
			self.pixels.append([])
			for c in range(self.cols):
				self.pixels[r].append(Pixel(px_width * c, px_height * r, px_width, px_height))


grid = Grid(28, 28, w, h)
grid.createPixels()


def main():
	run = True
	while run:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_c:
					grid.clear()
				if event.key == pygame.K_RETURN:
					grid.getTensor()
					grid.predictNum()
			if pygame.mouse.get_pressed()[0]:
				for c in grid.pixels:
					for r in c:
						if r.isClicked(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]):
							r.toggleColour()

		window.fill(white)
		grid.drawPixels(window)
		pygame.display.update()


if __name__ == '__main__':
	main()
pygame.quit()
