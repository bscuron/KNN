import pygame
from PIL import Image, ImageOps, ImageEnhance
import sys
import numpy as np
from time import sleep, time_ns
import pandas as pd

# window settings
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode([WIDTH, HEIGHT])

# canvas settings
coords = [] # (x, y) coordinates of user drawing
PEN_WIDTH = 5
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

# image filter settings
CONTRAST_FACTOR = 10 # the factor to increase the 28x28 representation of the screen's contrast

def main():
    knn_model = getModel()
    pygame.init()
    while True:
        screen.fill(COLOR_BLACK)
        handleEvents()
        drawCoords()
        pixels = getPixels()
        pygame.display.flip()

# train the KNN classification model using the MNIST dataset
def getModel():
    print('Loading dataset...')
    X_train, X_test, y_train, y_test = loadData()
    print('Loaded dataset')

# load MNIST dataset into pandas dataframes (train, test). Use cached pkl format if available.
def loadData():
    try:
        trainingData = pd.read_pickle('./train_cached.pkl')
        testingData = pd.read_pickle('./test_cached.pkl')
    except OSError:
        trainingData = pd.read_csv('./train.csv')
        testingData = pd.read_csv('./test.csv')
        trainingData.to_pickle('./train_cached.pkl')
        testingData.to_pickle('./test_cached.pkl')
    X_train = trainingData.drop('label', axis='columns').values
    y_train = trainingData['label'].values
    X_test = testingData.drop('label', axis='columns').values
    y_test = testingData['label'].values
    return X_train, X_test, y_train, y_test
        

# draw a line between adjacent points/coordinates in the `coords` list
def drawCoords():
    if len(coords) >= 2:
        pygame.draw.lines(surface=screen, color=COLOR_WHITE, closed=False, points=coords, width=PEN_WIDTH)

# handle the different pygame events (mouse events, keyboard events, quit event)
def handleEvents():
    for event in pygame.event.get():
        # if left mouse button is clicked try to append mouse coordinates to the coords list
        if pygame.mouse.get_pressed()[0]:
            try:
                coords.append(event.pos)
            except AttributeError:
                pass
        # clear screen on <SPACE> kepress
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                coords.clear()
        # if window is closed by user, quit program
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

# returns a 28x28 representation of the screen pixels. The pixels are in the form of a 1d vector values [0-255]
def getPixels():
    imgStr = pygame.image.tostring(screen, 'RGBA', False)
    img = Image.frombytes('RGBA', (WIDTH, HEIGHT), imgStr)
    img = ImageOps.grayscale(img)
    img.thumbnail((28, 28), Image.ANTIALIAS)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    # img.save(f'./images/screen_{time_ns()}.jpg')
    pixels = np.array(list(img.getdata()))
    return pixels

if __name__ == '__main__':
    main()
