import pygame
import os
from PIL import Image, ImageOps, ImageEnhance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import shuffle
import sys
import numpy as np
from time import time_ns
import pandas as pd
from skimage import filters
from skimage.measure import regionprops

# program settings
DELAY = 100 # ms

# window settings
WIDTH, HEIGHT = 504, 504
screen = pygame.display.set_mode([WIDTH, HEIGHT])

# canvas settings
coords = [] # (x, y) coordinates of user drawing
draw = False
PEN_WIDTH = min(WIDTH, HEIGHT) // 20
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

# image filter settings
CONTRAST_FACTOR = 1 # the factor to increase the 28x28 representation of the screen's contrast
PIXEL_PADDING = min(WIDTH, HEIGHT) // 8 # when centering the drawing, this is how many pixels will be added to the top, bot, left, and right for padding
IMAGE_SIZE = 28

# model settings
K_RANGE = range(1, 6)
CROSS_VALIDATION_FOLDS = 5

def main():
    knn = getModel()
    pygame.display.set_caption('Digit Classification')
    pygame.init()
    label, screenStr, prevLabel, prevCoords, prevScreenStr = None, None, None, None, None
    while True:
        screen.fill(COLOR_BLACK)
        handleEvents()
        drawCoords()
        screenStr = pygame.image.tostring(screen, 'RGBA', False)

        # only make new predictions if the screen has updated (for better performance)
        if screenStr != prevScreenStr:
            pixels = getPixels()
            probablities = knn.predict_proba(pixels)[0]
            label = probablities.argmax(axis=0)
            if label != prevLabel:
                pygame.display.set_caption(str(label))
                say(label)
            prevLabel = label
            prevScreenStr = screenStr
        pygame.display.flip()

        # yield the process if not currently drawing
        if not draw:
            pygame.time.wait(DELAY)

# say the given word as a new process in the background
def say(word):
    if SAY:
        os.system(f'say "{word}" &')

# train the KNN classification model using the MNIST dataset
def getModel():
    # load dataset split into training/validation and testing
    print('Loading dataset...')
    X_train, X_test, y_train, y_test = loadData()
    print('Loaded dataset')

    # normalize features
    print('Normalizing features...')
    X_train = X_train / np.max(X_train)
    X_test = X_test / np.max(X_test)
    print('Normalized features')

    # use cross-validation to select the optimal hyperparameter 'k'
    print('Finding optimal hyperparameter...')
    # k = getK(X_train, y_train)
    k = 4
    print('Found optimal hyperparameter')

    # fit the data to the model with the optimal hyperparameter 'k'
    print('Fitting data...')
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    print('Fit data')

    # evaluate the model on the testing set
    if STATS:
        print('Computing statistics...')
        y_test_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average=None)
        recall = recall_score(y_test, y_test_pred, average=None)
        precision = precision_score(y_test, y_test_pred, average=None)
        print(f'Accuracy: {acc}')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')
        print(f'F1 score: {f1}')

    return knn

# perform cross-validation to find the optimal hyperparameter 'k'
def getK(features, labels):
    param_grid = dict(n_neighbors=K_RANGE)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=CROSS_VALIDATION_FOLDS, scoring='accuracy')
    grid.fit(features, labels)
    print(grid.best_score_)
    print(grid.best_params_)
    return grid.best_params_['n_neighbors']

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
    trainingData = shuffle(trainingData)
    testingData = shuffle(testingData)
    X_train = trainingData.drop('label', axis='columns').values
    y_train = trainingData['label'].values
    X_test = testingData.drop('label', axis='columns').values
    y_test = testingData['label'].values
    return X_train, X_test, y_train, y_test
        

# draw a line between adjacent points/coordinates in the `coords` list
def drawCoords():
    for points in coords:
        if len(points) > 1:
            pygame.draw.lines(screen, COLOR_WHITE, False, points, PEN_WIDTH)
            for point in points:
                pygame.draw.circle(screen, COLOR_WHITE, point, PEN_WIDTH / 2)

# handle the different pygame events (mouse events, keyboard events, quit event)
def handleEvents():
    global draw
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw = True
            if draw:
                coords.append([event.pos])
        elif event.type == pygame.MOUSEBUTTONUP:
            draw = False
        elif event.type == pygame.MOUSEMOTION and draw:
            coords[-1].append(event.pos)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            coords.clear()
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

def getCenterMass(pixels):
    threshold = filters.threshold_otsu(pixels)
    if threshold == 0:
        return IMAGE_SIZE // 2, IMAGE_SIZE // 2
    labeled = (pixels > threshold).astype(int)
    properties = regionprops(labeled, pixels)
    centerMass = properties[0].weighted_centroid
    return centerMass

def getPixels():
    img = pygame.image.tostring(screen, 'RGBA', False)
    img = Image.frombytes('RGBA', (WIDTH, HEIGHT), img)
    img = ImageOps.grayscale(img)
    img = img.crop(img.getbbox())
    img = addPadding(img, PIXEL_PADDING, COLOR_BLACK)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    pixels = np.array(list(img.getdata()))
    pixels = np.resize(pixels, (IMAGE_SIZE, IMAGE_SIZE))

    # center image around center of mass
    centerMass = getCenterMass(pixels)
    pixels = translate(pixels, IMAGE_SIZE // 2 - int(centerMass[1]), IMAGE_SIZE // 2 - int(centerMass[0]))

    # save preprocessed image
    if not os.path.isdir('./images/'):
        os.makedirs('./images/')
    img = Image.fromarray(np.uint8(pixels) , 'L')
    img.save(f'./images/screen_{time_ns()}.png')

    # normalize pixels
    pmax = np.max(pixels)
    if pmax != 0:
        pixels = pixels / pmax

    # reshape array to match MNIST format
    pixels = pixels.reshape(1, -1)
    return pixels

# translate the given array to the new center
def translate(arr, dx, dy):
    arr = np.roll(arr, dy, axis=0)
    arr = np.roll(arr, dx, axis=1)
    if dy > 0:
        arr[:dy, :] = 0
    elif dy < 0:
        arr[dy:, :] = 0
    if dx > 0:
        arr[:, :dx] = 0
    elif dx < 0:
        arr[:, dx:] = 0
    return arr

def addPadding(img, padding, color):
    w, h = img.size
    nw = w + 2 * padding
    nh = h + 2 * padding
    nimg = Image.new(img.mode, (nw, nh), color[0])
    nimg.paste(img, (padding, padding))
    return nimg

if __name__ == '__main__':
    global STATS
    STATS = True if input('Compute statistics? (y/n) ') == 'y' else False
    SAY = True if input('Say predicted digit? (y/n) ') == 'y' else False
    main()
