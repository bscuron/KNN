import pygame
import os
from PIL import Image, ImageOps, ImageEnhance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import shuffle
import sys
import numpy as np
from time import sleep, time_ns, time # TODO: remove these. just here for debugging
import pandas as pd

# window settings
WIDTH, HEIGHT = 504, 504
screen = pygame.display.set_mode([WIDTH, HEIGHT])

# canvas settings
coords = [] # (x, y) coordinates of user drawing
PEN_WIDTH = min(WIDTH, HEIGHT) // 25
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

# image filter settings
CONTRAST_FACTOR = 1 # the factor to increase the 28x28 representation of the screen's contrast
PIXEL_PADDING = min(WIDTH, HEIGHT) // 8 # when centering the drawing, this is how many pixels will be added to the top, bot, left, and right for padding

# model settings
K_RANGE = range(1, 6)
CROSS_VALIDATION_FOLDS = 5

def main():
    knn = getModel()
    pygame.display.set_caption('Digit Classification')
    pygame.init()
    label = None
    prev_label = None
    while True:
        screen.fill(COLOR_BLACK)
        handleEvents()
        drawCoords()
        pixels = getPixels()
        label = knn.predict(pixels)[0]
        pygame.display.set_caption(str(label))
        if label != prev_label:
            say(label)
        prev_label = label
        # probablities = knn.predict_proba(pixels)[0]
        # print(probablities)
        pygame.display.flip()

# say the given word as a new process in the background
def say(word):
    os.system(f'say "{word}" &')

# train the KNN classification model using the MNIST dataset
def getModel():
    # load dataset split into training/validation and testing
    print('Loading dataset...')
    X_train, X_test, y_train, y_test = loadData()
    print('Loaded dataset')

    # normalize features
    print('Normalizing features...')
    normalizer = StandardScaler()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    print('Normalized features')

    # use cross-validation to select the optimal hyperparameter 'k'
    print('Finding optimal hyperparameter...')
    # k = getK(X_train, y_train)
    k = 4
    print('Found optimal hyperparameter')

    # fit the data to the model with the optimal hyperparameter 'k'
    print('Fitting data...')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print('Fit data')

    # evaluate the model on the testing set
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
    img = img.crop(img.getbbox())
    img = addPadding(img, PIXEL_PADDING, COLOR_BLACK)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    # img.save(f'./images/screen_{time_ns()}.png')
    pixels = np.array(list(img.getdata()))
    pixels = pixels.reshape(1, -1)
    pixels = StandardScaler().fit_transform(pixels.T)
    pixels = pixels.T
    return pixels

def addPadding(img, padding, color):
    w, h = img.size
    nw = w + 2 * padding
    nh = h + 2 * padding
    nimg = Image.new(img.mode, (nw, nh), color[0])
    nimg.paste(img, (padding, padding))
    return nimg

if __name__ == '__main__':
    main()
