import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import model_from_json
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid


# Define path to directory where my data is
path = '/Users/paulscemama/AutomatingArith/train'
def create_random_sample(symbol, percentage):
        path = '/Users/paulscemama/AutomatingArith/train'
        if symbol == 'times':
                filenames = np.array([f for f in os.listdir(path + '/' + symbol)])
        else:
                filenames = np.array([f for f in os.listdir(path + '/' + symbol) if f[0] == symbol])
        np.random.seed(1)
        random_samp = np.random.choice(filenames, int(len(filenames)* percentage))
        return random_samp


'-----------------------------------------------------------------------------------'

# create samples
np.random.seed(1)
random_minus = create_random_sample('-', 0.1)
random_plus = create_random_sample('+',0.1)
random_equals = create_random_sample('=',0.2)
random_zeros = create_random_sample('0',0.25)
random_ones = create_random_sample('1', 0.25)
random_twos = create_random_sample('2', 0.25)
random_threes = create_random_sample('3', 0.25)
random_fours = create_random_sample('4', 0.25)
random_fives = create_random_sample('5', 0.25)
random_sixes = create_random_sample('6', 0.25)
random_sevens = create_random_sample('7', 0.25)
random_eights = create_random_sample('8', 0.25)
random_nines = create_random_sample('9', 0.25)
random_times = create_random_sample('times', 0.1)

'-----------------------------------------------------------------------------------'

np.random.seed(1)
minus = Image.open(path + '/-/' + random_minus[np.random.randint(0,len(random_minus))])
plus = Image.open(path + '/+/' + random_plus[np.random.randint(0,len(random_plus))])
equals = Image.open(path + '/=/' + random_equals[np.random.randint(0,len(random_equals))])
zero = Image.open(path + '/0/' + random_zeros[np.random.randint(0,len(random_zeros))])
one = Image.open(path + '/1/' + random_ones[np.random.randint(0,len(random_ones))])
two = Image.open(path + '/2/' + random_twos[np.random.randint(0,len(random_twos))])
three = Image.open(path + '/3/' + random_threes[np.random.randint(0,len(random_threes))])
four = Image.open(path + '/4/' + random_fours[np.random.randint(0,len(random_fours))])
five = Image.open(path + '/5/' + random_fives[np.random.randint(0,len(random_fives))])
six = Image.open(path + '/6/' + random_sixes[np.random.randint(0,len(random_sixes))])
seven = Image.open(path + '/7/' + random_sevens[np.random.randint(0,len(random_sevens))])
eight = Image.open(path + '/8/' + random_eights[np.random.randint(0,len(random_eights))])
nine = Image.open(path + '/9/' + random_nines[np.random.randint(0,len(random_nines))])
times = Image.open(path + '/times/' + random_times[np.random.randint(0,len(random_times))])

'-------------------------------------------------------------------------------------------'

fig = plt.figure(figsize=(4.,4.))
grid = ImageGrid(fig, 111,
            nrows_ncols = (2,7),
            axes_pad = 0.05)

for ax, im in zip(grid, [minus,plus,equals,zero,one,two,three,four,five,six,seven,eight,nine,times]):
    ax.set_axis_off()
    ax.imshow(im)

plt.show()


'----------------------------------------------------------------------------------------------'

# Need to now create lists of images of proper expression form
np.random.seed(1)
# Make a dataset of shuffled digits (exclude 0)
import itertools
random_digit_files = itertools.chain(random_ones,random_twos,random_threes,
    random_fours,random_fives,random_sixes,random_sevens,random_eights,
    random_nines)
random_digit_files = list(random_digit_files)
np.random.shuffle(random_digit_files)

# Make a dataset of shuffled operators
random_operator_files = itertools.chain(random_minus, random_plus, random_times)
random_operator_files = list(random_operator_files)
np.random.shuffle(random_operator_files)


'--------------------------------------------------------------------------------------------------'

def random_expression_generator(digit_files, operator_files):
    length = np.random.randint(3,10)
    if (length % 2) != 0:
        length += 1
    expression = []
    for i in range(length-1):
        expression.append(digit_files[np.random.randint(0,len(digit_files))])
        expression.append(operator_files[np.random.randint(0,len(operator_files))])
    expression.append(digit_files[np.random.randint(0,len(digit_files))])
    return expression

np.random.seed(1)
unmerged_expression = []
for i in range(5):
    unmerged_expression.append(random_expression_generator(random_digit_files, random_operator_files))

'------------------------------------------------------------------------------------------------'

## Construct the (unfinished) dataset of expressions

def merge_images(expression):
        images = []
        for element in expression:
            if element[0:3] == 'exp' or element[0:5] == 'times':
                image = Image.open(path + '/times/' + element)
            else:
                image = Image.open(path + '/' + element[0] + '/' + element)
                            
            as_np = np.asarray(image)
            mod_image = np.hstack((as_np, np.ones(len(as_np)).reshape(len(as_np),1)*255))
            mod_image = Image.fromarray(mod_image)
            mod_image = mod_image.convert("L")
            images.append(mod_image)

        print(images)
        images_com = np.hstack((np.asarray(image) for image in images))
        images_com = Image.fromarray(images_com)
        print(images_com)
        #imgs_com.save('exp_'+str(i))
        images_com.save('Test.jpg')

test = merge_images(unmerged_expression[0])