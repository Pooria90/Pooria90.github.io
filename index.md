# Computational Intelligence

- Instructor   : [Sepideh Hajipour](http://sharif.edu/~hajipour/)
- Assistant    : [Pooria Ashrafian](https://pooria90.github.io/)

This page accompanies the class material of EE25-729 at [the Sharif University of Technology](https://en.sharif.edu/). Feel free to write your questions/comments to <pooria.ashrafian@gmail.com>.



## Table of Contents

- [A gentle intro to Python](#a-gentle-intro-to-python)
  * [Writing Python](#writing-python)
  * [Installing a Python package](#installing-a-python-package)
  * [Python shell](#python-shell)
  * [Basic data types](#basic-data-types)
  * [Lists](#lists)
  * [Tuples](#tuples)
  * [Dictionaries](#dictionaries)
  * [Conditional statements](#conditional-statements)
  * [Loops](#loops)
    + [For loops](#for-loops)
    + [While loops](#while-loops)
    + [Loop control](#loop-control)
  * [Functions](#functions)
  * [Classes](#classes)
- [Numpy](#numpy)
  * [Array](#array)
  * [Functions to create arrays](#functions-to-create-arrays)
  * [Array indexing](#array-indexing)
    + [Slicing](#slicing)
    + [Boolean indexing](#boolean-indexing)
    + [Array operations](#array-operations)
- [Matplotlib](#matplotlib)
- [TensorFlow and Keras](#tensorflow-and-keras)
  * [Exploring the data](#exploring-the-data)
  * [Split train and validation dataset](#split-train-and-validation-dataset)
  * [Converting the labels to one-hot format](#converting-the-labels-to-one-hot-format)
  * [Building the model](#building-the-model)
    + [Setting up layers](#setting-up-layers)
    + [Compiling the model](#compiling-the-model)
  * [Training the model](#training-the-model)
- [Pyeasyga](#pyeasyga)



## A gentle intro to Python

Python was developed in the early 90's by Guido van Rossum at the National Research Institute for Mathematics and Computer Science in the Netherlands. 

Python is designed to be highly readable. It uses English keywords frequently, and it has fewer syntactical constructions than other languages.

- **Python is interpreted**: You don't need to compile your code. It is processed at runtime like MATLAB.
- **Python is interactive**: You can use command prompt and interact with the interpreter, like command line in MATLAB.
- **Python is object-oriented**: You can define custom classes and objects, and use inheritance from parent classes.

You can download a suitable version from [python.org](https://www.python.org). 



### Writing Python

To run Python codes, you may either write your code in a Python script (`.py` format) or a Python notebook (`.ipynb` format).

For writing Python scripts you can either use the default Python editor in Python standard library, **IDLE**, or any other editor for this purpose like **VS Code** or **PyCharm**.

Notebooks provide you with the the capability called **cell**. You can add several cells in a notebook and run each cell separately. You can either install **Jupyter notebook** after installing Python, or use Colab notebooks at [colab.research.google.com](https://colab.research.google.com/). The benefit of using Colab is that you have access to many preinstalled Python packages.



### Installing a Python package

After you successfully installed Python and set up Python path, you can open your command prompt and install additional packages by using `pip`. For example, in order to install Jupyter notebook you can run:

```shell
pip install notebook
```

Then to run Jupyter notebook enter `jupyter notebook` in your cmd.

### Python shell

From now on, you can use scripts, notebooks, or even command prompt to run code. At the beginning, I strongly recommend you to write some statements in Python shell for getting used to it. Python shell is an enviornment where you can run single Python statements and see the result in place. Type `python` in your command prompt and press `enter`. The Python shell opens like this:

```shell
C:\Users\Asus>python
Python 3.8.3 (tags/v3.8.3:6f8c832, May 13 2020, 22:20:19) [MSC v.1925 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Try to interact with the interpreter. Two basic functions to try are `print` and `type`. With the former you can show a variable to the user, and by using the latter you can use data type (or *class*) of a variable. An example of playing with the shell can be like the following box. As you may notice, `#` is used to define single-line comments in Python. The interpreter ignores everything after `#`.

```shell
>>> print ('Hello World!') # Strings can be defined by '' or "".
Hello World!
>>> x = 216
>>> print (x)
216
>>> type(x) # x is integer because it is define without a floating point
<class 'int'>
>>> y = 12.56 # y is floating point number
>>> type(y)
<class 'float'>
```

To define a variable, you may use lower or upper case letters, or underscore.

```shell
>>> family_name = 'Ashrafian'
>>> print (family_name)
Ashrafian
>>> type(family_name) # it is a string (str) variable
<class 'str'>
>>> Course = 'Computational Intelligence'
>>> print (Course)
Computational Intelligence
```

 

### Basic data types

**Numbers**: You have already seen integer and float numbers. In the following example you get familiar with arithmetic and assignment operators.

```python
x = 12  # = is assignment operator; As you have seen in C or MATLAB

print (type (x))  # prints <class 'int'>

print (x + 2)  # prints 14
print (x - 3)  # prints 9

print (x / 4)  # prints 3.0; / is float division. It means the result is a float
print (type (x/4)) # prints <class 'float'>

print (x // 5)  # prints 2; // is the integer division, or so called 'quotient'
print (x ** 2)  # prints 144; ** is the power operation

# Assignment operators
x += 3
print (x)  # prints 15
x -= 1
print (x)  # prints 14
x *= 3
print (x)  # prints 42
x **= 2
print (x)  # what do you think?
```



**Booleans**: They can be either `True` or `False`. Logical operators (`and`, `or`, `not`) and comparison (`==`, `>`, `>=`, `!=` (not equal), ...) operators work on them.

```python
x = True
print (type (x))  # prints <class 'bool'>
print (not x)  # prints False

y = 15
print (y > 5)  # prints True

p = y > 5
print (type (p)) # prints <class 'bool'>
print (y == 12)  # prints False

print ((y > 3) and (y < 10))  # prints False; because one of them is False
print ((y > 3) or (y != 10))  # prints True; because one of them is Ture
```



**Strings**: We said that they are either defined by single quote (`''`) or double quote (`""`).

```python
s1 = 'computational'
s2 = 'intelligence'

L1 = len (s1)  # len counts the number of characters
print (L1)  # 13 characters; yeah?

print (s1 + s2)	# prints computationalintelligence (+ for str = concatenation)
s_cat = s1 + ' ' + s2
print (s_cat)  # prints computational intelligence
print (f'hello from {s_cat}')  # You see that f and {}? Those are for formatted output; prints hello from computational intelligence
```

Strings have useful methods. Wait...What is a method?

- Method is a function. However, unlike the functions that `print` and `len` that we have already seen, it is defined for its object. So, without defining a string, there is **no string method**. Methods are applied using a `.` after the variable. For example:

```python
s = 'Pooria'
print(s.upper())  # makes everything uppercase: 'POORIA'
print(s.lower())  # makes everything lowercase: 'pooria'

s = 'i am pooria'
print(s.capitalize()) # make first letter capital: 'I am...
print(s.find('p'))  # finds the index of 'p' by starting from zero; prints 5
print(s.split())  # splits s into a list of words: ['i', 'am', 'pooria']
```

For a complete list refer to Python documentations in [here](https://docs.python.org/3/library/stdtypes.html#string-methods).



### Lists

A `list` in Python is an ordered collection of elements. When we say *element*, we mean that it doesn't matter that the members of the list are not the same type. To define a `list` you can use square brackets (`[]`).

```python
L1 = []  # empty list
print (type(L1))  # prints <class 'list'>

L2 = [2, 4.5, 'hello', [1,2,3]]  # a list of different element types
print(len(L2))  # prints 4; L2 has 4 elements
```

**index**: Similar to the arrays in C or C++, the first index in a list is 0 (the first element). In Python, there is also an index system that start at the end of the list. The final element is indexed as -1, the element before the last is -2, and so on.

```python
L = [12, 5, 13, 18, 15]
print (len(L))  # a list of 5 elements

print (L[0])  # prints 12; the first element
print (L[2])  # prints 13; the third element

print (L[-1])  # prints 15; the last element
print (L[-2])  # prints 18; the last element
```

**methods**: Maybe, the most useful method for working with a `list` object is `append`. It is used to add a single element at the end of a `list`. For a complete list of methods, check out Python documentations.

```python
L = [10, 8, 6]
print (L)  # prints [10, 8, 6]

L.append(2)
L.append(5)
L.append(8)
print (L)  # prints [10, 8, 6, 2, 5, 8]

L.remove(8)  # removes the first 8 that is in the list
print (L)  # prints [10, 6, 2, 5, 8]

L.reverse()  # reverses the order of the list
print (L)  # prints [8, 5, 2, 6, 10]

z = L.pop(-1)  # removes the element at the specified position and returns that element
print (L)  # prints [8, 5, 2, 6]

L.extend([20, 10])  # adds the list in the () to L
print (L)  # prints [8, 5, 2, 6, 20, 10]
```

Remember an important point about `list` objects. When you assign a `list` to another one using `=`, the Python assigns the reference of the primary `list`. So, both of the variables point to the same memory space. As a result, any alternation in one of the lists will happen similarly in the other one.

To avoid this, you can do the assignment with `copy` method.

```python
L1 = [4, 5, 'salam', 'hello']
L2 = L1
L1.append('hey')  # 'hey' is appended to both lists
print (L1)  # prints [4, 5, 'salam', 'hello', 'hey']
print (L2)  # prints [4, 5, 'salam', 'hello', 'hey']

L3 = L1.copy()
L1.remove('hello')
print (L1)  # prints [4, 5, 'salam', 'hey']
print (L2)  # prints [4, 5, 'salam', 'hey']
print (L3)  # prints [4, 5, 'salam', 'hello', 'hey']; L3 is not affected
```

**slicing**: There several ways to slice a `list` (taking elements of some indices).

```python
# L[beg:end] ; returns the elements starting at index beg, ending at end-1
# L[beg:end:step] ; returns the elements starting at index beg, increasing with step, ending at end-1
# L[:end] ; from the begining to end-1
# L[beg:] ; from beg to the last element

L = [3, 4, 10, 45, 6, 13, 9]
print (L[1::2])  # prints [4, 45, 13]; end not specified (means until the last element)
print (L[:-2])  # prints [3, 4, 10, 45, 6]
print (L[1:5])  # prints [4, 10, 45, 6]

L[1:4] = [1, 2, 3]  # change part of the list
print (L)  # prints [3, 1, 2, 3, 6, 13, 9]
```



### Tuples

Tuples are very similar to lists, for example they are ordered. However, unlike lists, they are *immutable*. It means that you can't change a single element of a tuple, or append something to it. If you want to change something in a tuple, you have to assign a whole new tuple to that. Tuples are defined using `()`. The indexing and slicing methods also work here.

```python
t = (21, 10, 'sepideh', 'hajipour', 5.5, 10, 55)
print (type(t))  # prints <class 'tuple'>
print (t[-1])  # prints 55
print (t[1:3])  # prints (10, 'sepideh')

print (t.count(10))  # prints 2; there are two 10s in the tuple
print (t.index('hajipour'))  # prints 3; 'hajipour' is in the index 3

t[1] = 40  # TypeError: 'tuple' object does not support item assignment
```



### Dictionaries

A dictionary stores (key, value) pairs. Think of the key as a *word* and the value as its *meaning* in an English dictionary like Merriam Webster.

The value can be any kind of element, but the key should be an immutable element (so the key cannot be a list). To define a dictionary you can use `{}` and separate a key from its value using `:`. 

```python
d1 = {'ali':1, 10:30, 20:[4,5]}
print (type(d1))  # prints <class 'dict'>

d1['eli'] = 3  # adding another (key, value) pair
print (d1)  # prints {'ali': 1, 10: 30, 20: [4, 5], 'eli': 3}
```

**methods**: There are some useful dictionary methods in Python.

 ```python
 d = {1:1, 2:4, 3:9, 4:16}
 
 print (d.keys())  # prints dict_keys([1, 2, 3, 4])
 print (d.values())  # prints dict_values([1, 4, 9, 16])
 print (d.items())  # prints dict_items([(1, 1), (2, 4), (3, 9), (4, 16)])
 
 # You can easily convert each of the following to a list by applying list()
 L = list(d.items())
 print (L)  # prints [(1, 1), (2, 4), (3, 9), (4, 16)]
 
 print (d.get(2))  # prints the value of the key 2; prints 4
 print (d.get(5))  # prints None because 5 is not in the keys
 
 d.update({5:25, 6:36})  # updates d with the dictionary in ()
 print (d)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}
 ```



### Conditional statements

In Python, we use `if`, `elif` (called else if in C), and `else` to define conditional statements. The general structure is given below. Recall that in C, when a condition was true, everything inside a `{}` block was executed. But in Python we just *indent* (usually using a tab or 4 white space) the block.

```markdown
if condition1:
	do something (pay attention to the indent)
elif condition2:
	do something else
elif condition3:
	do something else
else:
	do something else if none of the above conditions hold
```

Here is a child example:

```python
grade = int(input('Enter your computational intelligence grade: '))

if grade >= 18:
    print ('Wow you did great!')
elif grade >= 15:
    print ('You did well.')
elif grade >= 10:
    print ('You need to study more.')
else:
    print ('You have failed the course.')
```



### Loops

#### For loops

You can use a for loop to iterate over lists, tuples, dictionary keys or values or items, or range of integer numbers. Just like the conditional statements you have to use `:` and indent to define the block inside the loop.

```python
L = [1, 3, 5] # or a tuple like (1, 3, 5)
for item in L:
    print (item ** 2)
# prints 1, 9, 25

d = {'ali':1, 'pooria':2, 'ahmad':3}

for name in d.keys():
    print (name)
# prints 'ali', 'pooria', 'ahmad'

for k,v in d.items(): # iterate over (key,value) pairs
    print (f'key: {k} --- value: {v}')
# prints key: 'ali' --- value: 1 ...

# range(start, stop, step)
# start and step are optional. Default value for start is 0 and for step is 1
# range(5): 0,1,2,3,4
# range(2,6): 2,3,4,5
# range(2,7,2): 2,4,6

for i in range(100):
    if i%15 == 0: # or simply 'for i in range(0,100,15)' and without 'if'
        print (i)
# prints 0, 15, ..., 90
```

#### While loops

As you already know, while loops are defined using a condition (a *Boolean statement*). The indented statements after the while loop will be executed until the the condition is `True`.

```python
counter = 0
while counter < 10:
    counter += 1
    print (f'Execution number {counter}')
# prints Execution number 1, ... , Execution number 10
```

#### Loop control

You can use `continue` and `break` to control the execution of a loop under certain condition. For example in the following loop ignores multipliers of 3 and prints natural numbers of the forms 3k+1 and 3k+2. When the counter equals 100, the execution ends.

```python
j = 0
while True:
    j += 1
    if j%3 == 0:
        continue
    if j == 100:
        break
    print (j)
    
# prints 1,2,4,5,...,94,95,97,98
```



### Functions

To define a function in Python you can use the following structure using the keyword `def`.

```markdown
def function_name (input1, input2, ...):
	statement 1
	statement 2
	...
	return something (or nothing!)
```

For example the following function checks whether the input is prime or not (you may notice that we don't specify the type of input variable. For checking that the input is correct or not one approach is using the built-in function `isinstance` and the keyword `assert`. Search them!).

```python
def isprime(n):
    result = True
    for i in range(2,n):
        if n%i == 0:
            result = False
            break
    return result

print (isprime(21)) # prints False
print (isprime(23)) # prints True
```

You can also define functions using inputs with default values. In this case if you do not specify the value of those inputs, the function uses the default value when called.

```python
def devide(a, b=10):
    return a/b

print (devide(5))  # prints 0.5; uses default value of b
print (devide(3,4)) # prints 0.75; here b=4
print (devide(b=50,a=5))  # prints 0.1; note that you can call a function by telling exactly which variable has what value. In this case the order of the inputs doesn't matter.
```



### Classes

You have already seen several Python built-in classes like `list` and `dict`. You can define your custom classes. To do so, you should specify the attributes (or variables) and methods of your objects. The general structure is like below:

```markdown
class class_name ():
	def __init__(self, input1, input2, ...): (This is the constructor of your objects; tells the Python how to intialize the objects)
		self.attribute1 = something
		self.attribute2 = something
		...
		statement1
		statement2
		...
	
	def method1(self, some inputs):
		some statements
		
	def method2(self, some inputs):
		some statements
		
	...
```

`self` represents the instance of the class. By using the `self` keyword we can access the attributes and methods inside the class to write out statements.

And you can define the object like this:

```markdown
object = class_name(input1, input2, ...)
object.attribute1
object.method1(some inputs)
```

 Look at the example below. In this example we define a point in 2D space.

```python
import math

class point():
    def __init__(self,x,y,name='A'):
        self.x = x  # height
        self.y = y  # width
        self.name = name  # label of the point
    
    def distance(self, p): # computes distance from another point
        d = math.sqrt ( (p.x - self.x)**2 + (p.y - self.y)**2 )
        return d
    
    def norm(self):  # euclidean norm or distance from origin
        n = math.sqrt (self.x**2 + self.y**2)
        return n

a = point(x=3,y=4,name='A')
print (a.x, a.y, a.name)  # prints 3, 4, A
print (a.norm())  # prints sqrt(3^2+4^2)=5

b = point(x=0,y=4,name='B')
print (a.distance(b))  # prints 3; a and b in 2D space are 3 units apart
```

For a complete explanation for classes, check out the [documentations](https://docs.python.org/3.5/tutorial/classes.html)

## Numpy

Numpy is the fundamental library for scientific computing in Python. It provides multi-dimensional arrays and matrices, and the tools for working with them.

### Array

A numpy array is a grid of values, all of the same type (`numpy.array.dtype`), and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array (`numpy.ndarray.ndim`) and the shape of an array is a tuple of integers giving the size of the array along each dimension (`numpy.ndarray.shape`).

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

```python
import numpy as np

a = np.array([1,2,3,4,5])
print (type(a))  # prints <class 'numpy.ndarray'>
print (a.ndim)   # prints 1; rank of the array
print (a.shape)  # prints (5,)
print (a.dtype) # prints int32

print (a[0],a[-1]) # prints 1 5
a[0] = 3
print (a)  # prints [3 2 3 4 5]

b = np.array([[1,2,3],[4,5,6]])
print (b.ndim)   # prints 2
print (b.shape)  # prints (2,3)
print (b[0,0], b[1,1], b[0,2], b[1,2]) # prints 1 5 3 6
```



### Functions to create arrays

There are some useful functions in for making special arrays.

```python
z = np.zeros((1,3))  # creates array with zero entries with the input tuple shape
print (z)

o = np.ones((2,2))  # creates array with one entries with the input tuple shape
print (o)

e = np.eye(3)  # creates identity matrix with number of rows and columns equal to the input integer
print (e)

r = np.random.random((3,2))  # creates random array with entries between 0-1 with the input shape
print (r)
```

For more details check out [array creation](https://numpy.org/doc/stable/user/basics.creation.html) on numpy documentations.



### Array indexing

#### Slicing

```python
a = np.reshape(np.arange(16), newshape=(4,4)) # 0 to 15 in 4*4 array form

print (a[:2,1:3])
'''prints
[[1 2]
 [5 6]]
'''

print (a[2,:]) # third row
# prints [ 8  9 10 11]

print (a[:,1]) # second column 
# prints [ 1  5  9 13]

print (a[:,1:2]) # second column as a column!
'''prints
[[ 1]
 [ 5]
 [ 9]
 [13]]
'''

# Look at the two last examples carefully.
# Accessing data using only an integer (1)
# yields to an array with lower rank. However,
# Accessing using slicing (1:2) yields to an array
# with the same rank.
```

#### Boolean indexing

Boolean array indexing lets you pick out arbitrary elements of an array. This type of indexing is used to select the elements of an array that satisfy some condition.

```python
y = np.array([1,2,3,3,1,0,1,3,2,3])

I = (y > 1) 
print (I)  # prints [False  True  True  True False False False  True  True  True]
print (y[I])  # prints [2 3 3 3 2 3]

np.set_printoptions(precision=2) # for printing numpy arrays with 2 digits
S = np.random.random((4,5))
I = S < 0.3
print (S)
print (I)
print (S[I])
'''A sample output (it's random!):
[[0.3  0.39 0.74 0.18 0.11]
 [0.41 0.03 0.25 0.01 0.65]
 [0.7  0.86 0.71 0.01 0.49]
 [0.98 0.6  0.99 0.27 0.93]]
 
[[False False False  True  True]
 [False  True  True  True False]
 [False False False  True False]
 [False False False  True False]]
 
[0.18 0.11 0.03 0.25 0.01 0.01 0.27]'''
```

#### Array operations

```python
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

print(x + y)
print(np.add(x, y))
'''Elementwise sum
[[ 6  8]
 [10 12]] '''

print(x - y)
print(np.subtract(x, y))
'''Elementwise difference
[[-4 -4]
 [-4 -4]] '''

print(x * y)
print(np.multiply(x, y))
'''Elementwise product
[[ 5 12]
 [21 32]] '''

print(x / y)
print(np.divide(x, y))
'''Elementwise division
[[0.2  0.33]
 [0.43 0.5 ]] '''

print(np.sqrt(x))
'''Elementwise square root
[[1.   1.41]
 [1.73 2.  ]] '''

print(np.dot(x,y))
print(x.dot(y))
'''matrix/vector multiplication
[[19 22]
 [43 50]] '''
```



## Matplotlib

For our visualizations, `matplotlib.pyplot` is all we need. Let's import the module and set up figure and axes.
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
```
`plt.subplots()` is a function that returns a tuple containing a figure and axes object(s). `fig` is useful for saving what we have drawn using `ax`. 
Now I draw a sine wave to show how `ax` works.

```python
import numpy as np

t = np.linspace(0, 1, 100)
f = 5
x = np.sin(2 * np.pi * f * t)

ax.plot(t,x)
plt.show() # To display open figures; fig in here
```

After running we see this:

![image](images/sine_1.png)

Let's make it prettier using `ax`'s methods:

```python
ax.plot(t, x)
ax.set_title(f'Sine wave with freq={f}')
ax.set_xlabel('Time')
ax.set_ylabel('Signal value')
ax.grid(True)
plt.show()
```

There we go:

![sine_2](images/sine_2.png)

You can also define a grid of subplots:

```python
fig, ax = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        t = np.linspace(0, 1, 200)
        f = i*2 + j + 1
        x = np.sin(2 * np.pi * f * t)
        ax[i,j].plot(t, x)
        ax[i,j].set_xlabel('Time')
        ax[i,j].grid(True)
plt.show()
```

![sine_3](images/sine_3.png)



## TensorFlow and Keras

**TensorFlow** is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. 

**Keras** is the high-level API of TensorFlow for deep learning applications. Keras makes it much easier than TensorFlow core to define neural networks and train them. Getting familiar with Tensorflow core is highly recommended to understand computational graphs, forward, and backward pass. However, we use Keras for our purposes in this tutorial because TensorFlow core is beyond the outline of this course and it's more suitable for a graduate course in deep learning.

From now on we, walk through an example and take what we need. You can install tensorflow on your computer by running `pip install tensorflow` in your command prompt, or you can simply use colab notebooks. For information regrading TensorFlow installation refer to [this page](https://www.tensorflow.org/install).

For our example we use Fashion-MNIST dataset. This dataset contains 70000 28*28 grayscale images with 10 different labels. Our first step is to take a look at our data.

### Exploring the data

```python
# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras

f_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = f_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # prints (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
```

Since the pixels of a grayscale image are in 0-255 range, we divide the pixels values by 255 for standardization.

```pyth
x_train, x_test = x_train/255, x_test/255
```

Now we plot some samples:

```python
import matplotlib.pyplot as plt

classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

fig, ax = plt.subplots(5,5,figsize=(10,10))
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(x_train[5*i+j],cmap=plt.cm.binary)
        ax[i,j].set_xticks([]) # no numbers on x-axis
        ax[i,j].set_yticks([]) # no numbers on y-axis
        ax[i,j].set_xlabel(classes[y_train[5*i+j]])  # put the name of the sample on x-axis
```

![f_mnist](images/f_mnist.png)

 ### Split train and validation dataset

Assume that we want to randomly choose 20% of the training set as validation set, and use the the rest for training. Here we can do that by two different approches.

**Permutation:** We shuffle the samples using a random permutation, then take the first 80% as training and next 20% as validation.

```python
import numpy as np
from math import floor

samples = x_train.shape[0]  # 60000
perm = np.random.permutation(samples)
tr_size = floor(0.8*x_train.shape[0])  # 48000
x_tr = x_train[perm[:tr_size],:,:]
y_tr = y_train[perm[:tr_size]]
x_va = x_train[perm[tr_size:],:,:]
y_va = y_train[perm[tr_size:]]
print(x_tr.shape, y_tr.shape, x_va.shape, y_va.shape)  # prints (48000, 28, 28) (48000,) (12000, 28, 28) (12000,)
```

**Sklearn:** We can use `train_test_split` function of `sklearn` library.

```python
from sklearn.model_selection import train_test_split

x_tr, x_va, y_tr, y_va = train_test_split(x_train,y_train,test_size=0.2)
print(x_tr.shape, y_tr.shape, x_va.shape, y_va.shape)  # prints (48000, 28, 28) (48000,) (12000, 28, 28) (12000,)
```



### Converting the labels to one-hot format

We usually convert our labels to a one-hot vector in order to use MSE or Cross Entropy loss. In one-hot format we convert each labels to a vector with its length equal to the number of classes (10 in here). The vector elements all have zero value except in the position of the label.

```python
from tensorflow.keras.utils import to_categorical

y_tr_hot = to_categorical(y_tr, num_classes=10)
y_va_hot = to_categorical(y_va, num_classes=10)
y_te_hot = to_categorical(y_test, num_classes=10)

print(y_tr_hot.shape, y_va_hot.shape, y_te_hot.shape) # prints (48000, 10) (12000, 10) (10000, 10)
```



### Building the model

We define an MLP and train that on `x_tr`.

#### Setting up layers

To define our layers, we use `keras.Sequential`. `keras.Sequential` converts a list of layers into a [`keras.Model`](https://keras.io/api/models/model#model-class).

```python
# import necessary layers of our model
from tensorflow.keras.layers import Input, Flatten, Dense, ReLU, Softmax

# defining the model
model = keras.Sequential([
    Input(shape=(28,28)),
    Flatten(), # makes a 784-element vector to use in our MLP
    Dense(units=256), # our first hidden layer with 256 neurons
    ReLU(), # ReLU activation at our first hidden layer
    Dense(units=64), # our second hidden layer with 64 neurons
    ReLU(), # ReLU activation at second hidden layer
    Dense(units=10), # classification layer; 10 classes
    Softmax(axis=1) # converts output of classification neurons to a probablity
])

model.summary()
```

By using `model.summary()`, you can see the details of your model layers. Its print an output like below. We have an MLP with hidden size of 256 and 64, and ReLU activations. The `None` keyword in the summary refers to the size of mini-batches.

```markdown
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_1 (Flatten)         (None, 784)               0         
                                                                 
 dense_3 (Dense)             (None, 256)               200960    
                                                                 
 re_lu_2 (ReLU)              (None, 256)               0         
                                                                 
 dense_4 (Dense)             (None, 64)                16448     
                                                                 
 re_lu_3 (ReLU)              (None, 64)                0         
                                                                 
 dense_5 (Dense)             (None, 10)                650       
                                                                 
 softmax_1 (Softmax)         (None, 10)                0         
                                                                 
=================================================================
Total params: 218,058
Trainable params: 218,058
Non-trainable params: 0
_________________________________________________________________
```

#### Compiling the model

Now we compile the model to determine optimization method (Stochastic Gradient Descent in here), loss function (Cross Entropy in here), and metrics (accuracy in here).

```python
model.compile(
    optimizer = keras.optimizers.SGD(learning_rate=0.001),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy'] # we want to track the accuracy on training and validation sets during training
)
```

### Training the model

We fit our model to training data using `fit` method. By calling this method, model will be training using backpropagation and the history of training (involving losses and metrics values) will be return.

```python
hist = model.fit(
    x_tr, y_tr_hot, # training data and labels
    epochs = 50, # number of training iterations
    batch_size = 200, # size of mini-batches for batched training
    validation_data = (x_va, y_va_hot)
)
```

The last 5 steps of training look something like this:

```markdown
Epoch 46/50
240/240 [==============================] - 2s 7ms/step - loss: 0.3745 - accuracy: 0.8608 - val_loss: 0.4656 - val_accuracy: 0.8451
Epoch 47/50
240/240 [==============================] - 2s 7ms/step - loss: 0.3721 - accuracy: 0.8638 - val_loss: 0.4634 - val_accuracy: 0.8454
Epoch 48/50
240/240 [==============================] - 2s 7ms/step - loss: 0.3713 - accuracy: 0.8630 - val_loss: 0.4655 - val_accuracy: 0.8452
Epoch 49/50
240/240 [==============================] - 2s 7ms/step - loss: 0.3688 - accuracy: 0.8646 - val_loss: 0.4646 - val_accuracy: 0.8428
Epoch 50/50
240/240 [==============================] - 2s 7ms/step - loss: 0.3662 - accuracy: 0.8648 - val_loss: 0.4618 - val_accuracy: 0.8448
```

You may plot the training and validation accuarcy:

```python
print (hist.history.keys()) # prints dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

n_epochs = len(hist.history['accuracy'])
epochs = np.arange(1,n_epochs+1)

fig, ax = plt.subplots()
ax.plot(epochs, hist.history['accuracy'])
ax.plot(epochs, hist.history['val_accuracy'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(['Train','Validation'])
ax.grid(True)
```

![acc](images/acc.png)

Finally you can make predictions on test set.

```python
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
acc = np.sum(y_pred==y_test)/y_test.shape[0] * 100
print(f'Test Accuracy = {acc}') # prints Test Accuracy = 83.5 for me
```



## Pyeasyga

A simple implementation of genetic algorithm.

There is the `GeneticAlgorithm` class in the `pyeasyga` module. In this class, we can define fitness, crossover, ... and then run the algorithm by using `run` method.

As an example, assume that we want to maximize the function $f(x)=-\frac{x^2}{10} + 6x$  on integers in 0 to 1023.


```python
from pyeasyga import pyeasyga
import random

data = [0]*10 # length of data determines the length of our chromosomes
ga = pyeasyga.GeneticAlgorithm(data)

ga.population_size = 20 # default value is 50
ga.generations = 70 # default is 100

def fitness(individual, data=None):
    # individual: a list of 0 and 1 (chromosome), indicating a candidate solution
    # data: the length of data determines the length of our chromosomes;
    # 		in this problem we don't need any additional data, but in a problem such
    # 		as knapsack problem we need value and weights of objects in the data
    
    s = 0
    for i,b in enumerate(individual):
        s += b * 2**i # converting binary to decimal
    f = -s**2/10 + 6*s
    return f

def crossover(parent1, parent2):
    # two point crossover
    points = sorted([random.randint(0,len(parent1)-1) for _ in (1,2)])
    child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    return child1,child2

ga.fitness_function = fitness
ga.crossover_function = crossover # default is one-point crossover

ga.run()
print(ga.best_individual()) # prints (90.0, [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]) for me
```



