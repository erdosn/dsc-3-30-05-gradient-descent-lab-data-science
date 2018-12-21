
# Gradient Descent - Lab

## Introduction

In this lab, we'll continue to formalize our work with gradient descent and once again practice coding some implementations, starting with a review of linear regression. In the upcoming labs, you'll apply similar procedures to implement logistic regression on your own.

## Objectives
You will be able to:
* Create a full gradient descent algorithm
* Explain what gradient descent is minimizing and finding
    * minimizing - error of our cost curve
    * finding - local minimum to find the minimal error for our model 
    * by finding the optimal parameter value
* Relate gradient descent to logistic regression
    * What are we minimizing?
        * Error
    * What are we finding?
        * Optimal beta values for the logistic regression function (sigmoid function)
        * by using a loss function that is log-loss

# Outline
* Warm Up (Prior Knowledge Checking)
* Discuss Logistic Regression
* Discuss Gradient Descent
* Go through the algorithm
* Apply to an example
* Wrap Up

# Warm Up 
* What is logistic regression creating? 
* S curve rather than a straight line
* shows a relationship between binary variables
* Curve is called a sigmoid (special log function)
* Relationship describes between input and binary output (doesn't have to classification)
* Bound between 0 and 1

## Using Gradient Descent to Minimize OLS

In order to practice gradient descent, lets begin by investigating a simple regression case in which we are looking to minimize the Residual Sum of Squares (RSS) between our predictions and the actual values. Remember that this is referred to Ordinary Least Squares (OLS) regression. Below, is a mock dataset that we will work with. Preview the dataset. Then, we will compare to simplistic models. Finally, we will use gradient descent to improve upon these  initial models.

Good luck!


```python
#The dataset
import pandas as pd
df = pd.read_excel('movie_data.xlsx')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>domgross</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000000</td>
      <td>25682380</td>
      <td>21 &amp;amp; Over</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45658735</td>
      <td>13414714</td>
      <td>Dredd 3D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000000</td>
      <td>53107035</td>
      <td>12 Years a Slave</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61000000</td>
      <td>75612460</td>
      <td>2 Guns</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000000</td>
      <td>95020213</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (30, 3)



## Two Simplistic Models

Let's imagine someone is attempting to predict the domestic gross sales of a movie based on the movie's budget, or at least further investigate how these two quantities are related. Two models are suggested, and need to be compared.  
The two models are:  
$domgross = 1.575 \bullet budget$  
$domgross = 1.331 \bullet budget$  
Here's a graph of the two models along with the actual data:


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.linspace(start=df.budget.min(), stop=df.budget.max(), num=10**5)
plt.scatter(x, 1.575*x, label='Mean Ratio Model') #Model 1
plt.scatter(x, 1.331*x, label='Median Ratio Model') #Model 2
plt.scatter(df.budget, df.domgross, label='Actual Data Points')
plt.title('Gross Domestic Sales vs. Budget', fontsize=20)
plt.xlabel('Budget', fontsize=16)
plt.ylabel('Gross Domestic Sales', fontsize=16)
plt.legend(bbox_to_anchor=(1,1))
```




    <matplotlib.legend.Legend at 0x10f58a828>




![png](index_files/index_7_1.png)



```python
# how do we calculate error
# sum (y - yhat) ** 2
def rss(m, y, x):
    yhat = m*x
    residual = yhat - y
    return np.sum(residual**2)
```

## Error/Loss Functions

In compare the two models (and future ones), we need to define a metric for evaluating and comparing models to each other. Traditionally this is the residual sum of squares. As such we are looking to minimize  $ \sum(\hat{y}-y)^2$.
Write a function **rss(m)** which calculates the residual sum of squares for a simplistic model $domgross = m \bullet budget$.


```python
def rss(m, X=df.budget, y=df.domgross):
    yhat = m*X
    differences_squared = (yhat - y)**2
    return differences_squared.sum()
```

## Run your RSS function on the two models
Which of the two models is better?


```python
#Your code here
print("Mean slope error = {}".format(rss(1.575)))
print("Median slope error = {}".format(rss(1.331)))
```

    Mean slope error = 2.7614512142376128e+17
    Median slope error = 2.3547212057814554e+17



```python
#Your response here
"""
Median is better. It is lower. - Chris
Median is eh better
"""
```




    '\nMedian is better. It is lower. - Chris\nMedian is eh better\n'



# What does Gradient Descent do?
* Finding the minimum point in a cost curve
    * minimum point - lowest point on the curve
    * cost curve - residual curve - error - xaxis (parameter) vs yaxis (error)

# In this case what is our loss function?
* RSS for each value of m

## Gradient Descent

Now that we have a loss function, we can use numerical methods to find a minimum to the loss function. By minimizing our loss, we have achieved an optimal solution according to our problem formulation. Here's our outline of gradient descent from the previous lesson:  

1. Define initial parameters:
    1. pick a starting point
    2. pick a step size $\alpha$ (alpha)
    3. choose a maximum number of iterations; the algorithm will terminate after this many iterations if a minimum has yet to be found
    4. (optionally) define a precision parameter; similar to the maximum number of iterations, this will terminate the algorithm early. For example, one might define a precision parameter of 0.00001, in which case if the change in the loss function were less then 0.00001, the algorithm would terminate. The idea is that we are very close to the bottom and further iterations would make a negligable difference.
2. Calculate the gradient at the current point (initially, the starting point)
    
    * What does this mean? Calculating a DERIVATIVE of the cost curve at a point
    
    
3. Take a step (of size alpha) in the opposite direction of the gradient
4. Repeat steps 2 and 3 until the maximum number of iterations is met, or the difference between two points is less then your precision parameter  

To start, lets simply visualize our cost function. Plot the cost function output for a range of m values from -3 to 5.

# Based on the graph below, where is the min point?
* where m = 1 - Emily


```python
#Your code here
x = np.linspace(start=-3, stop=5, num=10**3)
y = [rss(xi) for xi in x]
plt.plot(x, y)
plt.title('RSS Loss Function for Various Values of m')
```




    Text(0.5,1,'RSS Loss Function for Various Values of m')




![png](index_files/index_18_1.png)


As you can see, this is a simple cost function. The minimum is clearly around 1. With that, let's try and implement gradient descent in order to find our optimal value for m.


```python
cur_x = 1.5 #Set a starting point
alpha = 10**(-7) #Initialize a step size
precision = 0.0000001 #Initialize a precision
previous_step_size = 1 #Helpful initialization
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter

#Create a loop to iterate through the algorithm until either the max_iteration or precision conditions is met
#Your code here; create a loop as described above
while (previous_step_size > precision) & (iters < max_iters):
    prev_x = cur_x
    #Calculate the gradient. This is often done by hand to reduce computational complexity.
    #For here, generate points surrounding your current state, then calculate the rss of these points
    #Finally, use the np.gradient() method on this survey region. This code is provided here to ease this portion of the algorithm implementation
    x_survey_region = np.linspace(start = cur_x - previous_step_size , stop = cur_x + previous_step_size , num = 101)
    rss_survey_region = [np.sqrt(rss(m)) for m in x_survey_region]
    gradient = np.gradient(rss_survey_region)[50] 
    
    #Update the current x, by taking a "alpha sized" step in the direction of the gradient
    cur_x -= alpha * gradient #Move opposite the gradient
    previous_step_size = abs(cur_x - prev_x)

    #Update the iteration number
    iters+=1

#The output for the above will be: ('The local minimum occurs at', 1.1124498053361267)
print("The local min = {}".format(cur_x))
```

    The local min = 1.1124498064238728


## Plot the minimum on your graph
Replot the RSS cost curve as above. Add a red dot for the minimum of this graph using the solution from your gradient descent function above.


```python
#Your code here
x = np.linspace(start=-3, stop=5, num=10**3)
y = [rss(xi) for xi in x]
plt.scatter(cur_x, rss(cur_x), c='r', s=50, label='minimum={}'.format(cur_x))
plt.plot(x, y)
plt.title('RSS Loss Function for Various Values of m')
plt.legend()
```




    <matplotlib.legend.Legend at 0x11527ad68>




![png](index_files/index_22_1.png)


## Summary 

In this lab you coded up a gradient descent algorithm from scratch! In the next lab, you'll apply this to logistic regression in order to create a full implementation yourself!
