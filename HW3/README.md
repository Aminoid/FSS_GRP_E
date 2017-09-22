# Homework 3

This is python code for Homework 3.

## Problem Statement

### Unsupervised Discretization:
Write code that takes a table column of N numbers, sorts in, and breaks into bins of size approximately sqrt(N). Note that these breaks have to satisfy the following sanity rules:
    no range contains too few numbers (sqrt(N));
    each range is different to the next one by some epsilon value (0.2 * standard deviation of that column);
    the span of the range (hi - lo) is greater than that epsilon;
    the lo value of one range is greater than than the hi value of the previous range

### Supervised Discretization:
Write code that reflects over the ranges found by the unsupervised discretizer. Combine ranges where some dependent variable is not changed across that combination of ranges. Specifically, sort the ranges and do a recursive descent of the ranges. At each level of the recursion, break the ranges at the point that most minimizes the expected value of the standard deviation of the dependent variable.

## Setup

All the code is run on `python 2.7`. It also uses `numpy`

## Running

File can be run using:
```
python Discretization.py
```
