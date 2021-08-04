Generation of Synthetic Join Functions
===============================================================================

To compare the impact of the number of prime paths, we design a set of synthetic join functions based on the functions 
selected from the project [scikit-learn](https://github.com/scikit-learn/scikit-learn"). First, we choose some functions 
in scikit-learn as native functions. 
We choose these native functions because they are similar to the operations that the join functions can perform 
(e.g., data transformation and decision logic). However, the selected native functions only have certain prime path 
numbers. There is no native function with ten prime paths, eleven prime paths, thirteen prime paths, or fourteen prime 
paths. Therefore, we use the following methods to generate the synthetic functions with one to sixteen prime paths for 
testing.

- *Extraction*:
    - Since some statements in native functions are independently operable, we can extract these statements and use 
    them to construct new synthetic functions.
- *Assembly*:
    - Since the operations in some native functions are compatible (i.e., performing operations on the same type of 
    data), we can construct new synthetic functions assembled from these native functions.
