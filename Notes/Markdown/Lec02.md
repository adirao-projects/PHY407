# Lecture 2

## 2024-09-16 12:00

# Numerical Errors

- Remember that non-integer numbers w/ greater than 16sf **are rounded**

## 32 Bit vs 64 Bit

```float32```is a single precision float, ```float64``` is a double precison (i.e. a double in java)

Python attempts to check memory type and use appropriate type by itelf. However, we should instead use it expliciently with ```np.float64```.

The *epsilon* is the smallest non-zero (absolute value) number that can be stored. This is also known as the **machine precision**.

## Error

Relative error **is not the same as fractional error**. Relative error is valued value compared to true value $y$ is $(x-y)/y$

Therefore, we must recall the rules on propogation of errors. Specificially, we numerical deriviatives, we enter the **danger zone**.

$$
\frac{df}{dt} \approx \frac{f_2 - f_1}{\Delta t} 
$$
