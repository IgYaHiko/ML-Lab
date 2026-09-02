

# Lab 1: NumPy Fundamentals

## 📌 Lab Information
- **Lab Number**: 1
- **Topic**: NumPy Array Operations
- **Date**: September 2026
- **Status**: ✅ Completed

## 🎯 Learning Objectives
By the end of this lab, students will be able to:
- Create NumPy arrays with random values using `np.random`
- Use `seed()` for reproducible random number generation
- Slice 2D arrays to extract sub-arrays
- Perform matrix operations (addition and multiplication)
- Reshape arrays between 2D and 3D dimensions
- Understand array shapes and dimensions

## 📋 Lab Exercises

### Exercise 1: Array Creation, Slicing, and Operations

#### Requirements
1. Create a 10×10 array with random integer values (1-100)
2. Use `np.random.seed(42)` for reproducibility
3. Extract three sub-arrays:
   - **Array A**: 5×7 (rows 0-4, columns 0-6)
   - **Array B**: 7×5 (rows 0-6, columns 0-4)
   - **Array C**: 5×7 (rows 0-4, columns 3-9)
4. Calculate:
   - Sum of A and C (element-wise)
   - Matrix multiplication of A and B

#### Solution Code
```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create a 10x10 array with random values
original_array = np.random.randint(1, 100, size=(10, 10))
print("Original 10x10 Array:")
print(original_array)

# Slice array A (5x7)
A = original_array[0:5, 0:7]
print(f"\nArray A (5x7):\n{A}")

# Slice array B (7x5)
B = original_array[0:7, 0:5]
print(f"\nArray B (7x5):\n{B}")

# Slice array C (5x7)
C = original_array[0:5, 3:10]
print(f"\nArray C (5x7):\n{C}")

# Sum of A and C
sum_AC = A + C
print(f"\nSum of A and C:\n{sum_AC}")

# Multiplication of A and B
product_AB = np.dot(A, B)
print(f"\nMultiplication of A and B:\n{product_AB}")
