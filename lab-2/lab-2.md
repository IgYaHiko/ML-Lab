# Lab 2 — Pandas DataFrame Basics

## Overview

This assignment demonstrates the use of **Python Pandas** for loading, inspecting, and exploring tabular data.

The notebook works with a `matches.csv` dataset and covers several fundamental Pandas operations that are useful for understanding and analyzing a dataset before performing further data processing or analysis.

The main objective of this assignment is to become familiar with:

* Importing Pandas and NumPy
* Loading a CSV dataset
* Understanding the structure of a DataFrame
* Accessing rows and columns
* Sampling records
* Checking column names and indexes
* Inspecting data types
* Getting dataset information
* Generating statistical summaries
* Identifying missing values

---

## Dataset

The notebook uses the following dataset:

[`dataset`](./matches.csv)

The dataset is loaded into a Pandas DataFrame using:

```python
data = pd.read_csv('./matches.csv')
```

The DataFrame is stored in the variable:

```text
data
```

---

## Libraries Used

### Pandas

Pandas is used for reading and manipulating the tabular dataset.

```python
import pandas as pd
```

### NumPy

NumPy is imported for numerical operations and working with arrays.

```python
import numpy as np
```

---

# Operations Performed

## 1. Import Required Libraries

The notebook begins by importing the required Python libraries:

```python
import pandas as pd
import numpy as np
```

Pandas is the main library used throughout the notebook for DataFrame operations.

---

## 2. Load the Dataset

The `matches.csv` file is loaded using Pandas' `read_csv()` function:

```python
data = pd.read_csv('./matches.csv')
```

The resulting DataFrame is displayed to inspect the dataset.

### Purpose

This step converts the CSV file into a Pandas DataFrame, making it easier to perform data exploration and analysis.

---

# Basic Pandas Commands

The notebook then demonstrates several basic commands for exploring a DataFrame.

## 3. Check the Type of the Object

```python
type(data)
```

This checks the Python data type of the variable `data`.

Since the CSV file is loaded using Pandas, `data` is a:

```text
pandas.DataFrame
```

### Purpose

This confirms that the dataset has been successfully loaded as a Pandas DataFrame.

---

## 4. Convert DataFrame Values to a NumPy Array

```python
data.values
```

The `.values` attribute returns the underlying data as a NumPy array.

### Purpose

This demonstrates how the values stored inside a Pandas DataFrame can be accessed in array form.

---

## 5. Access a Row Using NumPy Array Indexing

```python
data.values[1]
```

This accesses the row at index `1` from the NumPy representation of the DataFrame.

### Purpose

It demonstrates how individual rows can be accessed using indexing.

> Note: Python uses zero-based indexing, so index `1` refers to the second row.

---

## 6. Display the First 5 Rows

```python
data.head()
```

The `head()` function displays the first five rows of the DataFrame by default.

### Purpose

This is useful for quickly checking the beginning of a dataset.

---

## 7. Display the First 8 Rows

```python
data.head(8)
```

The number of rows can be specified as an argument to `head()`.

Here, the first eight rows are displayed.

### Purpose

This allows a specific number of records from the beginning of the dataset to be inspected.

---

## 8. Display the Last 5 Rows

```python
data.tail()
```

The `tail()` function displays the last five rows of the DataFrame by default.

### Purpose

This helps inspect the end of the dataset and verify that the data has been loaded correctly.

---

# Random Sampling

## 9. Display One Random Record

```python
data.sample()
```

The `sample()` function selects a random record from the DataFrame.

### Purpose

Random sampling can be useful for quickly examining records from different parts of a dataset.

---

## 10. Display 5 Random Records

```python
data.sample(5)
```

Here, five random records are selected from the dataset.

### Purpose

This provides a larger random sample for inspection.

---

## 11. Display 7 Random Records Using a Fixed Random State

```python
data.sample(7, random_state=9)
```

This selects seven random records while using `9` as the random seed.

### Why `random_state` is used

Normally, random sampling can produce different results each time the cell is executed.

By specifying:

```python
random_state=9
```

the same random records can be reproduced each time the notebook is run.

This makes the result **reproducible**.

---

# DataFrame Structure and Information

## 12. Display Column Names

```python
data.columns
```

The `.columns` attribute returns the names of all columns in the DataFrame.

### Purpose

Knowing the column names is important when accessing or analyzing individual variables in the dataset.

---

## 13. Display the Index Range

```python
data.index
```

The `.index` attribute provides information about the DataFrame's row index.

### Purpose

This helps understand how rows are indexed within the DataFrame.

---

## 14. Display Data Types of Each Column

```python
data.dtypes
```

The `.dtypes` attribute displays the data type of every column.

For example, columns may contain types such as:

* `int64` — integer values
* `float64` — decimal values
* `object` — commonly used for text/string data

### Purpose

Understanding data types is important before performing calculations or other data-processing operations.

---

## 15. Display Concise Dataset Information

```python
data.info()
```

The `info()` function provides a concise summary of the DataFrame.

It gives information such as:

* Number of rows
* Number of columns
* Column names
* Non-null values
* Data types
* Memory usage

### Purpose

This gives an overall understanding of the dataset structure in a single command.

---

# Statistical Analysis

## 16. Generate Statistical Summary

```python
data.describe()
```

The `describe()` function generates descriptive statistics for the numerical columns.

It can provide values such as:

* `count`
* `mean`
* `std`
* `min`
* `25%`
* `50%`
* `75%`
* `max`

### Purpose

This provides a quick statistical overview of the numerical data.

For example:

* **count** → number of available values
* **mean** → average value
* **std** → standard deviation
* **min** → minimum value
* **25%** → first quartile
* **50%** → median
* **75%** → third quartile
* **max** → maximum value

---

# Missing Values

## 17. Check for Missing Values

```python
data.isnull().sum()
```

The `isnull()` function identifies missing values in the DataFrame.

The `.sum()` operation then counts the number of missing values in each column.

### Purpose

This helps identify columns that contain incomplete data.

Checking missing values is an important part of the initial data exploration process because missing data may need to be handled before performing further analysis or machine learning tasks.

---

# Accessing Columns

## 18. Access a Single Column

The notebook demonstrates accessing an individual column using:

```python
data['city']
```

The column is printed using:

```python
print(data['city'])
```

When a single column is selected using this syntax, Pandas returns it as a **Series**.

### Purpose

This demonstrates how to access a specific variable/column from a DataFrame.

---

# Summary of Pandas Functions Used

| Function / Attribute         | Purpose                                         |
| ---------------------------- | ----------------------------------------------- |
| `pd.read_csv()`              | Loads data from a CSV file                      |
| `type()`                     | Checks the Python object type                   |
| `.values`                    | Returns DataFrame values as a NumPy array       |
| `.head()`                    | Displays the first 5 rows                       |
| `.head(n)`                   | Displays the first `n` rows                     |
| `.tail()`                    | Displays the last 5 rows                        |
| `.sample()`                  | Selects a random row                            |
| `.sample(n)`                 | Selects `n` random rows                         |
| `.sample(n, random_state=9)` | Selects reproducible random samples             |
| `.columns`                   | Returns column names                            |
| `.index`                     | Returns DataFrame index information             |
| `.dtypes`                    | Shows data types of columns                     |
| `.info()`                    | Provides a concise DataFrame summary            |
| `.describe()`                | Generates descriptive statistics                |
| `.isnull()`                  | Detects missing values                          |
| `.sum()`                     | Counts missing values when used with `isnull()` |
| `data['city']`               | Accesses the `city` column                      |

---

# Learning Outcomes

After completing this notebook, the following Pandas concepts have been demonstrated:

1. How to import Pandas and NumPy.
2. How to load a CSV dataset into a DataFrame.
3. How to inspect the first and last records of a dataset.
4. How to access DataFrame values as a NumPy array.
5. How to select rows using indexing.
6. How to randomly sample records from a DataFrame.
7. How to make random sampling reproducible using `random_state`.
8. How to identify DataFrame columns and indexes.
9. How to inspect the data type of each column.
10. How to obtain a concise overview of a dataset using `info()`.
11. How to generate descriptive statistics using `describe()`.
12. How to identify missing values using `isnull().sum()`.
13. How to access an individual DataFrame column as a Pandas Series.

---

# Conclusion

This notebook provides a basic introduction to **Pandas DataFrame inspection and exploration** using the `matches.csv` dataset.

The operations performed are mainly focused on understanding the dataset's structure, contents, data types, statistical properties, and missing values. These are important preliminary steps in a typical data analysis workflow before moving on to data cleaning, visualization, feature processing, or machine learning.

The notebook therefore demonstrates the fundamental Pandas commands required to **load, inspect, understand, and access data stored in a DataFrame**.
