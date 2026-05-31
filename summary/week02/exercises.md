# Week 2 Exercises — Python Basics + NumPy (Timing, Profiling)

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Basic Python](#section-1-basic-python)
- [Exercise 1 `[AUTOLAB]`](#exercise-1-autolab)
- [Exercise 2 `[AUTOLAB]`](#exercise-2-autolab)
- [Exercise 3 `[AUTOLAB]`](#exercise-3-autolab)
- [Exercise 4 `[AUTOLAB]`](#exercise-4-autolab)
- [Exercise 5 `[AUTOLAB]`](#exercise-5-autolab)
- [Exercise 6 `[AUTOLAB]`](#exercise-6-autolab)
- [Exercise 7 `[AUTOLAB]`](#exercise-7-autolab)
- [Exercise 8 `[AUTOLAB]`](#exercise-8-autolab)
- [Section 2: Basic NumPy](#section-2-basic-numpy)
- [Exercise 9 `[AUTOLAB]`](#exercise-9-autolab)
- [Exercise 10 `[AUTOLAB]`](#exercise-10-autolab)
- [Exercise 11 `[AUTOLAB]`](#exercise-11-autolab)
- [Exercise 12 `[AUTOLAB]`](#exercise-12-autolab)
- [Exercise 13 `[AUTOLAB]`](#exercise-13-autolab)
- [Exercise 14 `[AUTOLAB]`](#exercise-14-autolab)

---

> All exercises this week are submitted via Autolab. The student `.py` files serve as the solutions.

---

## Section 1: Basic Python

---

## Exercise 1 `[AUTOLAB]`

Write a Python function called `listsum` that sums numbers in a list.

_Input:_ A Python list of numbers.
_Output:_ A single number which is the sum of the numbers in the input list.
_Example:_ The input [1, 2, 3, 4] should return 10.

> **Solution:**
>
> ```python
> def listsum(arr: list[int | float]) -> int | float:
>     return sum(arr)
>
> if __name__ == "__main__":
>     print(f"Output: {listsum([1, 2, 3.3])}")
> ```

---

## Exercise 2 `[AUTOLAB]`

Write a Python function called `deduplicate` that removes duplicates from a list. Hint: you can use the Python builtin `set`.

_Input:_ A Python list which may contain duplicate elements.
_Output:_ A Python list containing the unique elements of the input list.
_Example:_ The input [1, 2, 3, 3, 2, 2, 4] should return [1, 2, 3, 4].

> **Solution:**
>
> ```python
> def deduplicate(arr):
>     return list(set(arr))
>
> if __name__ == "__main__":
>     print(f"Output: {deduplicate([1, 1, 2, 3.3])}")
> ```

---

## Exercise 3 `[AUTOLAB]`

Write a Python function called `sorttuples` that sorts a list of tuples according to their last element. Hint: you may use the builtin function `sorted`. Look at the `key` parameter.

_Input:_ A Python list of tuples.
_Output:_ The input list with elements sorted according to the last element in each tuple.
_Example:_ The input [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)] should return [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)].

> **Solution:**
>
> ```python
> def sorttuples(arr):
>     return sorted(arr, key=lambda t: t[1])
>
> if __name__ == "__main__":
>     print(f"Output: {sorttuples([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)])}")
> ```

---

## Exercise 4 `[AUTOLAB]`

Write a Python function `squarecubes` which receives a Python list of numbers and returns two lists containing, respectively, the squares and cubes of each number.

_Input:_ A Python list of numbers.
_Output:_ A tuple of two Python lists. The first list contains the squares of the numbers in the input list. The second list contains the cubes.
_Example:_ The input [1, 2, 3, 4] should return ([1, 4, 9, 16], [1, 8, 27, 64]).

> **Solution:**
>
> ```python
> def squarecubes(arr):
>     return (list(map(lambda n: n**2, arr)), list(map(lambda n: n**3, arr)))
>
> if __name__ == "__main__":
>     print(f"Output: {squarecubes([1, 2, 3, 4])}")
> ```

---

## Exercise 5 `[AUTOLAB]`

Write a Python _program_ that will receive any number of numerical grades as command line arguments. It must then compute the mean and print it back to the user followed by "Pass" if the mean is at least 5 and "Fail" otherwise. Hint: you can use `sys.argv` to access the command line arguments.

_Input:_ Any number of grades given as command line arguments. Each grade will be a number.
_Output:_ The mean grade, followed by a space, followed by `Pass` if the mean is at least 5, otherwise `Fail`.
_Example:_ For the input 0, 2, 4 the program should print the string `4.0 Fail`. For the input [4, 7, 10, 12] the program should print the string `8.25 Pass`.

> **Solution:**
>
> ```python
> import sys
>
> def grades_check():
>     grades = [int(g) for g in sys.argv[1:]]
>     average = sum(grades) / len(grades)
>     return f"{average} {'Pass' if average >= 5 else 'Fail'}"
>
> if __name__ == "__main__":
>     print(f"{grades_check()}")
> ```

---

## Exercise 6 `[AUTOLAB]`

Write a Python _program_ that will receive any amount of numbers as command line arguments. It must then remove all odd numbers and print out the list of even numbers. Hint: you can use the builtin function `filter` or a list comprehension.

_Input:_ Any amount of numbers as command line arguments.
_Output:_ All even numbers in the input.
_Example:_ For the input 0, 1, 4, 2, 3, -2 the program should print the string `[0, 4, 2, -2]`.

> **Solution:**
>
> ```python
> import sys
>
> def even_only():
>     nums = [int(g) for g in sys.argv[1:]]
>     return list(filter(lambda n: n % 2 == 0, nums))
>
> if __name__ == "__main__":
>     print(f"{even_only()}")
> ```

---

## Exercise 7 `[AUTOLAB]`

Write a Python class `Student`, that has the student name and a list of courses the student is attending as attributes `name` and `courses`. These should be given as arguments to the class constructor. The class should also have a method `attends` that receives a course name and returns `True` if the student is attending that course and `False` if not.

_Constructor input:_ The first argument is a string with the student's name. The second is a list of course names, where each course name is a string.
_`attends` input:_ A string with a course name.
_`attends` output:_ `True` if the student attends that course (i.e., it is in the list of courses the student attends). `False` if not.
_Example:_ If we create a student as `s = Student('X', ['01005', '02613'])`, we expect that `s.attends('01005')` and `s.attends('02613')` returns `True` and `s.attends('02510')` returns `False`.

> **Solution:**
>
> ```python
> class Student:
>     def __init__(self, name, courses):
>         self.name = name
>         self.courses = courses
>     def attends(self, course):
>         return True if course in self.courses else False
> ```

---

## Exercise 8 `[AUTOLAB]`

Write a Python function `coursestudents` that takes a list of `Student` objects as well as a course name. It must return a new list containing the names of the students that attend that course.

_Input:_ The first argument is a Python list of `Student`s. The second is a string specifying a course name.
_Output:_ A list containing the names of the students that attend the given course.
_Example:_ Given the list of `Student`s:
```python
students = [Student('A', ['01005']), Student('B', ['02613']), Student('C', ['01005', '02613'])]
```
we expect that `coursestudents(students, '02613')` returns `['B', 'C']`.

Note: Autolab expects the attributes in the Student class to be called `name` and `courses`.

> **Solution:**
>
> ```python
> def coursestudents(students, course):
>     return [student.name for student in students if course in student.courses]
> ```

---

## Section 2: Basic NumPy

---

## Exercise 9 `[AUTOLAB]`

Write a Python function `magnitude`, which takes an n-dimensional vector stored as a NumPy array as input and returns its magnitude, i.e., norm.

_Input:_ An n-dimensional vector stored as a NumPy array.
_Output:_ A number giving the magnitude of the input vector.
_Example:_ For the input [1, 1, 3, 3, 4] we expect the output 6.

> **Solution:**
>
> ```python
> import numpy as np
>
> def magnitude(v):
>     return np.linalg.norm(v)
> ```

---

## Exercise 10 `[AUTOLAB]`

Write a Python _program_ that will receive any amount of numbers as command line arguments. These numbers are the components of an n-dimensional vector. The program must then print the magnitude of that vector.

_Input:_ An n-dimensional vector where each component is given as a command line argument.
_Output:_ A number giving the magnitude of the input vector.
_Example:_ For the input 1, 1, 3, 3, 4 we expect the output 6.

> **Solution:**
>
> ```python
> import numpy as np
> import sys
>
> def magnitude():
>     v = np.array(sys.argv[1:]).astype(float)
>     return np.linalg.norm(v)
>
> if __name__ == "__main__":
>     print(f"{magnitude()}")
> ```

---

## Exercise 11 `[AUTOLAB]`

Write a Python _program_ that will receive any amount of numbers as command line arguments. These numbers are the diagonal of an n×n matrix. The program must then save this matrix as a `.npy` file. Hint: use the `numpy.save` function.

_Input:_ The diagonal of an n×n matrix where each component is given as a command line argument.
_Output:_ A new `.npy` file containing a matrix with the input numbers as the diagonal.
_Example:_ For the input `1 8 4 5`, we expect the program to create a `.npy` file with the matrix:

```
[[1, 0, 0, 0],
 [0, 8, 0, 0],
 [0, 0, 4, 0],
 [0, 0, 0, 5]]
```

> **Solution:**
>
> ```python
> import numpy as np
> import sys
>
> def save_diag():
>     e = np.array(sys.argv[1:]).astype(float)
>     np.save("saved.npy", np.diag(e))
>     pass
>
> if __name__ == "__main__":
>     save_diag()
> ```

---

## Exercise 12 `[AUTOLAB]`

Write a Python _program_ that receives the path to a `.npy` file as a command line argument. This file will contain an n×m matrix. The program must then save two new files:

1. `cols.npy` containing an m-dimensional vector with the means of each column
2. `rows.npy` containing an n-dimensional vector with the means of each row

Hint: use the `numpy.load` function.

_Input:_ The path to an `.npy` file containing an n×m matrix.
_Output:_ Two new files `cols.npy` and `rows.npy` containing, respectively, the column and row means.
_Example:_ For a file containing the matrix:

```
[[1, 3, 2, 0],
 [5, 7, 3, 9],
 [9, 2, 4, 6]]
```

we expect `cols.npy` to contain the vector [5, 4, 3, 5] and `rows.npy` to contain the vector [1.5, 6, 5.25].

> **Solution:**
>
> ```python
> import numpy as np
> import sys
>
> def save_mean():
>     M = np.load(sys.argv[1])
>     np.save("cols.npy", np.mean(M, axis=0))
>     np.save("rows.npy", np.mean(M, axis=1))
>     pass
>
> if __name__ == "__main__":
>     save_mean()
> ```

---

## Exercise 13 `[AUTOLAB]`

Write a Python program that receives the path to a `.npy` file as well as a strictly positive integer p as a command line argument. The `.npy` file will contain a matrix A, and the program must create a new `.npy` file containing the result of multiplying A with itself p times, i.e., A^(p+1). The program must _also_ print the time (in seconds) that it took to perform these multiplications. Hint: you can use `perf_counter` from the `time` module.

_Input:_ The path to an `.npy` file containing a matrix A and a strictly positive integer p given as command line arguments.
_Output:_ A `.npy` file containing A^(p+1). Also, the time (in seconds) it took to compute A^(p+1) must be printed.
_Example:_ For a file containing the matrix:

```
[[1, 3],
 [5, 7]]
```

and for p=2, we expect the output to contain the matrix A^(2+1):

```
[[136, 216],
 [360, 568]]
```

> **Solution:**
>
> ```python
> import numpy as np
> import sys
> from time import perf_counter
>
> def save_mean():
>     A, p = np.load(sys.argv[1]), int(sys.argv[2])
>     start = perf_counter()
>     np.save("saved.npy", np.linalg.matrix_power(A, p + 1))
>     end = perf_counter()
>     print(f"{start - end}")
>     pass
>
> if __name__ == "__main__":
>     save_mean()
> ```
>
> **Note:** There is a sign bug in the solution — `start - end` should be `end - start`. The corrected version is in `numpy6.py` which fixes this.

---

## Exercise 14 `[AUTOLAB]`

Run the matrix power program (Exercise 13) as a batch job. Assume the input file always has the path `./input.npy` and p is always 10. Submit the job to the `hpc` queue and request 1 core. Remember to show good etiquette and also specify a job name, expected run time, memory usage, and files for stdout and stderr. The Autolab handin must be a zip-file containing both your submit script and your Python script. Hint: remember to initialize the course conda environment in the job script.

> **Solution:**
>
> **`numpy6.py`** — Python script (hardcoded for batch use):
>
> ```python
> import numpy as np
> import sys
> from time import perf_counter
>
> def save_mean():
>     A, p = np.load("./input.npy"), 10
>     start = perf_counter()
>     np.save("saved.npy", np.linalg.matrix_power(A, p + 1))
>     end = perf_counter()
>     print(f"{end - start}")
>     pass
>
> if __name__ == "__main__":
>     save_mean()
> ```
>
> **`submit.sh`** — LSF batch job script:
>
> ```bash
> #!/bin/bash
> #BSUB -J python
> #BSUB -q hpc
> #BSUB -W 1
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "select[model==XeonGold6226R]"
> #BSUB -o python%J.out
> #BSUB -e python_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> python numpy6.py
> ```
