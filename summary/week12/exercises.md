# Week 12 Exercises — Course Project

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Background](#background)
- [Exercise 1 `[PRACTICE]`](#exercise-1-practice)
- [Exercise 2 `[PRACTICE]`](#exercise-2-practice)
- [Exercise 3 `[PRACTICE]`](#exercise-3-practice)
- [Exercise 4 `[PRACTICE]`](#exercise-4-practice)
- [Exercise 5 `[PRACTICE]`](#exercise-5-practice)
  - [5a `[PRACTICE]`](#5a-practice)
  - [5b `[PRACTICE]`](#5b-practice)
  - [5c `[PRACTICE]`](#5c-practice)
  - [5d `[PRACTICE]`](#5d-practice)
- [Exercise 6 `[PRACTICE]`](#exercise-6-practice)
  - [6a `[PRACTICE]`](#6a-practice)
  - [6b `[PRACTICE]`](#6b-practice)
- [Exercise 7 `[PRACTICE]`](#exercise-7-practice)
  - [7a `[PRACTICE]`](#7a-practice)
  - [7b `[PRACTICE]`](#7b-practice)
  - [7c `[PRACTICE]`](#7c-practice)
- [Exercise 8 `[PRACTICE]`](#exercise-8-practice)
  - [8a `[PRACTICE]`](#8a-practice)
  - [8b `[PRACTICE]`](#8b-practice)
  - [8c `[PRACTICE]`](#8c-practice)
- [Exercise 9 `[PRACTICE]`](#exercise-9-practice)
  - [9a `[PRACTICE]`](#9a-practice)
  - [9b `[PRACTICE]`](#9b-practice)
  - [9c `[PRACTICE]`](#9c-practice)
- [Exercise 10 `[PRACTICE]`](#exercise-10-practice)
- [Exercise 11 `[PRACTICE]` (Optional)](#exercise-11-practice-optional)
- [Exercise 12 `[PRACTICE]`](#exercise-12-practice)
  - [12a `[PRACTICE]`](#12a-practice)
  - [12b `[PRACTICE]`](#12b-practice)
  - [12c `[PRACTICE]`](#12c-practice)
  - [12d `[PRACTICE]`](#12d-practice)
  - [12e `[PRACTICE]`](#12e-practice)

---

## Background

This project simulates the "Wall Heating!" concept: heating elements installed in inside walls of buildings from the Modified Swiss Dwellings dataset (4571 floorplans). Inside walls are fixed at 25°C, load bearing walls at 5°C. The goal is to find the steady-state heat distribution inside rooms by solving Laplace's equation using the Jacobi method on a 514×514 grid.

The reference script `simulate.py` loads building data from `/dtu/projects/02613_2025/data/modified_swiss_dwellings/`, runs Jacobi iterations (max 20,000, tolerance 1e-4), computes summary statistics (mean temp, std temp, % above 18°C, % below 15°C), and prints CSV output. Your goal is to optimize it.

---

## Exercise 1 `[PRACTICE]`

Familiarize yourself with the data. Load and visualize the input data for a few floorplans using a separate Python script, Jupyter notebook or your preferred tool.

---

## Exercise 2 `[PRACTICE]`

Familiarize yourself with the provided script. Run and time the reference implementation for a small subset of floorplans (e.g., 10 - 20). How long do you estimate it would take to process all the floorplans? Perform the timing as a batch job so you get reliable results.

---

## Exercise 3 `[PRACTICE]`

Visualize the simulation results for a few floorplans.

---

## Exercise 4 `[PRACTICE]`

Profile the reference `jacobi` function using kernprof. Explain the different parts of the function and how much time each part takes.

---

## Exercise 5 `[PRACTICE]`

Make a new Python program where you parallelize the computations over the floorplans. Use static scheduling such that each worker is assigned the same amount of floorplans to process. You should use no more than 100 floorplans for your timing experiments. Again, use a batch job to ensure consistent results.

### 5a `[PRACTICE]`

Measure the speed-up as more workers are added. Plot your speed-ups.

### 5b `[PRACTICE]`

Estimate your parallel fraction according to Amdahl's law. How much (roughly) is parallelized?

### 5c `[PRACTICE]`

What is your theoretical maximum speed-up according to Amdahl's law? How much of that did you achieve? How many cores did that take?

### 5d `[PRACTICE]`

How long would you estimate it would take to process all floorplans using your fastest parallel solution?

---

## Exercise 6 `[PRACTICE]`

The amount of iterations needed to reach convergence will vary from floorplan to floorplan. Re-do your parallelization experiment using dynamic scheduling.

### 6a `[PRACTICE]`

Did it get faster? By how much?

### 6b `[PRACTICE]`

Did the speed-up improve or worsen?

---

## Exercise 7 `[PRACTICE]`

Implement another solution where you rewrite the `jacobi` function using Numba JIT _on the CPU_.

### 7a `[PRACTICE]`

Run and time the new solution for a small subset of floorplans. How does the performance compare to the reference?

### 7b `[PRACTICE]`

Explain your function. How did you ensure your access pattern works well with the CPU cache?

### 7c `[PRACTICE]`

How long would it now take to process all floorplans?

---

## Exercise 8 `[PRACTICE]`

Implement another solution writing a custom CUDA kernel with Numba. To synchronize threads between each iteration, the kernel should only perform _a single iteration_ of the Jacobi solver. Skip the early stopping criteria and just run for a fixed amount of iterations. Write a helper function which takes the same inputs as the reference implementation (except for the `atol` input which is not needed) and then calls your kernel repeatedly to perform the implementations.

### 8a `[PRACTICE]`

Briefly describe your new solution. How did you structure your kernel and helper function?

### 8b `[PRACTICE]`

Run and time the new solution for a small subset of floorplans. How does the performance compare to the reference?

### 8c `[PRACTICE]`

How long would it now take to process all floorplans?

---

## Exercise 9 `[PRACTICE]`

Adapt the reference solution to run on the GPU using CuPy.

### 9a `[PRACTICE]`

Run and time the new solution for a small subset of floorplans. How does the performance compare to the reference?

### 9b `[PRACTICE]`

How long would it now take to process all floorplans?

### 9c `[PRACTICE]`

Was anything surprising about the performance?

---

## Exercise 10 `[PRACTICE]`

Profile the CuPy solution using the nsys profiler. What is the main issue regarding performance? (Hint: see exercises from week 10) Try to fix it.

---

## Exercise 11 `[PRACTICE]` (Optional)

**(Optional)** Improve the performance of one or more of your solutions further. For example, parallelize your CPU JIT solution. Or use job arrays to parallelize a solution over multiple jobs. How fast can you get?

---

## Exercise 12 `[PRACTICE]`

Process all floorplans using one of your implementations (ideally a fast one) and answer the below questions.
Hint: use Pandas to process the CSV results generated by the script.

### 12a `[PRACTICE]`

What is the distribution of the mean temperatures? Show your results as histograms.

### 12b `[PRACTICE]`

What is the average mean temperature of the buildings?

### 12c `[PRACTICE]`

What is the average temperature standard deviation?

### 12d `[PRACTICE]`

How many buildings had at least 50% of their area above 18°C?

### 12e `[PRACTICE]`

How many buildings had at least 50% of their area below 15°C?
