# Week 11 Exercises — HPC Workflows: Job Arrays + Job Dependencies

> [← Index](../index.md) · [Notes](notes.md) · [Syntax](syntax.md) · [Exercises](exercises.md)

## Contents

- [Section 1: Job Script Workflows](#section-1-job-script-workflows)
- [Exercise 1.1 `[AUTOLAB]`](#exercise-11-autolab)
- [Exercise 1.2 `[AUTOLAB]`](#exercise-12-autolab)
- [Exercise 1.3 `[AUTOLAB]`](#exercise-13-autolab)
- [Exercise 1.4 `[PRACTICE]`](#exercise-14-practice)
- [Exercise 1.5 `[AUTOLAB]`](#exercise-15-autolab)
- [Section 2: Face Colors (CelebA Dataset)](#section-2-face-colors-celeba-dataset)
- [Exercise 2.1 `[PRACTICE]`](#exercise-21-practice)
- [Exercise 2.2 `[PRACTICE]`](#exercise-22-practice)
- [Exercise 2.3 `[PRACTICE]`](#exercise-23-practice)
- [Exercise 2.4 `[PRACTICE]`](#exercise-24-practice)
- [Exercise 2.5 `[PRACTICE]`](#exercise-25-practice)

---

---

## Section 1: Job Script Workflows

> For all exercises in this section, your job scripts must submit to the `hpc` queue, use 1 core on 1 node and specify a job name, output file and error file.

---

## Exercise 1.1 `[AUTOLAB]`

Make a job script that creates a job array with job indices running from 1 to 10.

> **Solution:**
>
> Use `#BSUB -J name[1-10]` to define a job array with indices 1 through 10. Each element gets its own index available via `$LSB_JOBINDEX`.
>
> ```bash
> #!/bin/bash
> #BSUB -J array[1-10]
> #BSUB -q hpc
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -W 00:10
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -o array_%J_%I.out
> #BSUB -e array_%J_%I.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> echo "Running array element $LSB_JOBINDEX"
> ```

---

## Exercise 1.2 `[AUTOLAB]`

Make a job script that creates a job array with job indices 2, 29, 71, 73 and 127.

> **Solution:**
>
> LSF supports comma-separated index lists in the `#BSUB -J` directive: `name[2,29,71,73,127]`. Only those specific indices are submitted as array elements.
>
> ```bash
> #!/bin/bash
> #BSUB -J array[2,29,71,73,127]
> #BSUB -q hpc
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -W 00:10
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -o array_%J_%I.out
> #BSUB -e array_%J_%I.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> echo "Running array element $LSB_JOBINDEX"
> ```

---

## Exercise 1.3 `[AUTOLAB]`

Make a job script that waits until another job with id 1234567 exits successfully.

> **Solution:**
>
> Use `#BSUB -w "done(<job_id>)"` to declare a dependency. `done()` means the referenced job must exit with status 0 (success) before this job is allowed to start.
>
> ```bash
> #!/bin/bash
> #BSUB -J dependent_job
> #BSUB -q hpc
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -W 00:10
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -w "done(1234567)"
> #BSUB -o dependent_%J.out
> #BSUB -e dependent_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> echo "Running after job 1234567 finished successfully"
> ```

---

## Exercise 1.4 `[PRACTICE]`

Assume we have the following output from `bstat` for a job array which is already submitted:

```
$ bstat
JOBID      USER    QUEUE      JOB_NAME   NALLOC STAT  START_TIME      ELAPSED
21241475   patmjen hpc        array[1]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[2]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[3]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[4]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[5]        1 RUN   Apr  9 13:40    0:00:09
```

Make a job script that creates a job array, where each job in the array waits for the corresponding job in the already submitted array to finish successfully.

> **Solution:**
>
> Use `#BSUB -w "done(array[*])"` — the wildcard `[*]` means each element of the new array waits for the element with the **same index** in the referenced array (`array`) to finish successfully. The job name in the dependency refers to the already-submitted job array by its `JOB_NAME` (here `array`).
>
> ```bash
> #!/bin/bash
> #BSUB -J jobname[1-5]
> #BSUB -q hpc
> #BSUB -W 5
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -w "done(array[*])"
> #BSUB -o jobname_%J.out
> #BSUB -e jobname_%J.err
> ```

---

## Exercise 1.5 `[AUTOLAB]`

Again, assume we have the following output from `bstat` for a job array which is already submitted:

```
$ bstat
JOBID      USER    QUEUE      JOB_NAME   NALLOC STAT  START_TIME      ELAPSED
21241475   patmjen hpc        array[1]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[2]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[3]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[4]        1 RUN   Apr  9 13:40    0:00:09
21241475   patmjen hpc        array[5]        1 RUN   Apr  9 13:40    0:00:09
```

Make a job script that waits until all elements of the job array have exited for any reason.

> **Solution:**
>
> Use `#BSUB -w "ended(<job_id>)"` to wait for the entire job array (referenced by its numeric JOBID `21241475`) to exit for **any** reason — including failure. `ended()` does not require success, unlike `done()`.
>
> ```bash
> #!/bin/bash
> #BSUB -J dependent_job
> #BSUB -q hpc
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -W 00:10
> #BSUB -R "rusage[mem=1GB]"
> #BSUB -w "ended(21241475)"
> #BSUB -o dependent_%J.out
> #BSUB -e dependent_%J.err
>
> source /dtu/projects/02613_2025/conda/conda_init.sh
> conda activate 02613
>
> echo "Running after ALL elements of array 21241475 have exited (any reason)"
> ```

---

## Section 2: Face Colors (CelebA Dataset)

> The CelebA dataset of celebrity images is stored in subfolders at `/dtu/projects/02613_2024/data/celeba/images`. Each subfolder contains around 1000 images. The goal is to create a hue histogram over the entire dataset by processing each subfolder as a separate job (job array) and then aggregating results in a final dependent job.
>
> The following helper function computes a 64-bin hue histogram from an RGB image stored as a `(H, W, 3)` NumPy array:
>
> ```python
> import numpy as np
> from PIL import Image
>
> def huehist(image):
>     bins = np.linspace(0, 255, 64 + 1)
>     hsv_image = np.asarray(Image.fromarray(image).convert('HSV'))
>     hue_values = hsv_image[:, :, 0].reshape(-1)
>     hue_hist = np.histogram(hue_values, bins)[0]
>     return hue_hist
> ```

---

## Exercise 2.1 `[PRACTICE]`

Create a Python program which takes an index `i` as a command line argument. For the `i`th folder in `/dtu/projects/02613_2024/data/celeba/images`, the program must then:

1. Load every image in the folder. Hint: use `glob.glob` or `os.listdir` to get the files in a folder.
2. Compute the hue histogram for each image using the provided function.
3. Sum the histograms to get a combined histogram for the folder.
4. Save the summed histogram to a NumPy array named `subhist_<i>.npy`, where `<i>` is replaced with the value of the provided index `i`.

> **Solution:**
>
> Note that job array indices are 1-based, so `idx = int(sys.argv[1]) - 1` converts to a 0-based folder index for `os.listdir`.
>
> ```python
> import os
> from os.path import join
> import sys
> import numpy as np
> from PIL import Image
>
> def huehist(image):
>     bins = np.linspace(0, 255, 64 + 1)
>     hsv_image = np.asarray(Image.fromarray(image).convert('HSV'))
>     hue_values = hsv_image[:, :, 0].reshape(-1)
>     hue_hist = np.histogram(hue_values, bins)[0]
>     return hue_hist
>
> if __name__ == '__main__':
>     idx = int(sys.argv[1]) - 1  # Job arrays must be 1-indexed
>     base_path = '/dtu/projects/02613_2024/data/celeba/images'
>     folders = sorted(os.listdir(base_path))
>     images = sorted(os.listdir(join(base_path, folders[idx])))
>
>     hist = 0
>     for im in images:
>         image = Image.open(join(base_path, folders[idx], im))
>         image = np.array(image)
>         hist += huehist(image)
>
>     np.save(f"subhist_{idx}.npy", hist)
> ```

---

## Exercise 2.2 `[PRACTICE]`

Create a Python program that loads the saved histogram arrays, sums them to a combined histogram for the entire dataset, plots the histogram using `plt.bar` and finally saves the histogram plot as an image.

> **Solution:**
>
> Use `glob` to collect all partial histogram `.npy` files, accumulate them with `+=`, then plot with `plt.bar`.
>
> ```python
> from glob import glob
> import numpy as np
> import matplotlib.pyplot as plt
>
> if __name__ == '__main__':
>     histfiles = glob('subhist_*.npy')
>     hist = 0
>     for hf in histfiles:
>         hist += np.load(hf)
>     plt.bar(np.linspace(0, 255, 64), hist, width=4)
>     plt.savefig('histogram.png')
> ```

---

## Exercise 2.3 `[PRACTICE]`

Make a job script that submits a job array where each job computes the hue histogram for the `i`th folder using the first Python program.

> **Solution:**
>
> The dataset has 203 subfolders, so the array runs `[1-203]`. Each element passes `$LSB_JOBINDEX` directly to the Python script as the 1-based folder index.
>
> ```bash
> #!/bin/bash
> #BSUB -J subhist[1-203]
> #BSUB -q hpc
> #BSUB -W 15
> #BSUB -n 1
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -o batch_output/subhist_%I_%J.out
> #BSUB -e batch_output/subhist_%I_%J.err
>
> source /dtu/projects/02613_2024/conda/conda_init.sh
> conda activate 02613
>
> python huedir.py $LSB_JOBINDEX
> ```

---

## Exercise 2.4 `[PRACTICE]`

Make a job script that waits for all jobs in the previous job array to finish successfully and then runs the second Python program to compute and plot the final hue histogram.

> **Solution:**
>
> Use `#BSUB -w done(subhist)` to declare a dependency on the job named `subhist`. LSF will hold this job pending until all elements of the `subhist` array have completed successfully.
>
> ```bash
> #!/bin/bash
> #BSUB -J plothist
> #BSUB -q hpc
> #BSUB -W 5
> #BSUB -n 1
> #BSUB -w done(subhist)
> #BSUB -R "span[hosts=1]"
> #BSUB -R "rusage[mem=512MB]"
> #BSUB -o batch_output/plothist_%J.out
> #BSUB -e batch_output/plothist_%J.err
>
> source /dtu/projects/02613_2024/conda/conda_init.sh
> conda activate 02613
>
> python plothuehist.py
> ```

---

## Exercise 2.5 `[PRACTICE]`

Submit your jobs and compute the histogram.

> **Solution:**
>
> Submit the two scripts in sequence from the cluster login node:
>
> ```bash
> bsub < subhist_array.sh   # submits the job array (subhist[1-203])
> bsub < plothist.sh        # submits the dependent aggregation job
> ```
>
> After submission, `bstat` will show all 203 `subhist` elements in RUN/PEND state and `plothist` in PEND state. Once all `subhist` jobs complete successfully, `plothist` starts automatically and writes `histogram.png`.
>
> The resulting histogram shows red/orange hues dominating — consistent with the fact that skin tones fall in the red-orange hue range. The most used colors are the red hues, which makes sense since skin colors are from red and orange hues.
