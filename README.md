# Python HPC

Notes and solutions for the **Python HPC** course.

**Author:** Kyle Nathan Yahya

---

## 🚀 Setup & Installation

### 1. Python Environment
This project manages dependencies using `conda` and `invoke`.

1. **Install Conda**:
   Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
2. **Install invoke**:
   ```
   pip install invoke
   ```
3. **Create the environment**:
   ```bash
   invoke create-env
   ```
   You can also specify the path:
   ```bash
   invoke create-env --path <FOLDER_PATH>
   ```
4. **Install dependencies**:
   ```bash
   invoke install
   ```
   *Note: This automatically installs packages from `requirements.txt` and `requirements_project.txt`.*

### 2. LaTeX Setup (for Notes)

To edit and compile the LaTeX notes (`notes/main.tex`) in VS Code, follow these steps:

#### **Step 1: Install TeX Live**
You need a TeX distribution to compile `.tex` files.
- **Windows**: Download [TeX Live](https://tug.org/texlive/acquire-netinstall.html) (install-tl-windows.exe).
  - *Tip: The full installation is large (~5GB+). You can choose a smaller scheme if you only need basic functionality, but "Full" is safest.*
- **Mac**: Install [MacTeX](https://tug.org/mactex/).
- **Linux**: `sudo apt install texlive-full` (or equivalent).

#### **Step 2: Install VS Code Extension**
Install the **LaTeX Workshop** extension by James Yu.
- Open VS Code Extensions (`Ctrl+Shift+X`).
- Search for `LaTeX Workshop`.
- Click **Install**.

#### **Step 3: Usage**
- Open `notes/main.tex`.
- The extension should automatically build the PDF on save (configured in `.vscode/settings.json`).
- View the PDF by clicking the "View LaTeX PDF" icon in the top right or pressing `Ctrl+Alt+V`.
- **Formatting**: The project is configured to use `latexindent`. You can format the document using `Shift+Alt+F`.

---

## 📂 Project Structure
- `notes/`: LaTeX notes and weekly summaries.
- `tasks.py`: Automation scripts.
- `requirements.txt`: Basic Python dependencies.
- `requirements_project.txt`: Course Specific Python dependencies.