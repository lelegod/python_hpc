import os
import shutil
from invoke import Context, task

WINDOWS = os.name == "nt"
PYTHON_VERSION = "3.12"
COURSE_NAME = "python_hpc"

SPECIAL_PACKAGES = []

@task
def install(c: Context):
    """Install dependencies. Checks if runs within env, otherwise runs in it."""
    if os.environ.get("CONDA_DEFAULT_ENV") == COURSE_NAME:
        run_cmd = lambda cmd: c.run(cmd, echo=True, pty=not WINDOWS)
    else:
        run_cmd = lambda cmd: c.run(f"conda run -n {COURSE_NAME} {cmd}", echo=True, pty=not WINDOWS)

    run_cmd("pip install -r requirements.txt")
    if SPECIAL_PACKAGES:
        run_cmd(f"pip install {' '.join(SPECIAL_PACKAGES)}")

@task(help={'path': f"Optional path to create the environment in. If not provided, uses default name '{COURSE_NAME}'."})
def create_env(c: Context, path=None):
    """Create a conda environment."""
    if path:
        env_path = f"{path}\\{COURSE_NAME}"
        c.run(f'conda create -p "{env_path}" python={PYTHON_VERSION} pip --no-default-packages --yes', echo=True, pty=not WINDOWS)
        
        pip_exe = os.path.join(env_path, "Scripts", "pip.exe") if WINDOWS else os.path.join(env_path, "bin", "pip")
        
        c.run(f'"{pip_exe}" install -r requirements.txt', echo=True, pty=not WINDOWS)
        if SPECIAL_PACKAGES:
            c.run(f'"{pip_exe}" install {' '.join(SPECIAL_PACKAGES)}', echo=True, pty=not WINDOWS)
    else:
        c.run(f'conda create -n {COURSE_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes', echo=True, pty=not WINDOWS)
        c.run(f"conda run -n {COURSE_NAME} pip install -r requirements.txt", echo=True, pty=not WINDOWS)
        if SPECIAL_PACKAGES:
            c.run(f"conda run -n {COURSE_NAME} pip install {' '.join(SPECIAL_PACKAGES)}", echo=True, pty=not WINDOWS)

@task
def sync(c: Context):
    """Sync the project with the template using cruft."""
    if os.environ.get("CONDA_DEFAULT_ENV") == COURSE_NAME:
        c.run(f"conda run -n {COURSE_NAME} cruft update", echo=True, pty=not WINDOWS)
    else:
        c.run("cruft update", echo=True, pty=not WINDOWS)

@task
def hpc_get(c: Context, week_num: int, file_name: str):
    """Transfer files from HPC to local"""
    hpc_path = f"Documents/{COURSE_NAME}/week{week_num}/{file_name}"
    if not WINDOWS:
        local_path = f"/Users/kyleelyk/Documents/DTU/SEM2/{COURSE_NAME}/week{week_num}/{file_name}"
    c.run(f"scp s252786@login.hpc.dtu.dk:{hpc_path} {local_path}", echo=True, pty=not WINDOWS)

@task
def hpc_post(c: Context, week_num: int, file_name: str):
    """Send files from local to HPC"""
    hpc_path = f"Documents/{COURSE_NAME}/week{week_num}/{file_name}"
    if not WINDOWS:
        local_path = f"/Users/kyleelyk/Documents/DTU/SEM2/{COURSE_NAME}/week{week_num}/{file_name}"
    c.run(f"scp {local_path} s252786@login.hpc.dtu.dk:{hpc_path}", echo=True, pty=not WINDOWS)


@task(help={'folder': 'Folder path to list files from (e.g., week1/question_scripts)'})
def run_file(c: Context, folder: str):
    """List files in a folder and run the selected one."""
    from simple_term_menu import TerminalMenu
    
    if not os.path.isabs(folder):
        folder_path = os.path.join(os.getcwd(), folder)
    else:
        folder_path = folder
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder}' does not exist.")
        return
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
    
    if not files:
        print(f"No Python files found in '{folder}'")
        return
    
    files.sort()
    
    # Interactive menu
    print(f"Select a file to run from {folder}:")
    print("(Use ↑/↓ arrow keys to navigate, Enter to select, q to quit)\n")
    
    terminal_menu = TerminalMenu(files, title="Python Files:")
    menu_index = terminal_menu.show()
    
    if menu_index is None:
        print("Cancelled.")
        return
    
    selected_file = files[menu_index]
    file_path = os.path.join(folder_path, selected_file)
    
    if os.environ.get("CONDA_DEFAULT_ENV") == COURSE_NAME:
        c.run(f"python {file_path}", echo=True, pty=not WINDOWS)
    else:
        c.run(f"conda run -n {COURSE_NAME} python {file_path}", echo=True, pty=not WINDOWS)