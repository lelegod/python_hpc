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
