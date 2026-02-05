from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model

# Insert your dockerhub image below (e.g. "docker.io/<username>/<image_name>:<tag>")
DOCKER_IMAGE = "<docker_image>"


@dsl.component(base_image="python:3.14-slim")
def download_repo(
        github_repo_url: str,
        project_files: Output[Model],
        data: Output[Dataset],
        branch: str = "main",
) -> None:
    """Download specific scripts and data from a GitHub repository.

    :param github_repo_url: URL of the GitHub repository to clone.
    :param project_files: Output path for project scripts.
    :param data: Output path for data folder.
    :param branch: Branch name to pull from (defaults to 'main').
    """
    import shutil
    from pathlib import Path
    import subprocess

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            github_repo_url,
            str(repo_dir),
        ],
        check=True,
    )
    print(f"Cloned repo {github_repo_url} (branch: {branch}).")

    # Copy everything from src/ folder to project_files
    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    src_folder = repo_dir / "src"

    if src_folder.exists():
        for item in src_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, proj_path / item.name)
                print(f"Copied src/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, proj_path / item.name, dirs_exist_ok=True)
                print(f"Copied src/{item.name}/ directory")
    else:
        print("Warning: src/ folder not found in repo")

    # Verify all required files exist
    required_files = [
        "fairness_bias_analysis.py",
        "explainer.py",
        "fairness_bias_visualization.py",
        "explainer_visualization.py",
        "STAR_model.py",
        "utils/data_helpers.py",
        "utils/generic_utils.py",
        "utils/explainer_helpers.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = proj_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"ERROR: Required file missing: {file_path}")
        else:
            print(f"✓ Verified: {file_path}")

    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    # Copy everything inside data folder
    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"

    if src_data_path.exists():
        for item in src_data_path.iterdir():
            if item.is_file():
                shutil.copy2(item, data_path / item.name)
                print(f"Copied data/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, data_path / item.name, dirs_exist_ok=True)
                print(f"Copied data/{item.name}/ directory")
    else:
        print("Warning: data folder not found in repo")


@dsl.container_component
def star_predictions(
        input_data: Input[Dataset],
        predictions: Output[Dataset],
):
    """Run STAR model predictions on patient JSON files.

    :param input_data: Input path containing patient JSON files.
    :param predictions: Output path for predictions CSV.
    """
    command_str = f"""
        set -e
        export AEONICS_JAVA_OPTIONS="-Xmx1g"
        export AEONICS_LICENSE_STORE_PATH="/opt/aeonics/aeonics.license"
        export AEONICS_LICENSE_STORE_PASS="secret"
        export AEONICS_ACCEPT_UNSIGNED_MODULES="true"
        export AEONICS_LOG_LEVEL="1000"
        export REALM_INPUT_DIR="{input_data.path}"
        export REALM_OUTPUT_DIR="{predictions.path}"
        mkdir -p "{predictions.path}"
        cd /opt/aeonics
        /opt/aeonics/jre/bin/java -Xmx1g -jar aeonics.jar
        [ -f "{predictions.path}/results.csv" ] || exit 1
    """

    return dsl.ContainerSpec(
        image=DOCKER_IMAGE,
        command=["sh", "-c"],
        args=[command_str]
    )


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["pandas==3.0.0", "tqdm==4.67.2"],
)
def fairness_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        predictions: Input[Dataset],
        fairness_results: Output[Dataset],
) -> None:
    """Run fairness and bias analysis using predictions from STAR model.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing patient JSON files.
    :param predictions: Input path containing STAR predictions CSV.
    :param fairness_results: Output path for fairness analysis results (JSON).
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    predictions_path = Path(predictions.path)
    results_path = Path(fairness_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "fairness_bias_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Fairness analyzer script not found at {script}")

    predictions_csv = predictions_path / "results.csv"
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")

    print(f"Running fairness analysis with {script}")
    print(f"Data path: {data_path}")
    print(f"Predictions: {predictions_csv}")

    cmd = [
        "python",
        str(script),
        "--data_path",
        str(data_path),
        "--predictions_path",
        str(predictions_csv),
        "--output",
        str(results_path / "fairness_analysis.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Fairness analysis finished. Results saved to {results_path}")


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["matplotlib==3.10.7"],
)
def fairness_visualization(
        project_files: Input[Model],
        fairness_results: Input[Dataset],
        fairness_plots: Output[Dataset],
) -> None:
    """Create visualizations for fairness and bias analysis results.

    :param project_files: Input path containing project scripts.
    :param fairness_results: Input path containing fairness_analysis.json.
    :param fairness_plots: Output path for visualization PNG files.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    results_path = Path(fairness_results.path)
    plots_path = Path(fairness_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "fairness_bias_visualization.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Fairness Bias visualization script not found at {script}"
        )

    analysis_results_file = results_path / "fairness_analysis.json"
    if not analysis_results_file.exists():
        raise FileNotFoundError(
            f"Fairness Bias analysis results not found at {analysis_results_file}"
        )

    print(f"Running fairness bias visualization with {script}")

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_results_file),
        "--output",
        str(plots_path),
    ]
    subprocess.run(cmd, check=True)

    print(f"Fairness Bias visualization completed. Plots saved to {plots_path}")


@dsl.container_component
def explainer_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        explainer_results: Output[Dataset],
        sensitivity: float,
):
    """Run explainer analysis inside STAR Docker container.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing patient JSON files.
    :param explainer_results: Output path for explainer results.
    :param sensitivity: Sensitivity parameter for the explainer script.
    """
    command_str = f"""
        set -e
        apt-get update
        apt-get install -y python3 python3-dev wget curl
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages
        python3 -m pip install --break-system-packages pandas==3.0.0 tqdm==4.67.2 numpy==2.4.2
        cd {project_files.path}

        python3 explainer.py \
            --data_path {data.path} \
            --output {explainer_results.path} \
            --sensitivity {sensitivity} \
            --docker_image {DOCKER_IMAGE} \
            --in_docker True
        ls -la {explainer_results.path}
    """

    return dsl.ContainerSpec(
        image=DOCKER_IMAGE,
        command=["sh", "-c"],
        args=[command_str]
    )


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["tqdm==4.67.1", "pandas==2.3.3", "matplotlib==3.10.7"],
)
def explainer_visualization(
        project_files: Input[Model],
        explainer_results: Input[Dataset],
        explainer_plots: Output[Dataset],
        sensitivity: float,
) -> None:
    """Create visualizations for explainer analysis results.

    :param project_files: Input path containing project scripts.
    :param explainer_results: Input path containing explainer results.
    :param explainer_plots: Output path for visualization PNG files.
    :param sensitivity: Sensitivity parameter to determine which method was used.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    explainer_results_path = Path(explainer_results.path)
    plots_path = Path(explainer_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "explainer_visualization.py"
    if not script.exists():
        raise FileNotFoundError(f"Explainer visualization script not found at {script}")

    # Determine which analysis file to use based on sensitivity
    if sensitivity < 0.5:
        # ablation
        analysis_file = explainer_results_path / "feature_ablation_analysis.json"
        method = "feature_ablation"
    elif sensitivity >= 0.5:
        # perturbation
        analysis_file = explainer_results_path / "feature_perturbation_analysis.json"
        method = "feature_perturbation"

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Explainer analysis results not found at {analysis_file}"
        )

    print(f"Running explainer visualization with {script} (method: {method})")

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_file),
        "--output",
        str(plots_path),
        "--sensitivity",
        str(sensitivity),
    ]
    subprocess.run(cmd, check=True)

    print(f"Explainer visualization completed. Plots saved to {plots_path}")


# -----------------------
# Define Pipeline
# -----------------------
@dsl.pipeline(
    name="STAR Model Fairness-Bias and Explainer Pipeline",
    description="Runs fairness-bias and explainer analyses.",
)
def star_pipeline(
        github_repo_url: str,
        branch: str = "main",
        sensitivity: float = 0.3,
):
    """Pipeline to run STAR model fairness/bias and explainer analyses.

    :param github_repo_url: URL of the GitHub repository containing the STAR code and data.
    :param branch: Branch name to pull from (defaults to 'main').
    :param sensitivity: Sensitivity parameter for the explainer analysis. Defaults to 0.3.
    """
    # Step 1: Download repository
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    # Step 2: Make and store predictions
    predictions_task = star_predictions(
        input_data=repo_task.outputs["data"]
    )
    predictions_task.after(repo_task)
    predictions_task.set_caching_options(False)
    predictions_task.set_cpu_request("4000m")
    predictions_task.set_cpu_limit("6000m")
    predictions_task.set_memory_request("8Gi")
    predictions_task.set_memory_limit("12Gi")

    # Step 3: Fairness analysis
    fairness_task = fairness_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        predictions=predictions_task.outputs["predictions"],
    )
    fairness_task.after(predictions_task)
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("2000m")
    fairness_task.set_cpu_limit("4000m")
    fairness_task.set_memory_request("4Gi")
    fairness_task.set_memory_limit("8Gi")

    # Step 4: Fairness visualization
    fairness_viz_task = fairness_visualization(
        project_files=repo_task.outputs["project_files"],
        fairness_results=fairness_task.outputs["fairness_results"],
    )
    fairness_viz_task.after(fairness_task)
    fairness_viz_task.set_caching_options(False)
    fairness_viz_task.set_cpu_request("1000m")
    fairness_viz_task.set_cpu_limit("2000m")
    fairness_viz_task.set_memory_request("2Gi")
    fairness_viz_task.set_memory_limit("4Gi")

    # Step 5: Explainer analysis
    explainer_task = explainer_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        sensitivity=sensitivity,
    )
    explainer_task.after(repo_task)
    explainer_task.set_caching_options(False)
    explainer_task.set_cpu_request("4000m")
    explainer_task.set_cpu_limit("6000m")
    explainer_task.set_memory_request("8Gi")
    explainer_task.set_memory_limit("12Gi")

    # Step 6: Explainer visualization
    explainer_viz_task = explainer_visualization(
        project_files=repo_task.outputs["project_files"],
        explainer_results=explainer_task.outputs["explainer_results"],
        sensitivity=sensitivity,
    )
    explainer_viz_task.after(explainer_task)
    explainer_viz_task.set_caching_options(False)
    explainer_viz_task.set_cpu_request("1000m")
    explainer_viz_task.set_cpu_limit("2000m")
    explainer_viz_task.set_memory_request("2Gi")
    explainer_viz_task.set_memory_limit("4Gi")


if __name__ == "__main__":
    kfp_compiler = compiler.Compiler()
    kfp_compiler.compile(pipeline_func=star_pipeline, package_path="star_pipeline.yaml")
