import subprocess
import pandas as pd
from pathlib import Path
from typing import List
import shutil
import os


class STARDockerWrapper:
    def __init__(
            self,
            in_mount: str = "temp_mount/in",
            out_mount: str = "temp_mount/out",
            docker_image: str = "glucomeo",
            in_docker_run: bool = False,
    ):
        """
        Wrapper for the STAR Dockerized model to allow batch prediction from Python.

        :param in_mount: Local directory to mount as `/home/in` inside the container.
        :param out_mount: Local directory to mount as `/home/out` inside the container.
        :param docker_image: Name of the Docker image containing the STAR model.
        :param in_docker_run: Indicates if the class runs inside the container.
        """
        self.docker_image = docker_image
        self.in_mount = Path(in_mount).resolve()
        self.out_mount = Path(out_mount).resolve()
        self.in_docker_run = in_docker_run

        if not in_docker_run:
            docker_cmd = shutil.which("docker")
            if docker_cmd is None:
                docker_cmd = "docker"
            self.docker_executable = docker_cmd

            try:
                subprocess.run(
                    [self.docker_executable, "version"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Docker is installed but not responding correctly.\n"
                    f"Error: {e.stderr if e.stderr else e.stdout}\n"
                    f"Please ensure Docker Desktop is running."
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "Docker command timed out. Please ensure Docker Desktop is running."
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Cannot execute Docker at: {self.docker_executable}\n"
                    f"Please ensure Docker Desktop is installed and running."
                )

    def _format_volume_mount(self, local_path: Path, container_path: str) -> str:
        """
        Format volume mount string for Docker, handling Windows paths.

        :param local_path: Local filesystem path.
        :param container_path: Container filesystem path.
        :return: Formatted volume mount string.
        """
        # Use the path as-is on Windows - Docker Desktop handles Windows paths with backslashes
        return f"{str(local_path)}:{container_path}"

    def predict_batch(self, patient_files: List[str]) -> pd.DataFrame:
        """
        Run batch prediction on multiple patient files using Docker.

        :param patient_files: List of paths to patient JSON files.
        :return: DataFrame with columns BG5TH and BG95TH for each patient.
        :raises RuntimeError: If Docker execution fails or output file not found.
        """
        self.in_mount.mkdir(parents=True, exist_ok=True)
        self.out_mount.mkdir(parents=True, exist_ok=True)

        for patient_file in patient_files:
            src = Path(patient_file)
            dst = self.in_mount / src.name
            shutil.copy2(src, dst)

        if not self.in_docker_run:
            in_volume = self._format_volume_mount(self.in_mount, "/home/in")
            out_volume = self._format_volume_mount(self.out_mount, "/home/out")

            cmd = [
                self.docker_executable,
                "run",
                "--rm",
                "-e", "AEONICS_JAVA_OPTIONS=-Xmx1g",
                "-e", "AEONICS_LICENSE_STORE_PATH=/opt/aeonics/aeonics.license",
                "-e", "AEONICS_LICENSE_STORE_PASS=secret",
                "-e", "AEONICS_ACCEPT_UNSIGNED_MODULES=true",
                "-e", "AEONICS_LOG_LEVEL=1000",
                "-e", f"REALM_INPUT_DIR=/home/in",
                "-e", f"REALM_OUTPUT_DIR=/home/out",
                "-w", "/opt/aeonics",
                "-u", "0",
                "-v", in_volume,
                "-v", out_volume,
                self.docker_image,
            ]
        else:
            os.environ["REALM_INPUT_DIR"] = str(self.in_mount)
            os.environ["REALM_OUTPUT_DIR"] = str(self.out_mount)
            os.environ["AEONICS_JAVA_OPTIONS"] = "-Xmx1g"
            os.environ["AEONICS_LICENSE_STORE_PATH"] = "/opt/aeonics/aeonics.license"
            os.environ["AEONICS_LICENSE_STORE_PASS"] = "secret"
            os.environ["AEONICS_ACCEPT_UNSIGNED_MODULES"] = "true"
            os.environ["AEONICS_LOG_LEVEL"] = "1000"

            cmd = "cd /opt/aeonics && /opt/aeonics/jre/bin/java -Xmx1g -jar aeonics.jar"

        try:
            if self.in_docker_run:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            else:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            cmd_str = cmd if isinstance(cmd, str) else ' '.join(cmd)
            error_msg = f"Model execution failed.\nCommand: {cmd_str}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            raise RuntimeError(error_msg)

        output_file = self.out_mount / "results.csv"
        if not output_file.exists():
            raise RuntimeError(f"Output file not generated: {output_file}")

        results_df = pd.read_csv(output_file)

        if "BG5TH" not in results_df.columns or "BG95TH" not in results_df.columns:
            raise ValueError(
                f"Output CSV missing required columns. Got: {results_df.columns.tolist()}"
            )

        # Cleanup temporary directories
        try:
            if self.in_mount.exists():
                shutil.rmtree(self.in_mount, ignore_errors=True)

            if self.out_mount.exists():
                shutil.rmtree(self.out_mount, ignore_errors=True)

            # Remove parent temp_mount directory
            temp_mount_parent = self.in_mount.parent
            if temp_mount_parent.exists() and temp_mount_parent.name == "temp_mount":
                # Remove all remaining files in temp_mount
                for item in temp_mount_parent.iterdir():
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                # Remove the directory itself
                temp_mount_parent.rmdir()
        except Exception:
            pass

        return results_df

    def validate_predictions(
            self,
            predictions: pd.DataFrame,
            ground_truth: pd.Series,
    ) -> pd.Series:
        """
        Check whether ground truth values fall within predicted ranges.

        :param predictions: DataFrame with BG5TH and BG95TH columns.
        :param ground_truth: Series of actual blood glucose values.
        :return: Series of binary indicators (1 if inside range, 0 otherwise).
        """
        within_lower_bound = ground_truth >= predictions["BG5TH"]
        within_upper_bound = ground_truth <= predictions["BG95TH"]
        is_inside = within_lower_bound & within_upper_bound

        return is_inside.astype(int)
