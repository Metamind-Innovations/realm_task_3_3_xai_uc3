import subprocess
import json
from pathlib import Path
from typing import Dict, Any
import uuid


class STARDockerWrapper:
    def __init__(
            self,
            in_mount: str = "temp_mount",
            out_mount: str = "temp_mount",
            docker_image: str = "glucomeo",
            in_docker_run: bool = False,
    ):
        """
        Wrapper for the STAR Dockerized model to allow prediction from Python.

        :param in_mount: Local directory to mount as `/home/in` inside the container.
        :param out_mount: Local directory to mount as `/home/out` inside the container.
        :param docker_image: Name of the Docker image containing the STAR model.
        :param in_docker_run: Indicates if the class runs inside the container.
        """
        self.docker_image = docker_image
        self.in_mount = Path(in_mount).resolve()
        self.out_mount = Path(out_mount).resolve()
        self.in_docker_run = in_docker_run

    def _validate_patient_data(self, patient_data: Dict[str, Any]) -> None:
        """
        Validate that patient data contains required fields.

        :param patient_data: Patient data dictionary.
        :raises ValueError: If required fields are missing or invalid.
        """
        required_fields = ["__class", "hospitalID", "updateTime", "episodes"]

        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field in patient data: {field}")

        if not patient_data["episodes"]:
            raise ValueError("Patient data must contain at least one episode")

        # Check for required episode fields
        episode = patient_data["episodes"][0]
        required_episode_fields = [
            "bloodGlucose",
            "insulinInfusion",
            "nutritionInfusion",
        ]

        for field in required_episode_fields:
            if field not in episode:
                raise ValueError(f"Missing required field in episode: {field}")

    def _validate_prediction_time(
            self, patient_data: Dict[str, Any], prediction_time: int
    ) -> None:
        """
        Validate that prediction time is within acceptable range.

        :param patient_data: Patient data dictionary.
        :param prediction_time: Unix epoch time in milliseconds.
        :raises ValueError: If prediction time is outside valid range.
        """
        update_time = patient_data["updateTime"]
        max_time = update_time + (180 * 60 * 1000)

        if prediction_time < update_time:
            raise ValueError(
                f"Prediction time ({prediction_time}) must be >= updateTime ({update_time})"
            )

        if prediction_time > max_time:
            raise ValueError(
                f"Prediction time ({prediction_time}) must be <= updateTime + 180 minutes ({max_time})"
            )

    def predict(
            self,
            patient_data: Dict[str, Any],
            prediction_time: int,
    ) -> Dict[str, float]:
        """
        Predict blood glucose range at a specific time using Docker.

        :param patient_data: Complete patient data JSON object.
        :param prediction_time: Unix epoch time in milliseconds when to predict the blood glucose range.
        :return: Dictionary containing prediction interval with keys BG5TH and BG95TH.
        :raises ValueError: If patient data or prediction time validation fails.
        :raises RuntimeError: If Docker execution fails.
        """
        # Validate inputs
        self._validate_patient_data(patient_data)
        self._validate_prediction_time(patient_data, prediction_time)

        self.in_mount.mkdir(parents=True, exist_ok=True)
        self.out_mount.mkdir(parents=True, exist_ok=True)
        u_id = uuid.uuid4().hex

        tmp_in_filename = f"patient_{u_id}.json"
        input_file = self.in_mount / tmp_in_filename

        request_payload = {
            "patient": patient_data,
            "predictionTime": prediction_time
        }

        with open(input_file, "w") as f:
            json.dump(request_payload, f)

        tmp_out_filename = f"prediction_{u_id}.json"
        output_file = self.out_mount / tmp_out_filename

        if not self.in_docker_run:
            cmd = [
                "docker",
                "run",
                "--rm",
                "-e", "AEONICS_JAVA_OPTIONS=-Xmx1g",
                "-e", "AEONICS_LICENSE_STORE_PATH=/opt/aeonics/aeonics.license",
                "-e", "AEONICS_LICENSE_STORE_PASS=secret",
                "-e", "AEONICS_ACCEPT_UNSIGNED_MODULES=true",
                "-e", "AEONICS_LOG_LEVEL=1000",
                "-e", "REALM_INPUT_DIR=/home/in",
                "-e", "REALM_OUTPUT_DIR=/home/out",
                "-w", "/opt/aeonics",
                "-u", "0",
                "-v", f"{self.in_mount}:/home/in",
                "-v", f"{self.out_mount}:/home/out",
                self.docker_image,
            ]
        else:
            cmd = [
                "/opt/aeonics/star_predict",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Docker execution failed: {e.stderr if e.stderr else e.stdout}"
            )

        if not output_file.exists():
            raise RuntimeError(f"Output file not generated: {output_file}")

        with open(output_file, "r") as f:
            result = json.load(f)

        if "BG5TH" not in result or "BG95TH" not in result:
            raise ValueError(
                f"Docker output missing required fields. Got: {result.keys()}"
            )

        prediction_interval = {
            "BG5TH": result["BG5TH"],
            "BG95TH": result["BG95TH"],
        }

        try:
            input_file.unlink(missing_ok=True)
            output_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: could not delete temp files: {e}")

        return prediction_interval

    def validate_prediction(
            self,
            interval: Dict[str, float],
            ground_truth: float,
    ) -> int:
        """
        Check whether ground truth value falls within the predicted range.

        :param interval: Prediction interval with BG5TH and BG95TH.
        :param ground_truth: Actual blood glucose value to compare against the last predicted range.
        :return: Binary prediction correctness indicator (1 if inside, 0 otherwise).
        """

        # Check if ground truth is within the predicted range
        bg_5th = interval["BG5TH"]
        bg_95th = interval["BG95TH"]

        is_inside = int(bg_5th <= ground_truth <= bg_95th)

        return is_inside
