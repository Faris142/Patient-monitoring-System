# Patient Monitoring System

![Patient Monitoring System](path-to-your-image)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Patient Monitoring System is designed to provide real-time monitoring and classification of various medical conditions using a webcam feed. This system leverages machine learning models to classify conditions such as back pain, chest pain, coughing, falling down, headache, and neck pain based on body posture detected by the Mediapipe library. Additionally, it monitors the patient's heartbeat, providing visual feedback and generating reports on the detected conditions.

## Features
- Real-time classification of medical conditions using webcam feed.
- Heartbeat monitoring and visualization.
- Condition report generation and visualization.
- Email notifications for condition reports.

## Installation

### Prerequisites
- Python 3.6+
- Mediapipe
- OpenCV
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit

### Setup
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Faris142/Patient-monitoring-System.git
    cd Patient-monitoring-System
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained models and place them in the project directory:**
    - logistic_regression_model.pkl
    - label_encoder.pkl

## Usage

1. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2. **Using the Application:**
    - The webcam feed will start automatically, displaying real-time classification and heartbeat monitoring.
    - Press the "Stop and Generate Report" button to stop the monitoring and generate a detailed report of the detected conditions.
    - The generated report will be displayed, and an email notification will be sent to the configured email address.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes.
4. Commit your changes.
    ```bash
    git commit -m "Add some feature"
    ```
5. Push to the branch.
    ```bash
    git push origin feature-branch
    ```
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions, feel free to reach out to the project maintainer at faisal.alkhuraif@gmail.com.

## Publications

All publications using "NTU RGB+D" or "NTU RGB+D 120" Action Recognition Database or any of the derived datasets(see Section 8) should include the following acknowledgement: "(Portions of) the research in this paper used the NTU RGB+D (or NTU RGB+D 120) Action Recognition Dataset made available by the ROSE Lab at the Nanyang Technological University, Singapore."
Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 [PDF].
Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex C. Kot, "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding", IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2019. [PDF].

