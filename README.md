## About The Project

This repository contains a set of AI models designed to improve energy arbitrage by leveraging insights from a substantial cluster of domestic electric vehicle users. Developed during the hackathon [AI fighting climate change: Open Innovation Challenge](https://nuwe.io/dev/competitions/schneider-electric-european-2023/ai-fighting-climate-change-open-innovation-challenge), these source files are intended as a proof of concept and are not suitable for deployment in any production environment.

The primary functionality of the project includes:

- Creating mock vehicle data for training both models.
- Training a discharge capacity prediction model: This model forecasts the amount of energy discharged into the network, considering factors such as the date (day of the week, week of the year) and available capacity (total battery storage).
- Training a user behavior prediction model: This model predicts the duration, in minutes, that each user will have their car connected to the power grid. It considers factors such as the total minutes the user connected their car during the previous 30 days.

### Built Using

- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

## Getting Started

> You can find two notebooks ([discharge_capacity.ipynb](doc/discharge_capacity.ipynb) and [user_behaviour.ipynb](doc/user_behaviour.ipynb)) that walk you through the base use case for each of the implemented models.

Although the files in this repository are not production-ready, they are intended to be easily integrated into a microservices architecture.

Assuming [Python 3.11+](https://www.python.org/downloads/) and [pip](https://pypi.org/project/pip/) are installed and correctly configured, and you have [CUDA-capable hardware](https://developer.nvidia.com/cuda-gpus) installed, follow these steps:

### Prerequisites

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 11.0 or above is correctly installed.
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) version 7 or above is correctly installed.

### Installation

1. Clone this repository locally.

    ```bash
    git clone git@github.com:Pumaymf/schneider-ai-challenge.git
    ```

2. Create a Python [virtual environment](https://docs.python.org/3/library/venv.html) and activate it (**recommended**).

    ```bash
    python -m venv env
    source env/bin/activate 
    ```

3. Install all required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

## Execution

Basic use-case execution examples can be ran using the scripts provided at [scripts/](scripts/). These include

1. Generating mock data for `discharge_capacity` and `user_behaviour` models.

    Example:

    ```bash
    python scripts/discharge_capacity/data_population.py
    python scripts/user_behaviour/data_population.py
    ```

    > The generated datasets will be saved at [data/discharge_capacity/train.csv](data/discharge_capacity/train.csv) and [data/user_behaviour/train.csv](data/user_behaviour/train.csv), respectively.

2. Training each of the models.

    Example:

    ```bash
    python scripts/discharge_capacity/model_training.py
    python scripts/user_behaviour/model_training.py
    ```

    > The generated models will be stored at [model/discharge_capacity/recurrent_regression_model.pkl](model/discharge_capacity/recurrent_regression_model.pkl) and [model/user_behaviour/recurrent_regression_window_model.pkl](model/user_behaviour/recurrent_regression_window_model.pkl), respectively.

## Contributing

As this project is being developed during a competition, PRs from individuals outside the competition will **not** be **allowed**. Feel free to fork this repo and follow the development as you see fit.

Don't forget to give the project a star!

## Acknowledgements

Here is a list of people who contributed to the outcomes of this project. We apologize for any omissions:

- [Schneider](https://www.se.com/) mentors — for their invaluable assistance in guiding us to provide a solution to a real problem that will undoubtedly save many lives.
- [Victor Figueroa](https://www.linkedin.com/in/victorfigma/), [Iago Barreiro](https://www.linkedin.com/in/iagobarreirorio/), and [Santiago Pérez](https://www.linkedin.com/in/sperezacuna) — for allowing us to use their [hackathon-winning repository](https://github.com/sperezacuna/oracle-challenge-f3) as a starting point for this one.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

- Andrea Herrera Arrieta - andreaherrera0367@gmail.com
- Álvaro Luis Martínez González - alvaroluismartinezgonzalez@gmail.com
