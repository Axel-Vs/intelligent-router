# DialARideSolver
An implementation of various algorithms for the Dial-a-Ride Problem (DARP). Focused on optimizing vehicle routing and scheduling for ride-sharing services to minimize travel time and improve customer satisfaction.


## Overview
The Dial-a-Ride Problem (DARP) is an optimization problem that involves the scheduling and routing of vehicles to pick up and deliver passengers while meeting a set of service constraints. This project tackles this problem by implementing various algorithms to provide an efficient solution, ultimately aiming to minimize travel time and improve customer satisfaction.


## Getting Started
### Prerequisites
The project is developed using Python 3.8+. You'll need it installed on your system before proceeding.

### Installation
1. Clone this repository to your local machine using
```{bash}
git clone https://github.com/Axel-Vs/DialARideSolver.git
```

2. Navigate to the project directory with
```{bash}
cd DialARideProblemSolver
```

3. Install the required dependencies with
```{bash}
pip install -r requirements.txt
``` 


## Algorithms
* Quantum Annealing: Quantum Annealing is a metaheuristic for finding the global minimum of a given objective function over a given set of candidate solutions. Quantum annealing uses quantum mechanics to perform optimization and search tasks far more efficiently than traditional methods. In this project, we apply Quantum Annealing to solve DARP by mapping the problem into a suitable Ising model, finding optimal solutions in a significantly reduced search space. <br>

* CBC Solver: The Coin-or branch and cut (CBC) is an open-source linear programming solver that is part of the COIN-OR project. It is a highly efficient algorithm used for solving linear programming problems, mixed integer programming (MIP), and other related problems. For the DARP, we utilize CBC to model and solve the problem as a MIP. By doing so, we can find optimal or near-optimal solutions within reasonable computational time. <br>

Each of these algorithms presents its own benefits and trade-offs. Quantum Annealing provides a quantum-computational approach that can potentially find optimal solutions in a reduced search space, while the CBC Solver uses traditional mathematical programming techniques that are proven to be effective for such combinatorial problems.


## Contributing
We welcome contributions! Please see the CONTRIBUTING.md file for more details




