Uniform Circular Array wav-files synthesis library in python.

Table of Contents
===

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Scripts](#scripts)
  - [uca_matrix_files_synthese.py](#uca_matrix_files_synthesepy)
  - [neural_network_dataset_prepare.py](#neural_network_dataset_preparepy)
  - [neural_network.py](#neural_networkpy)
  - [benchmark.py](#benchmarkpy)
- [Summary](#summary)

Introduction
===
Complete set consist of 4 scripts described below. For more information, 
please, analyze function's documentation inside of the scripts.


Scripts
====

uca_matrix_files_synthese.py
---
Script used to generate .wav files for N microphones, based on number of microphones, matrix radius and audiowave arrival angle.
Audiowave arrival angle is set from 0-359, where 0 is northern microphone

neural_network_dataset_prepare.py
---
Script used to prepare dataset for neural_network.py, using structures generated via uca_matrix_files_synthese.py.

neural_network.py
---
Uses neural_network_dataset_prepare.py datasets as input values into neural network. It's core for ML solution to solve DOA-estimation problem.

benchmark.py
---
Used for benchmarking usability of NN (ML) in comparison with GCC-PHAT and MUSIC algorithm.


Summary
===
Readme currently would not be updated until project go into stable phase. Last Edit: 06.02.2022
