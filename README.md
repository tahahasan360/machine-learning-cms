# machine-learning-cms

This repository contains the code, results and final report for my final project for Rice's course COMP 480: Probabilistic Algorithms and Data Structures (Fall 2023). My project was an exploration of how machine learning models can be used to improve the resource-accuracy tradeoff of the Count-Min Sketch (CMS) data structure. 

The results of the project are documented in `Final_Report.pdf`.

To reproduce the results, do the following:
1. Ensure that the `data/aol folder` is in the top-level directory.
2. Ensure that the `data/aol folder` contains 5 days of training data from the AOL dataset (https://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/). Each day of data must be in a separate `.txt` file with the naming convention "data/aol/user-ct-test-collection-0`i`.txt" where `i` ranges from 1 to 5.
3. Run `main.ipynb` in its entirety.


