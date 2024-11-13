# agent-type-identification-algorithms-with-experiment
In this package, it includes the whole Agent Miner approach and four agent-type identification techniques (AM, BME, AMatrix, CWM). We created the pipeline that combine the agent-type identification techniques and the Agent system mining technique. Here are some points related to how to utilize this package:

## How to add new event logs
When adding new event logs, you should create a package (give it a name) inside the bpic package, and it's better to store event logs as csv files. Then open the `agent_miner_main.py` and find the `bpic_log_dir_list`. And add a tuple with two elements, the first element is the name of the package that contains the new event data, the second element is the filtered threshold, which can help you to avoid the noise.

## How to select the agent type identification techniques and set the thresholds?
There are multiple parameter triplets, and all the identification techniques and the thresholds can be set in the `agent_miner_main.py`:
1. Agent Miner (AM) parameters: these parameters are collected in the `am_threshold_list`, each parameter triplet of AM is set as a tuple with two float number. You can modify the `am_threshold_list` by adding more parameter triplets based on your preferences.
2. Inductive Miner (IM) parameters: these parameters are collected in the `soa_threshold_list`. To run IM approach, only need to set the single float number as its parameter. You can modify the `soa_threshold_list` by adding more parameters based on your preferences.
3. Agent type identification technique selection: we use the `clustering_type_list` to collect the identification techniques we would like to experiment. These four techniques can be written as `AgentMiner` (it is an identification approach here), `BME`, `ActivityMatrix`, `CWM`. You can modify the `soa_threshold_list` by adding or deleting identification approaches based on your preferences.
4. Agent type identification technique threshold: in order to set thresholds to these approaches, we create a dictionary `clustering_threshold_dictionary` to collect these thresholds. For `AgentMiner`, `CWM` and `ActivityMatrix` approaches, we need to set the single float number as its parameter. For `BME` approach, each parameter triplet is set as a tuple with three float number. You can modify the `clustering_threshold_dictionary` based on your preferences.
   

## How to run the experiment?
Before running the experiment, make sure that you already installed all the python packages (i.e., pm4py, os, bs4). Also you need to download java to run the entropia package (for the precision and recall calculation). If everything is already, use this command `python3 agent_miner_main.py` to run.

## What are the outputs of the experiment?
There are multiple outputs:
1. the interaction model and the MAS model
2. agent type information (csv file, includes agent types and their resources)
3. for each agent type, there is an agent model
4. a csv file contains results of all MAS models generated from this dataset, multiple metrics are employed to evaluate these MAS models (precision, recall, size, directed/undirected modularity, Gini coefficient).
