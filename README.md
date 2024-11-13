# agent-type-identification-algorithms-with-experiment
In this package, it includes the whole Agent Miner approach and four agent-type identification techniques (AM, BME, AMatrix, CWM). We created the pipeline that combine the agent-type identification techniques and the Agent system mining technique. Here are some points related to how to utilize this package:

## How to select the agent type identification techniques and set the thresholds?
There are multiple parameter triplets, and all the identification techniques and the thresholds can be set in the `agent_miner_main.py`:
1. Agent Miner (AM) parameters: these parameters are collected in the `am_threshold_list`, each parameter triplet of AM is set as a tuple with two float number. You can modify the `am_threshold_list` by adding more parameter triplets based on your preferences.
2. Inductive Miner (IM) parameters: these parameters are collected in the `soa_threshold_list`. To run IM approach, only need to set the single float number as its parameter. You can modify the `soa_threshold_list` by adding more parameters based on your preferences.
3. Agent type identification technique selection: we use the `clustering_type_list` to collect the identification techniques we would like to experiment. These four techniques can be written as `AgentMiner` (it is an identification approach here), `BME`, `ActivityMatrix`, `CWM`. You can modify the `soa_threshold_list` by adding or deleting identification approaches based on your preferences.
4. Agent type identification technique threshold: in order to set thresholds to these approaches, we create a dictionary `clustering_threshold_dictionary` to collect these thresholds. For `AgentMiner`, `CWM` and `ActivityMatrix` approaches, we need to set the single float number as its parameter. For `BME` approach, each parameter triplet is set as a tuple with three float number. You can modify the `clustering_threshold_dictionary` based on your preferences.
   

## How to run the experiment?

## What are the outputs of the experiment?
