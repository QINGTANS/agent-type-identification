import os
import datetime as dt
import pandas as pd
import pm4py
import new_pipeline as p
'''
bpic_log_dir_list = [("bpic_2013",80),("bpic_2020_TravelPermitData",80),("bpic_2012",80),("bpic_2017",80),("bpic_2015_1",80),
                     ("bpic_2014",10),("bpic_2018",10),("bpic_2019",10),("bpic_2011",10)]
'''
bpic_log_dir_list = [("bpic_2013",80)]

run_name="bpic"
# am_threshold_list = [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]

am_threshold_list = [(round(0.1+i*0.1,1),round(0.9-i*0.1,1)) for i in range(0,10)]
# am_threshold_list = [(0.3,0.7),(0.4,0.6),(0.5,0.5)]
# soa_threshold_list = [0.9, 0.7, 0.5, 0.3, 0.1]
# soa_threshold_list = [round(0.9-i*0.2,1) for i in range(0,5)]
# soa_threshold_list = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
soa_threshold_list = [round(0.9-i*0.1,1) for i in range(0,10)]
# run four clustering algorithms
clustering_type_list = ['AgentMiner', 'BME', 'ActivityMatrix', 'CWM']
# clustering_type_list = ['CWM']
clustering_threshold_dictionary = {# 'AgentMiner': [0.2,0.4,0.6,0.8,1.0], 0.2 Timeout
                                'AgentMiner': [0.5,0.6,0.8,0.9,1.0], # delete 0.4
                                'CWM': [0.5,0.6,0.7,0.8,0.9], # delete 0.4, 0.5, 0.6
                                'BME': [(0.4,0.9,0.9),(0.6,0.7,0.7),(0.7,0.7,0.7),(0.8,0.6,0.6),(0.9,0.5,0.5)],
                                'ActivityMatrix':[0.6,0.7,0.8,0.9,0.97]} # delete 0.5, 0.6
'''
clustering_threshold_dictionary = {'AgentMiner': [1.0],
                                   'CWM': [0.7,0.9],
                                   'BME': [(0.2,0.9,0.9),(0.4,0.8,0.8),(0.6,0.6,0.6),(0.6,0.7,0.7),(0.7,0.6,0.6)],
                                   'ActivityMatrix':[0.8,0.9]}
'''


for bpic_log_dir,variant_frequency_cover_percent in bpic_log_dir_list:
    dev_params_dict_am = {
        'run_name':run_name,
        'example_dir':bpic_log_dir,
        'manual_inst_cluster_map_map':{},
        'run_threshold_list':am_threshold_list,
        'run_log_percent_list':[variant_frequency_cover_percent],
        # new add
        'clustering_type_list':clustering_type_list,
        # new change
        'run_cluster_threshold_dictionary':clustering_threshold_dictionary,
        #'run_cluster_max_dist_list':[1.0],
        'label_type':'aol',
        'more_viz':False
    }
    # Discover AM
    # dev_params_dict_am['label_type'] = 'aal'
    am_output_dictionary = p._run_discover_am(**dev_params_dict_am)
    # p._run_completed(bpic_log_dir+run_name+"_discovery_am_aol")
    # output the result
    am_data = pd.DataFrame.from_dict(am_output_dictionary)
    file_name = f"{bpic_log_dir}_{str(int(variant_frequency_cover_percent))}_{dev_params_dict_am['label_type']}_am_output.csv"
    file_path = os.path.join(os.getcwd(),run_name,bpic_log_dir,file_name)
    # am_data.to_csv(file_path)


for bpic_log_dir,variant_frequency_cover_percent in bpic_log_dir_list:
    dev_params_dict_soa = {
        'run_name':run_name, # event_logs_csv
        'example_dir':bpic_log_dir, # bpic_2015_1
        'manual_inst_cluster_map_map':{},
        'run_threshold_list':soa_threshold_list, # [0.9, 0.7, 0.5, 0.3, 0.1]
        'run_log_percent_list':[variant_frequency_cover_percent], # [80]
        # new add
        # 'clustering_type_list':clustering_type_list,
        # new change
        # 'run_cluster_threshold_dictionary':clustering_threshold_dictionary,
        # 'run_cluster_max_dist_list':[1.0],
        'label_type':'aol',
        'more_viz':False
    }
    # Discover SOA
    dev_params_dict_soa['label_type'] = 'aol'
    soa_output_dictionary = p._run_discover_soa(**dev_params_dict_soa)
    # p._run_completed(bpic_log_dir+run_name+"_discover_soa_aol")
    # output the result
    soa_data = pd.DataFrame.from_dict(soa_output_dictionary)
    file_name = f"{bpic_log_dir}_{str(int(variant_frequency_cover_percent))}_{dev_params_dict_soa['label_type']}_soa_output.csv"
    file_path = os.path.join(os.getcwd(),run_name,bpic_log_dir,file_name)
    # soa_data.to_csv(file_path)
