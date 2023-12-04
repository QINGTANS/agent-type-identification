import pm4py
import _filter as f
# import _cluster as cl
# import _discover as di
# import _evaluate as ev
# import _viz as v
import os
import pandas as pd
import cluster_algorithms as CAL
import config as config
import asm_data as modify_log
from pm4py.objects.log.importer.xes import importer as xes_importer
import discovery as di
import asm_dfg as dfg
import asm_pn as pn
import datetime
import networkx as nx
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from timeout_decorator import timeout
import signal

'''
dev_params_dict_soa = {
        'run_name':run_name, # event_logs_csv
        'example_dir':bpic_log_dir, # bpic_2015_1
        'manual_inst_cluster_map_map':{},
        'run_threshold_list':soa_threshold_list, # [0.9, 0.7, 0.5, 0.3, 0.1]
        'run_log_percent_list':[variant_frequency_cover_percent], # [80]
        # new add
        'clustering_type_list':clustering_type_list,
        # new change
        'run_cluster_threshold_dictionary':clustering_threshold_dictionary,
        #'run_cluster_max_dist_list':[1.0],
        'label_type':'aol',
        'more_viz':False
    }
'''
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


# @timeout(1)
def calculate_pre_recall(entropia_jar, log_file_name, pn_file_name, entropia_file_name):
    os.system(f"timeout 3600 java -Xms4G -Xmx48G -jar {entropia_jar} -empr -rel={log_file_name} -ret={pn_file_name} > {entropia_file_name}")
    prec, recall = read_entropia_results(entropia_file_name)
    return prec, recall
    # output_dictionary['precision_list'].append(prec)
    # output_dictionary['recall_list'].append(recall)
    # output_dictionary['algorithm_threshold'].append(threshold)
    # output_dictionary['place_num'].append(len(std_pn_pm4py.places))
    # output_dictionary['transition_num'].append(len(std_pn_pm4py.transitions))
    # output_dictionary['arc_num'].append(len(std_pn_pm4py.arcs))

def _run_filter(example_dir=None,run_log_percent=None, label_type=None):
    pc = run_log_percent
    filter_v = label_type+str(pc)
    config._label_type = label_type
    config._filter_ver = filter_v
    df1 = f._filter_inst_log(log_dir=example_dir,filter_ver=filter_v,case_pc=pc)
    return df1

# create the event log for entropia test
def add_clusters_to_log(inst_cluster_map, log_df):
    def make_agent_evt(evt):
        inst_id = evt['agent_id']
        cluster_id = inst_cluster_map[inst_id]
        evt['agent_inst_id'] = inst_id
        evt['agent_id'] = cluster_id
        evt['agent_activity_type'] = f"{cluster_id}|{evt['activity_type']}"
        return evt
    log_df_test = log_df.apply(make_agent_evt,axis=1)
    return log_df_test

def viz_pn(pn_pm4py,pn_im_pm4py,pn_fm_pm4py,pn_file_name_base):
    print(datetime.datetime.now(),f"Writing PN to NX GML file {pn_file_name_base}.gml ...") 
    pn_nx = pn.pn_pm4py2nx(pn_pm4py,initial_marking=pn_im_pm4py,final_marking=pn_fm_pm4py)
    nx.write_gml(pn_nx,pn_file_name_base+".gml")
    print(datetime.datetime.now(),f"Writing PN GVIZ to file {pn_file_name_base}.pdf ...") 
    gviz = pn_visualizer.apply(pn_pm4py, pn_im_pm4py, pn_fm_pm4py, parameters={pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "pdf"}, variant=pn_visualizer.Variants.FREQUENCY)
    pn_visualizer.save(gviz, pn_file_name_base+".pdf")

def read_entropia_results(entropia_output_file_name):
    precision = -0.99
    recall = -0.99
    with open(entropia_output_file_name, 'r') as entopia_file:
            lines = entopia_file.readlines()
            if len(lines)>1: 
                    precision_line_parts = lines[-2].split(' ')
                    print("precision_line_parts",precision_line_parts)
                    if len(precision_line_parts)>1:
                            precStr = precision_line_parts[1].strip('\n').strip('.')
                            if precStr.replace(".","").isnumeric():
                                    precision = float(precStr)
                    recall_line_parts = lines[-1].split(' ')
                    print("recall_line_parts",recall_line_parts)
                    if len(recall_line_parts)>1:
                            recallStr = recall_line_parts[1].strip('\n').strip('.')
                            if recallStr.replace(".","").isnumeric():
                                    recall = float(recallStr)
    return (precision,recall)

def _run_discover_am(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_log_percent_list=[],
                     clustering_type_list=None,run_cluster_threshold_dictionary=None,label_type='aol',more_viz=False,
                     run_threshold_list=None):
    # add one to test
    # activity_type = 'agent_activity_type'
    output_dictionary = {'cluster_type':[], 'cluster_threshold':[], 'agent_number':[], 'algorithm_threshold':[],'precision_list':[], 
                         'recall_list':[], 'mas_place_num':[], 'mas_transition_num':[], 'mas_arc_num':[], '(pl+tr)':[], '(pl+tr+arc)':[]}
    activity_type = 'activity_type' if label_type == 'aol' else 'agent_activity_type'
    precision_list = []
    recall_list = []
    for run_log_percent in run_log_percent_list:
        combine_dir = os.path.join(run_name, example_dir)
        config._combine_dir = combine_dir
        type_information = label_type + str(run_log_percent)
        mas_log_df = _run_filter(combine_dir, run_log_percent, label_type)
        unclustered_log_path = os.path.join(os.getcwd(),combine_dir,type_information,'_no_clustering_',f"_log_{label_type}.xes")
        unclustered_log = xes_importer.apply(unclustered_log_path)

        for cluster_algorithm in clustering_type_list:
            cluster_threshold_list = run_cluster_threshold_dictionary[cluster_algorithm]
            config._cluster_alg = cluster_algorithm
            for threshold in cluster_threshold_list:
                config._cluster_alg_threshold = threshold
                inst_cluster_map = CAL.run_cluster(config, mas_log_df, unclustered_log)

                # to get the new event log for entropia test
                test_log_path = os.path.join(os.getcwd(), config._combine_dir, config._filter_ver, config._cluster_alg, config._part_file_name)
                log_df_test = add_clusters_to_log(inst_cluster_map, mas_log_df)
                log_df = pm4py.format_dataframe(log_df_test, case_id='case_id', activity_key=activity_type, timestamp_key='timestamp')
                log_pm4py = pm4py.convert_to_event_log(log_df)
                log_pm4py = modify_log.add_fragments_to_events(log_pm4py)
                log_df = pm4py.convert_to_dataframe(log_pm4py)
                clustered_log_name_base = os.path.join(test_log_path, f"{label_type}_test_log")
                modify_log.save_xes_log(log_df,clustered_log_name_base)

                for run_threshold in run_threshold_list:
                    config._agent_interaction_threshold = 'agent'+'_'+str(int(run_threshold[0]*100))+'_'+'interaction'+'_'+str(int(run_threshold[1]*100))
                    config._agent_threshold = run_threshold[0]
                    config._interaction_threshold = run_threshold[1]
                    cluster_mas_log_df = modify_log.add_clusters_to_events(mas_log_df, inst_cluster_map)
                    
                    
                    # cluster_mas_log_df = pm4py.format_dataframe(cluster_mas_log_df, case_id='case_id', activity_key=activity_type, timestamp_key='timestamp')
                    cluster_mas_log_df = log_df
                    cluster_mas_log = pm4py.convert_to_event_log(cluster_mas_log_df)
                    cluster_mas_log_fragment = modify_log.add_fragments_to_events(cluster_mas_log)
                    cluster_mas_log_fragment_df = pm4py.convert_to_dataframe(cluster_mas_log_fragment)
                    path_file_1 = os.path.join(os.getcwd(), config._combine_dir, config._filter_ver, config._cluster_alg, 
                                               config._part_file_name, config._agent_interaction_threshold)
                    if not os.path.exists(path_file_1):
                        os.makedirs(path_file_1)
                    df_file_name = os.path.join(path_file_1, 'cluster_mas_log_fragment_df.csv')
                    cluster_mas_log_fragment_df.to_csv(df_file_name)
                    # get agent models
                    agent_petri_model_map = di.get_agent_model(cluster_mas_log_fragment_df, activity_type)
                    cluster_mas_log = pm4py.convert_to_event_log(cluster_mas_log_df)
                    log_add_fragment = modify_log.add_fragments_to_events(cluster_mas_log)
                    log_df_add_fragment = pm4py.convert_to_dataframe(log_add_fragment)
                    # then comes to the function of da.create_interaction_log
                    fragment_log = modify_log.create_interaction_log(log_df_add_fragment, agent_trace_col_name='fragment_id')
                    # in_dfg_obj = dfg.discover_dfg_pm4py("i-net", fragment_log, activity_frequency_filter=config._in_dfg_ff)
                    if config._soa_alg:
                        in_pm4py,in_im_pm4py,in_fm_pm4py = pn.discover_pn_imf_pm4py(fragment_log, config._soa_alg, noise_threshold=config._interaction_threshold)
                    else:
                        in_dfg_obj = dfg.discover_dfg_pm4py("i-net", fragment_log, activity_frequency_filter=config._interaction_threshold)
                        in_pm4py,in_im_pm4py,in_fm_pm4py = pn.discover_pn_from_dfg(in_dfg_obj)
                    # discovering the MAS model
                    mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py = pn.integrate_mas_pn_pm4py(in_pm4py,in_im_pm4py,in_fm_pm4py,agent_petri_model_map)
                    # visualization part
                    viz_pn(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py, os.path.join(path_file_1, f"mas_net"))
                    pm4py.write_pnml(mas_pn_pm4py,mas_pn_im_pm4py,mas_pn_fm_pm4py, os.path.join(path_file_1, f"mas_petri"))
                    viz_pn(in_pm4py,in_im_pm4py,in_fm_pm4py, os.path.join(path_file_1, f"interaction_net"))
                    pm4py.write_pnml(in_pm4py,in_im_pm4py,in_fm_pm4py, os.path.join(path_file_1, f"interaction_petri"))
                    path_file_2 = os.path.join(path_file_1, 'agents')
                    if not os.path.exists(path_file_2):
                        os.makedirs(path_file_2)
                    for agent_id in agent_petri_model_map:
                        item = agent_petri_model_map[agent_id]
                        viz_pn(item[0],item[1],item[2], os.path.join(path_file_2, f"{agent_id}_net"))
                        pm4py.write_pnml(item[0],item[1],item[2], os.path.join(path_file_2, f"{agent_id}_petri"))
                    
                    # using entropia and the test log to test the mas_net
                    log_file_name = os.path.join(test_log_path, f"{label_type}_test_log.xes")
                    petri_path = os.path.join(path_file_1, f"mas_petri")
                    pn.fix_pnml_for_jbpt(petri_path)
                    pn_file_name = os.path.join(petri_path)+"_jbpt.pnml"
                    entropia_file_name = os.path.join(path_file_1, f"{label_type}_entropia.txt")
                    entropia_jar = os.path.join(os.getcwd(),"entropia","jbpt-pm-entropia-1.6.jar")
                    output_dictionary['cluster_type'].append(cluster_algorithm)
                    output_dictionary['cluster_threshold'].append(threshold)
                    output_dictionary['agent_number'].append(len(agent_petri_model_map))
                    output_dictionary['algorithm_threshold'].append(run_threshold)
                    output_dictionary['mas_place_num'].append(len(mas_pn_pm4py.places))
                    output_dictionary['mas_transition_num'].append(len(mas_pn_pm4py.transitions))
                    output_dictionary['mas_arc_num'].append(len(mas_pn_pm4py.arcs))
                    output_dictionary['(pl+tr)'].append(len(mas_pn_pm4py.places)+len(mas_pn_pm4py.transitions))
                    output_dictionary['(pl+tr+arc)'].append(len(mas_pn_pm4py.places)+len(mas_pn_pm4py.transitions)+len(mas_pn_pm4py.arcs))
                    prec, recall = calculate_pre_recall(entropia_jar, log_file_name, pn_file_name, entropia_file_name)
                    if prec == -0.99 or recall == -0.99:
                        output_dictionary['precision_list'].append('Timeout')
                        output_dictionary['recall_list'].append('Timeout')
                    else:
                        output_dictionary['precision_list'].append(prec)
                        output_dictionary['recall_list'].append(recall)
                    am_data = pd.DataFrame.from_dict(output_dictionary)
                    file_name = f"{example_dir}_{str(int(run_log_percent))}_{label_type}_am_output.csv"
                    file_path = os.path.join(os.getcwd(),run_name,example_dir,file_name)
                    am_data.to_csv(file_path)
    return output_dictionary

def _run_discover_soa(run_name=None,example_dir=None,manual_inst_cluster_map_map={},run_threshold_list=[],run_log_percent_list=[],
                     label_type='aol',more_viz=False):
    output_dictionary = {'algorithm_threshold':[],'precision_list':[],'recall_list':[], 'place_num':[], 'transition_num':[], 'arc_num':[], '(pl+tr)':[], '(pl+tr+arc)':[]}
    # activity_type = 'activity_type' if label_type == 'aol' else 'agent_activity_type'
    precision_list = []
    recall_list = []
    for run_log_percent in run_log_percent_list:
        combine_dir = os.path.join(run_name, example_dir)
        config._combine_dir = combine_dir
        type_information = label_type + str(run_log_percent)
        # mas_log_df = _run_filter(combine_dir, run_log_percent, label_type)
        unclustered_log_path = os.path.join(os.getcwd(),combine_dir,type_information,'_no_clustering_',f"_log_{label_type}.xes")
        unclustered_log = pm4py.read_xes(unclustered_log_path)
        store_path = os.path.join(os.getcwd(),combine_dir,type_information,'_no_clustering_')
        for threshold in run_threshold_list:
            store_name = 'no_cluster'+'_'+str(int(threshold*100))
            std_pn_pm4py, std_im_pm4py, std_fm_pm4py = pn.discover_pn_imf_pm4py(unclustered_log, config._soa_alg, noise_threshold=threshold)
            viz_pn(std_pn_pm4py, std_im_pm4py, std_fm_pm4py,os.path.join(store_path, f"{store_name}_mas_net"))
            pm4py.write_pnml(std_pn_pm4py, std_im_pm4py, std_fm_pm4py, os.path.join(store_path, f"{store_name}_mas_petri"))
            log_file_name = os.path.join(unclustered_log_path)
            petri_path = os.path.join(store_path, f"{store_name}_mas_petri")
            pn.fix_pnml_for_jbpt(petri_path)
            pn_file_name = os.path.join(petri_path)+"_jbpt.pnml"
            entropia_file_name = os.path.join(store_path, f"{store_name}_{label_type}_entropia.txt")
            entropia_jar = os.path.join(os.getcwd(),"entropia","jbpt-pm-entropia-1.6.jar")
            output_dictionary['algorithm_threshold'].append(threshold)
            output_dictionary['place_num'].append(len(std_pn_pm4py.places))
            output_dictionary['transition_num'].append(len(std_pn_pm4py.transitions))
            output_dictionary['arc_num'].append(len(std_pn_pm4py.arcs))
            output_dictionary['(pl+tr)'].append(len(std_pn_pm4py.places)+len(std_pn_pm4py.transitions))
            output_dictionary['(pl+tr+arc)'].append(len(std_pn_pm4py.places)+len(std_pn_pm4py.transitions)+len(std_pn_pm4py.arcs))
            prec, recall = calculate_pre_recall(entropia_jar, log_file_name, pn_file_name, entropia_file_name)
            if prec == -0.99 or recall == -0.99:
                output_dictionary['precision_list'].append('Timeout')
                output_dictionary['recall_list'].append('Timeout')
            else:
                output_dictionary['precision_list'].append(prec)
                output_dictionary['recall_list'].append(recall)
            soa_data = pd.DataFrame.from_dict(output_dictionary)
            file_name = f"{example_dir}_{str(int(run_log_percent))}_{label_type}_soa_output.csv"
            file_path = os.path.join(os.getcwd(),run_name,example_dir,file_name)
            soa_data.to_csv(file_path)
            '''
            os.system(f"java -Xms4G -Xmx48G -jar {entropia_jar} -empr -rel={log_file_name} -ret={pn_file_name} > {entropia_file_name}")
            prec, recall = read_entropia_results(entropia_file_name)
            precision_list.append(prec)
            recall_list.append(recall)
            output_dictionary['precision_list'].append(prec)
            output_dictionary['recall_list'].append(recall)
            '''
    return output_dictionary



