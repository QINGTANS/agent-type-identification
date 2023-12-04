import pandas as pd
import pm4py
import os
import asm_data as da
import datetime as dt
import asm_dfg as dfg
import config as config

'''
df0 = f._preprocess_inst_log(
    log_dir = 'bpic_2015_01',
    orig_log_file_name = 'bpic_2015_01.csv',
    separator = ','
    col_map={
    'timestamp':'completeTime',
    'case_id':'case',
    'activity_type':'event',
    'agent_id':'org:resource'
    },
    # which is a function here
    preproces_evt_callback = preprocess_evt
    )
'''
def _preprocess_inst_log(
    log_dir,
    orig_log_file_name,
    separator,
    col_map,
    preproces_evt_callback = None,
    ):
    # get the path for the csv file: .../bpic_2015_01/bpic_2015_01.csv
    orig_log_file_path = os.path.join(os.getcwd(),log_dir,orig_log_file_name)
    print(dt.datetime.now(),f"Reading the original log {orig_log_file_path} ...")

    # convert the csv file into dataframe
    df0 = pd.read_csv(orig_log_file_path,sep=separator)

    print(dt.datetime.now(),f"Preprocessing the original log ...")
    def preproces_evt(evt):
        if not preproces_evt_callback is None:
            evt = preproces_evt_callback(evt)
        # add the agent_id information to the dataframe 'evt'
        # agent_activity_type = 'agent_1|01_HOOFD'
        evt['agent_activity_type'] = f"{evt[col_map['agent_id']]}|{evt[col_map['activity_type']]}"
        return evt
    # apply the function to each row in the dataframe
    df0 = df0.apply(preproces_evt,axis=1)
    col_map['agent_activity_type']='agent_activity_type'
    # only selected useful columns
    df0 = da.select_columns_xes_log(df0,col_map)
    # finally prepared csv file has been created
    prep_file_name = os.path.join(os.getcwd(),log_dir,"_log_preprocessed_not_filtered.csv")
    df0.to_csv(prep_file_name,sep=',')
    print(f"Preprocessed log '{prep_file_name}' is ready for the Agent Miner evaluation pipeline.")
    return df0

def _filter_inst_log(log_dir="bpic_",filter_ver="aol1",case_pc=10):
    log_file_path = os.path.join(os.getcwd(),log_dir,"_log_preprocessed_not_filtered.csv")
    print(dt.datetime.now(),f"Loading preprocessed not filtered log from {log_file_path} ...")
    df0 = pd.read_csv(log_file_path,sep=',')
    print(dt.datetime.now(),f"Filtering events ...")
    df0 = pm4py.format_dataframe(df0, case_id='case_id', activity_key='activity_type', timestamp_key='timestamp')
    log_pm4py = pm4py.convert_to_event_log(df0)
    variants = pm4py.get_variants_as_tuples(log_pm4py)
    k = round(len(variants) * case_pc/100) # use only top case_pc % most frequent logs
    log_pm4py = pm4py.filter_variants_top_k(log_pm4py, k)
    log_pm4py = da.add_fragments_to_events(log_pm4py)
    df0 = pm4py.convert_to_dataframe(log_pm4py)

    input_dir = os.path.join(os.getcwd(),log_dir,filter_ver,"_no_clustering_")
    os.system(f"mkdir {os.path.join(os.getcwd(),log_dir,filter_ver)}")
    os.system(f"mkdir {input_dir}")
    filtered_inst_log_path_base = os.path.join(input_dir,"_log_"+f"{config._label_type}")
    print(dt.datetime.now(),f"Writing filtered event selection with ativity-only labels ({config._label_type}) {filtered_inst_log_path_base}.csv ...")
    da.save_xes_log(df0,filtered_inst_log_path_base)
    return df0

def _discover_agent_inst_dfg(df0, log_dir="bpic_", filter_ver="aol1"):
    print(dt.datetime.now(),f"Started creating agent instance logs and DFGs ...")
    input_dir = os.path.join(os.getcwd(),log_dir,filter_ver,"_no_clustering_")
    agent_dfgs_dir = os.path.join(input_dir,"agents")
    os.system(f"mkdir {agent_dfgs_dir}")
    print(dt.datetime.now(),"Identifying agent instance traces ...")
    xes_inst_log_with_fragments_df = df0
    agent_dfg_map = {}
    print(dt.datetime.now(),"Grouping agent instance traces ...")
    xes_log_groupby_agent = xes_inst_log_with_fragments_df.groupby(['agent_id'],sort=False)
    inst_count = len(xes_log_groupby_agent)
    print(dt.datetime.now(),f"Discovering DFG for {inst_count} agent instances ...")
    agent_stats_list = []
    for agent_id,xes_agent_log_df in xes_log_groupby_agent:
        df0 = pm4py.format_dataframe(xes_agent_log_df, case_id='fragment_id', activity_key='activity_type', timestamp_key='timestamp')
        da.save_xes_log(df0,os.path.join(agent_dfgs_dir,str(agent_id)+'_log.csv'),save_xes=False)
        agent_log_pm4py = pm4py.convert_to_event_log(df0)
        a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=1.0)
        dfg.viz_dfg(a_dfg_obj,os.path.join(agent_dfgs_dir,str(agent_id)+'_dfg'),more_viz=False)
        #a_dfg_nx = dfg.dfg_obj2nx(a_dfg_obj)
        agent_dfg_map[agent_id] = a_dfg_obj
        agent_stats_list.append({
            'agent_inst' : agent_id,
            'event_count' : a_dfg_obj['event_count'],
            'activity_count' : a_dfg_obj['activity_count'],
            'flow_count' : a_dfg_obj['flow_count']
        })
    agent_stats_df = pd.DataFrame(agent_stats_list)
    agent_stats_df.to_csv(os.path.join(agent_dfgs_dir,'agents_dfg_summary.csv'))

    return agent_dfg_map
