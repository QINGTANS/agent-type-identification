import pm4py
import asm_dfg as dfg
import asm_pn as pn
import config as config

def discover_agent_model(agent_id,agent_log_pm4py,agent_xes_log_name=None):
    # a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=config._a_dfg_ff)
    a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=config._agent_threshold)
    a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pn.discover_pn_from_dfg(a_dfg_obj)
    return (a_pn_pm4py, a_im_pm4py, a_fm_pm4py,a_dfg_obj)

def get_agent_model(cluster_mas_log_df, activity_type):
    agent_petri_model_map = {}
    log_groupby_agent = cluster_mas_log_df.groupby(['agent_id'],sort=False)
    for agent_id,agent_log_df in log_groupby_agent:
        if type(agent_id) == tuple:
            agent_id = agent_id[0]
        df0 = pm4py.format_dataframe(agent_log_df, case_id='fragment_id', activity_key=activity_type, timestamp_key='timestamp')
        agent_log_pm4py = pm4py.convert_to_event_log(df0)
        cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py, cl_dfg_obj = discover_agent_model(agent_id,agent_log_pm4py,agent_xes_log_name=None)
        agent_petri_model_map[agent_id] = (cl_pn_pm4py, cl_im_pm4py, cl_fm_pm4py, cl_dfg_obj)
    return agent_petri_model_map