import pandas as pd
import pm4py
import config as config
import os
import math
import datetime as dt
import asm_dfg as dfg
import asm_pn as pn
import asm_data as modify_log
from pm4py.algo.filtering.dfg import dfg_filtering
from pm4py.objects.conversion.dfg import converter as dfg_mining
from AM_cluster import discover_agent_dfg_distance_graph as get_distance_matrix
from AM_cluster import group_agents_to_clusters as group_clusters

# for the Agent Miner clustering algorithm
def discover_agent_model(agent_id,agent_log_pm4py,agent_xes_log_name=None):
    # a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=config._a_dfg_ff)
    a_dfg_obj = dfg.discover_dfg_pm4py(agent_id,agent_log_pm4py,activity_frequency_filter=1.0)
    a_pn_pm4py, a_im_pm4py, a_fm_pm4py = pn.discover_pn_from_dfg(a_dfg_obj)
    return (a_pn_pm4py, a_im_pm4py, a_fm_pm4py,a_dfg_obj)

def get_agent_model_map(mas_log_df):
    event_concept_name_attr = 'activity_type'
    agent_model_map = {}
    # agent_stats_list = []
    log_without_fragment = pm4py.convert_to_event_log(mas_log_df)
    # log_with_fragment = combine_resource_and_case(log_without_fragment)
    log_with_fragment = modify_log.add_fragments_to_events(log_without_fragment)
    new_mas_log_df = pm4py.convert_to_dataframe(log_with_fragment)
    log_groupby_agent = new_mas_log_df.groupby(['agent_id'],sort=False)
    for agent_id,agent_log_df in log_groupby_agent:
        # print(f"{datetime.now()} Discovering agent net for {str(agent_id)} ...")
        df0 = pm4py.format_dataframe(agent_log_df, case_id='fragment_id', activity_key=event_concept_name_attr, timestamp_key='timestamp')
        agent_log_pm4py = pm4py.convert_to_event_log(df0)
        dfg_pm4py, start_activities, end_activities = pm4py.discover_directly_follows_graph(agent_log_pm4py)
        activities_count = pm4py.get_event_attribute_values(agent_log_pm4py, "concept:name")
        activity_frequency_filter = 1
        dfg_pm4py, start_activities, end_activities, activities_count = dfg_filtering.filter_dfg_on_activities_percentage(
            dfg_pm4py, 
            start_activities, 
            end_activities, 
            activities_count, 
            activity_frequency_filter
            )
        total_event_count = 0
        for activity_type in activities_count:
            total_event_count += activities_count[activity_type]
        dfg_obj = {
            'agent_types':{agent_id},
            'event_count': total_event_count,
            'activity_count': len(activities_count),
            'flow_count': len(dfg_pm4py),
            'activity_types': activities_count,
            'control_flows': dfg_pm4py,
            'input_activity_types': start_activities,
            'output_activity_types': end_activities
        }
        # agent_model_map[agent_id[0]] = dfg_obj
        agent_model_map[agent_id] = dfg_obj
    return agent_model_map

def run_AgentMiner(log_df, threshold):
    agent_model_map = get_agent_model_map(log_df)
    dfg_dist_map,dfg_intersect_map,agent_with_dedicated_activities_list = get_distance_matrix(agent_model_map)
    inst_cluster_map, addg_nx = group_clusters(dfg_dist_map, threshold)
    final_inst_cluster_map = {}
    for resource_tuple in inst_cluster_map:
        if type(resource_tuple) == tuple:
            resource = resource_tuple[0]
            final_inst_cluster_map[resource] = inst_cluster_map[resource_tuple]
    if final_inst_cluster_map == {}:
        final_inst_cluster_map = inst_cluster_map
    return final_inst_cluster_map

def get_information_from_log(mas_log):
    # resource_action_dic = {trace_1:[(resource, action), ...], trace_2:[], ...}
    # can also make it as a dataframe
    resource_action_dic = {}
    # different resources set
    diff_resource_list = []
    # different actions set
    diff_task_list = []
    # in the BPIC15_1 xes, there are 1199 traces, each trace 43.55 tasks on average
    for trace_id in range(len(mas_log)):
        # resource_action_list = [(resource, action), (), ...]
        resource_action_list = []
        trace_name = 'trace_' + str(trace_id)
        current_trace = mas_log[trace_id]
        for task_id in range(len(mas_log[trace_id])):
            current_resource = current_trace[task_id]['agent_id']
            current_task_name = current_trace[task_id]['activity_type']
            if current_resource not in diff_resource_list:
                diff_resource_list.append(current_resource)
            if current_task_name not in diff_task_list:
                diff_task_list.append(current_task_name)
            resource_action_list.append((current_resource, current_task_name))
        resource_action_dic[trace_name] = resource_action_list
    return resource_action_dic, diff_resource_list, diff_task_list

def get_inst_cluster_map(cluster_agent_list):
    inst_cluster_map = {}
    agent_num = 1
    for cluster in cluster_agent_list:
        cluster_name = 'a' + str(agent_num)
        for resource in cluster:
            inst_cluster_map[resource] = cluster_name
        agent_num += 1
    return inst_cluster_map

# 2. comprehensive workflow mining
# get the ability map: ability_map = {task1: [agent1, agent2, ...], ...}
# get the frequency map, how many times agent1 do task1: frequency_map = {(task1, agent1):5, ...}
def get_ability_freq_maps(resource_action_dic):
    ability_map, frequency_map = {}, {}
    for trace_id in resource_action_dic:
        resource_action_list = resource_action_dic[trace_id]
        for resource_action_tuple in resource_action_list:
            curr_resource, curr_action = resource_action_tuple
            if curr_action in ability_map:
                if curr_resource not in ability_map[curr_action]:
                    ability_map[curr_action].append(curr_resource)
            elif curr_action not in ability_map:
                new_list = []
                new_list.append(curr_resource)
                ability_map[curr_action] = new_list
            if (curr_action, curr_resource) not in frequency_map:
                frequency_map[(curr_action, curr_resource)] = 1
            else:
                frequency_map[(curr_action, curr_resource)] += 1
    
    return ability_map, frequency_map

def CWM_cluster(ability_map, frequency_map, diff_resource_list, threshold):
    # get the filtered_ability_map (by frequency, if freq(task1, agent1)/freq(task1) < t, then delete this combination)
    def get_filtered_ability_map(ability_map, frequency_map, threshold):
        filtered_ability_map = {}
        task_freq_map = {}
        for task in ability_map:
            total_task_freq = 0
            for resource in ability_map[task]:
                total_task_freq += frequency_map[(task, resource)]
            task_freq_map[task] = total_task_freq
        for task_1 in ability_map:
            filtered_resource = []
            for resource_1 in ability_map[task_1]:
                if frequency_map[(task_1, resource_1)]/task_freq_map[task_1] >= threshold:
                    filtered_resource.append(resource_1)
            filtered_ability_map[task_1] = filtered_resource
        return filtered_ability_map

    # reverse_ability_map = {agent_1:[task_1, task_2, ...], ...}
    def get_reverse_ability_map(ability_map):
        reverse_ability_map = {}
        for task in ability_map:
            for resource in ability_map[task]:
                if resource not in reverse_ability_map:
                    new_task_list = []
                    new_task_list.append(task)
                    reverse_ability_map[resource] = new_task_list
                elif task not in reverse_ability_map[resource]:
                    reverse_ability_map[resource].append(task)
        return reverse_ability_map

    # return True or False, if all elements in element_list contains in the whole_list, then return True
    def contain_or_not(element_list, whole_list):
        if whole_list is None:
            return False
        if element_list is None:
            return True
        for element in element_list:
            if element not in whole_list:
                return False
        return True

    # dic = {m1:[a1, a2], m2:[a2, a4]}, keys_list = [m1, m2], result should be [a1, a2, a4]
    def get_results_from_keys(keys_list, dic):
        result_list = []
        for key in keys_list:
            for element in dic[key]:
                if element not in result_list:
                    result_list.append(element)
        return result_list

    if ability_map is None or frequency_map is None:
        return None
    cluster_agent_list = []
    cluster_task_list = []
    filtered_ability_map = get_filtered_ability_map(ability_map, frequency_map, threshold)
    reverse_filtered_ability_map = get_reverse_ability_map(filtered_ability_map)
    # total_resource_num = len(reverse_filtered_ability_map)
    considered_resource_list = []
    for task in filtered_ability_map:
        resource_list = filtered_ability_map[task]
        if contain_or_not(resource_list, considered_resource_list) is False:
            considered_task_list = []
            considered_task_list.append(task)
            task_list = get_results_from_keys(resource_list, reverse_filtered_ability_map)
            while not contain_or_not(task_list, considered_task_list):
                for task_1 in task_list:
                    if task_1 not in considered_task_list:
                        considered_task_list.append(task_1)
                resource_list = get_results_from_keys(task_list, filtered_ability_map)
                task_list = get_results_from_keys(resource_list, reverse_filtered_ability_map)
            fix_resource_list = resource_list
            fix_task_list = task_list
            cluster_agent_list.append(fix_resource_list)
            cluster_task_list.append(fix_task_list)
            # add fixed resource into considered_resource_list
            for considered_resource in fix_resource_list:
                if considered_resource not in considered_resource_list:
                    considered_resource_list.append(considered_resource)
            if len(considered_resource_list) == len(diff_resource_list):
                return cluster_agent_list, cluster_task_list
    # cluster other resources into these clustered groups
    if len(considered_resource_list) < len(diff_resource_list):
        reverse_ability_group = get_reverse_ability_map(ability_map)
        for agent in reverse_ability_group:
            if agent not in considered_resource_list:
                max_same_task_freq = -1
                max_index = -1
                max_task_list = []
                agent_task_list = reverse_ability_group[agent]
                for index in range(len(cluster_agent_list)):
                    curr_same_task_freq = 0
                    for task_2 in reverse_ability_group[agent]:
                        if task_2 in cluster_task_list[index]:
                            curr_same_task_freq += frequency_map[(task_2, agent)]
                    if max(max_same_task_freq, curr_same_task_freq) == curr_same_task_freq:
                        max_same_task_freq = curr_same_task_freq
                        max_index = index
                if max_same_task_freq == 0 or max_index == -1:
                    cluster_agent_list.append([agent])
                    cluster_task_list.append(agent_task_list)
                else:
                    cluster_agent_list[max_index].append(agent)
                    for task_3 in agent_task_list:
                        if task_3 not in cluster_task_list[max_index]:
                            cluster_task_list.append(task_3)
                    
    return cluster_agent_list, cluster_task_list

# 3. business models enhancement through discovery of roles
import pm4py.algo.discovery.causal.algorithm as process_discovery

def BME_first_step(log, t1, t2):
    # resource_action_freq_dic = {(task_1, task_2):{(agent_1, agent_2):4, ...}, ...}
    def get_resource_action_freq_dic(log, dfg, t1):
        final_dic = {}
        for depend_tasks in dfg:
            if dfg[depend_tasks] > t1:
                # meet the requirement, continue
                # inner_dic = {(agent_1, agent_2):4, ...}
                inner_dic = {}
                for trace in log:
                    for index in range(len(trace)-1):
                        curr_task = trace[index]['activity_type']
                        next_task = trace[index+1]['activity_type']
                        if (curr_task, next_task) == depend_tasks:
                            curr_resource = trace[index]['agent_id']
                            next_resource = trace[index+1]['agent_id']
                            if (curr_resource, next_resource) not in inner_dic:
                                inner_dic[(curr_resource, next_resource)] = 1
                            else:
                                inner_dic[(curr_resource, next_resource)] += 1
                final_dic[depend_tasks] = inner_dic
        return final_dic
    
    def get_weight_value(depend_agent_dictionary):
        first_agent_freq_dic = {}
        second_agent_freq_dic = {}
        # frac = nominator / denominator
        denominator = 0
        first_part_nominator = 0
        second_part_nominator = 0
        for depend_agent in depend_agent_dictionary:
            if depend_agent[0] not in first_agent_freq_dic:
                first_agent_freq_dic[depend_agent[0]] = 1
            else:
                first_agent_freq_dic[depend_agent[0]] += 1
            if depend_agent[1] not in second_agent_freq_dic:
                second_agent_freq_dic[depend_agent[1]] = 1
            else:
                second_agent_freq_dic[depend_agent[1]] += 1
            if depend_agent[0] == depend_agent[1]:
                second_part_nominator += depend_agent_dictionary[depend_agent]
            denominator += depend_agent_dictionary[depend_agent] * 2
        # calculate first_part_nominator
        for first_agent in first_agent_freq_dic:
            first_agent_freq = first_agent_freq_dic[first_agent]
            if first_agent in second_agent_freq_dic:
                second_agent_freq = second_agent_freq_dic[first_agent]
                first_part_nominator += min(first_agent_freq, second_agent_freq)
        if denominator == 0:
            return 0
        else:
            weight_value = (first_part_nominator + second_part_nominator) / denominator
            return weight_value
        
    dfg_pm4py, start_activities, end_activities = pm4py.discover_directly_follows_graph(log)
    dfg = process_discovery.apply(dfg_pm4py, variant=process_discovery.Variants.CAUSAL_HEURISTIC)
    resource_action_freq_dic = get_resource_action_freq_dic(log, dfg, t1)
    depend_task_list = []
    for double_actions in resource_action_freq_dic:
        depend_agent_dictionary = resource_action_freq_dic[double_actions]
        if get_weight_value(depend_agent_dictionary) > t2:
            depend_task_list.append(double_actions)
    return depend_task_list, resource_action_freq_dic


def BME_second_step(depend_task_list, ability_map, frequency_map, t1):
    # according to depend_task_list, divide the tasks into a few groups
    def get_clusters(depend_task_list):
        task_clusters = []
        completed_tasks = []
        for double_tasks in depend_task_list:
            if double_tasks[0] not in completed_tasks and double_tasks[1] not in completed_tasks:
                new_cluster = []
                if double_tasks[0] == double_tasks[1]:
                    new_cluster.append(double_tasks[0])
                else:
                    new_cluster.append(double_tasks[0])
                    new_cluster.append(double_tasks[1])
                new_cluster_len = len(new_cluster)
                newer_cluster_len = -1
                while (new_cluster_len != newer_cluster_len):
                    new_cluster_len = newer_cluster_len
                    for other_tasks in depend_task_list:
                        if other_tasks[0] in new_cluster or other_tasks[1] in new_cluster:
                            if other_tasks[0] not in new_cluster:
                                new_cluster.append(other_tasks[0])
                            if other_tasks[1] not in new_cluster:
                                new_cluster.append(other_tasks[1])
                    newer_cluster_len = len(new_cluster)
                task_clusters.append(new_cluster)
                for task in new_cluster:
                    if task not in completed_tasks:
                        completed_tasks.append(task)
        return task_clusters
    
    def merge_value_calculation(first_task_cluster, second_task_cluster, ability_map, frequency_map):
        # frac = nominator / denominator
        nominator = 0
        denominator = 0
        for first_task in first_task_cluster:
            for first_resource in ability_map[first_task]:
                denominator += frequency_map[(first_task, first_resource)]
                first_freq = frequency_map[(first_task, first_resource)]
                for second_task in second_task_cluster:
                    for second_resource in ability_map[second_task]:
                        if second_resource == first_resource:
                            second_freq = frequency_map[(second_task, second_resource)]
                            nominator += min(first_freq, second_freq)
        for second_task in second_task_cluster:
            for second_resource in ability_map[second_task]:
                denominator += frequency_map[(second_task, second_resource)]
        if denominator == 0:
            return 0
        else:
            final_value = nominator / denominator
            return final_value
    
    def merge_clusters(cluster1, cluster2, task_clusters):
        two_clusters = []
        for element in cluster1:
            if element not in two_clusters:
                two_clusters.append(element)
        for element in cluster2:
            if element not in two_clusters:
                two_clusters.append(element)
        new_clusters = []
        new_clusters.append(two_clusters)
        for cluster in task_clusters:
            if cluster != cluster1 and cluster != cluster2:
                new_clusters.append(cluster)
        return new_clusters

    task_clusters = get_clusters(depend_task_list)
    original_clusters = task_clusters
    go_next_round = True
    first_target_cluster = ''
    second_target_cluster = ''
    while go_next_round == True:
        largest_value = -1
        for i in range(len(task_clusters)-1):
            for j in range(i+1, len(task_clusters)):
                first_task_cluster = task_clusters[i]
                second_task_cluster = task_clusters[j]
                curr_value = merge_value_calculation(first_task_cluster, second_task_cluster, ability_map, frequency_map)
                if curr_value > largest_value:
                    largest_value = curr_value
                    first_target_cluster = first_task_cluster
                    second_target_cluster = second_task_cluster
        if largest_value <= t1:
            go_next_round = False
        else:
            task_clusters = merge_clusters(first_target_cluster, second_target_cluster, task_clusters)
        
    return task_clusters, original_clusters

def task_cluster_to_agent_cluster(task_clusters, diff_resource_list, ability_map, frequency_map):
    result_list = [[] for i in range(len(task_clusters))]
    for resource in diff_resource_list:
        largest_freq = 0
        largest_cluster_index = ''
        for index in range(len(task_clusters)):
            task_cluster = task_clusters[index]
            curr_freq = 0
            for task in task_cluster:
                if (task, resource) in frequency_map:
                    curr_freq += frequency_map[(task, resource)]
            if curr_freq > largest_freq:
                largest_freq = curr_freq
                largest_cluster_index = index
        if largest_freq > 0:
            result_list[largest_cluster_index].append(resource)
        else:
            result_list.append([resource])
    new_result_list = []
    for result in result_list:
        if result != []:
            new_result_list.append(result)
    return new_result_list

# 4. activity matrix
def get_activity_matrix(diff_task_list, diff_resource_list, frequency_map):
    # activity_matrix_dic = {agent_1: [0 ,4, 8, ...], agent_2: [...], ...} 
    # elements in [0, 4, 8, ...] is the same task order as in diff_task_list
    activity_matrix_dic = {}
    for resource in diff_resource_list:
        task_freq_list = []
        for task in diff_task_list:
            if (task, resource) in frequency_map:
                task_freq_list.append(frequency_map[(task, resource)])
            else:
                task_freq_list.append(0)
        activity_matrix_dic[resource] = task_freq_list
    return activity_matrix_dic

def get_correlation_value(activity_matrix, first_resource, second_resource):
    x_list = activity_matrix[first_resource]
    y_list = activity_matrix[second_resource]
    # frac = (nominator_1 - nominator_2) / (denominator_1 * denominator_2)
    n = len(x_list)
    nominator_2 = sum(x_list) * sum(y_list)
    sum_x_times_y = 0
    sum_x_square = 0
    sum_y_square = 0
    for index in range(len(x_list)):
        x = x_list[index]
        y = y_list[index]
        sum_x_times_y += (x * y)
        sum_x_square += x * x
        sum_y_square += y * y
    nominator_1 = n * sum_x_times_y
    denominator_1 = math.sqrt(n * sum_x_square - sum(x_list) * sum(x_list))
    denominator_2 = math.sqrt(n * sum_y_square - sum(y_list) * sum(y_list))
    if denominator_1 * denominator_2 == 0:
        return 0
    else:
        correlation_value = (nominator_1 - nominator_2) / (denominator_1 * denominator_2)
        return correlation_value

def get_relation_tuple_list(activity_matrix, t1):
    # relation_tuple_list = [(agent_1, agent_2), (agent_1, agent_3), ...]
    relation_tuple_list = []
    for first_resource in activity_matrix:
        for second_resource in activity_matrix:
            if first_resource != second_resource:
                correlation_value = get_correlation_value(activity_matrix, first_resource, second_resource)
                if correlation_value >= t1:
                    if (first_resource, second_resource) not in relation_tuple_list:
                        relation_tuple_list.append((first_resource, second_resource))
    return relation_tuple_list

def get_resource_clusters_list(relation_tuple_list, diff_resource_list):
    completed_resource_list = []
    resource_clusters_list = []
    for resource in diff_resource_list:
        if resource not in completed_resource_list:
            resource_cluster = []
            resource_cluster.append(resource)
            new_cluster_len = len(resource_cluster)
            cluster_len = -1
            while (new_cluster_len != cluster_len):
                cluster_len = new_cluster_len
                for relation_tuple in relation_tuple_list:
                    resource_1 = relation_tuple[0]
                    resource_2 = relation_tuple[1]
                    if resource_1 in resource_cluster or resource_2 in resource_cluster:
                        if resource_1 not in resource_cluster:
                            resource_cluster.append(resource_1)
                        if resource_2 not in resource_cluster:
                            resource_cluster.append(resource_2)
                new_cluster_len = len(resource_cluster)
            resource_clusters_list.append(resource_cluster)
            for completed_resource in resource_cluster:
                if completed_resource not in completed_resource_list:
                    completed_resource_list.append(completed_resource)
    return resource_clusters_list

def save_cluster_information_csv(inst_cluster_map, config):
    # BME method
    value = config._cluster_alg_threshold
    if type(value) == tuple:
        part_file_name = 'cluster_threshold' + '_' + str(int(value[0]*100))+'_'+str(int(value[1]*100))+'_'+str(int(value[2]*100))
    # other methods
    else:
        part_file_name = 'cluster_threshold' + '_' + str(int(value*100))
    config._part_file_name = part_file_name
    file_path = os.path.join(os.getcwd(), config._combine_dir, config._filter_ver, config._cluster_alg, part_file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    agent_resource_dictionary = {}
    for resource in inst_cluster_map:
        if inst_cluster_map[resource] not in agent_resource_dictionary:
            agent_resource_dictionary[inst_cluster_map[resource]] = [resource]
        else:
            agent_resource_dictionary[inst_cluster_map[resource]].append(resource)
    output_dict = {'agent id':[], 'resource id':[]}
    for agent in agent_resource_dictionary:
        output_dict['agent id'].append(agent)
        output_dict['resource id'].append(agent_resource_dictionary[agent])
    df = pd.DataFrame.from_dict(output_dict)
    final_file_path = os.path.join(file_path, 'cluster_groups.csv')
    df.to_csv(final_file_path)

def run_cluster(config, log_df, unclustered_log):
    resource_action_dic, diff_resource_list, diff_task_list = get_information_from_log(unclustered_log)
    if config._cluster_alg == 'AgentMiner':
        inst_cluster_map = run_AgentMiner(log_df, config._cluster_alg_threshold)
    if config._cluster_alg == 'CWM':
        ability_map, frequency_map = get_ability_freq_maps(resource_action_dic)
        cluster_agent_list, cluster_task_list = CWM_cluster(ability_map, frequency_map, diff_resource_list, config._cluster_alg_threshold)
        inst_cluster_map = get_inst_cluster_map(cluster_agent_list)
    if config._cluster_alg == 'BME':
        ability_map, frequency_map = get_ability_freq_maps(resource_action_dic)
        depend_task_list, resource_action_freq_dic = BME_first_step(unclustered_log, config._cluster_alg_threshold[0], config._cluster_alg_threshold[1])
        task_clusters, original_clusters = BME_second_step(depend_task_list, ability_map, frequency_map, config._cluster_alg_threshold[2])
        cluster_agent_list = task_cluster_to_agent_cluster(task_clusters, diff_resource_list, ability_map, frequency_map)
        inst_cluster_map = get_inst_cluster_map(cluster_agent_list)
    if config._cluster_alg == 'ActivityMatrix':
        ability_map, frequency_map = get_ability_freq_maps(resource_action_dic)
        activity_matrix = get_activity_matrix(diff_task_list, diff_resource_list, frequency_map)
        relation_tuple_list = get_relation_tuple_list(activity_matrix, config._cluster_alg_threshold)
        cluster_agent_list = get_resource_clusters_list(relation_tuple_list, diff_resource_list)
        inst_cluster_map = get_inst_cluster_map(cluster_agent_list)
    save_cluster_information_csv(inst_cluster_map, config)
    return inst_cluster_map
