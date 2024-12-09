import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ratsql.utils.analysis import cal_attention_flow
from ratsql.commands.infer import Inferer

import _jsonnet

# Path template
template = "/home/hkkang/NL2QGM/logdir/spider_bert_run_no_join_cond_seed_{}/bs=8,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,seed={},join_cond=false/ie_dirs/bert_run_true_1-step_41600-eval.json"

eval_paths = [template.format(0,0)]
# eval_paths += [template.format(2,2)]
# eval_paths += [template.format(3,3)]

infer_paths = [path.replace("-eval.json", "-infer.jsonl") for path in eval_paths]
debug_paths = [path.replace("-eval.json", "-debug.jsonl") for path in eval_paths]

def is_int(text):
    try:
        int(text)
        return True
    except:
        return False

def load_json_custom(path):
    result = json.load(open(path))['per_item']
    print(len(result))
    return result

def load_jsonl(path):
    with open(path, 'r') as f:
        results = [json.loads(line) for line in f.readlines()]
    return results
    
def load_model(seed):
    exp_config_file = 'experiments/spider-bert-run-no-join.jsonnet'
    exp_config = json.loads(_jsonnet.evaluate_file(exp_config_file))
    # Outer most arugments
    model_config_args = exp_config['model_config_args']
    model_config_args['seed'] = seed
    # Default arguments
    model_config_path = exp_config['model_config']
    config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))
    # Create model
    inferer = Inferer(config)
    # Set logdir
    assert is_int(exp_config['logdir'].split('_')[-1])
    logdir = '_'.join(exp_config['logdir'].split('_')[:-1] + [str(seed)])
    logdir = os.path.join(logdir, config['model_name'])
    # Load model
    model = inferer.load_model(logdir)
    relation_ids = model.encoder.encs_update.relation_ids
    encoder = model.encoder.encs_update.encoder.layers # each layer has different relation embeddings
    raise NotImplementedError("Not yet")

def get(t_list, idx):
    return [item[idx] for item in t_list]
    
def get_consistently_wrong_indices(eval_list):
    same_lengths = [len(eval_list[i-1]) == len(eval_list[i]) for i in range(1, len(eval_list))]
    assert False not in same_lengths
    indices = []
    for idx in range(len(eval_list[0])):
        correctness = [result[idx]['exact'] for result in eval_list]
        # Incorrect in all models
        if True not in correctness:
            indices += [idx]
    return indices

def get_info_list():
    eval_list = [load_json_custom(path) for path in eval_paths]
    debug_list = [load_jsonl(path) for path in debug_paths]
    infer_list = [load_jsonl(path) for path in infer_paths]
    wrong_indices = get_consistently_wrong_indices(eval_list)

    return wrong_indices, eval_list, debug_list, infer_list

def draw_question_schema_link(question, schema, match_info, title=None):
    def parse_item(key, value):
        idx1, idx2 = [int(item) for item in key.split(',')]
        if value[-2:] == 'EM':
            value = 1
        elif value[-2:] == 'PM':
            value = 0.5
        elif value in ['CELLMATCH', 'NUMBER', 'TIME']:
            value = 1
        else:
            raise RuntimeError("should not be here!")
        return idx1, idx2, value

    weight_matrix = torch.zeros(len(question), len(schema))
    infos = [parse_item(key, value) for key, value in match_info.items()]
    for idx1, idx2, value in infos:
        weight_matrix[idx1][idx2] = value

    visualize_attention(weight_matrix, question, schema, title=title)    
    
def visualize_attention(mma, target_labels, source_labels, title=None):
    """
    Inputs:
        mma: nxn weight matrix
        source_labels: List of column labels
        target_labels: List of row labels
    """
    fig, ax = plt.subplots(figsize=(20,20), dpi=100)
    im = ax.imshow(mma)
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(mma.shape[1]), minor=False) # mma.shape[1] = target seq 길이
    ax.set_yticks(np.arange(mma.shape[0]), minor=False) # mma.shape[0] = input seq 길이
   
    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)
  
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
 
    plt.xticks(rotation=45)
 
    if title:
        ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mma)):
        for j in range(len(mma[0])):
            text = ax.text(j, i, "{:.3f}".format(mma[i, j].item()), ha="center", va="center", color="k")
        
    plt.tight_layout()
    plt.show()

# Analysis
def analyze(eval_results, debug_results, infer_results, model_idx=0, inspect_step_idx=None):
    # Keep other models' info within this function for detailed comparison
    eval_result = eval_results[model_idx]
    debug_result = debug_results[model_idx]
    infer_result = infer_results[model_idx]

    inference_identical = False not in [eval_results[i-1]['predicted'] == eval_results[i]['predicted'] for i in range(1, len(eval_results))]
    print(f"Inference identical:{inference_identical}")

    # For easy referencing
    db_id = infer_result['db_id']
    nl = infer_result['question']
    nl_toks = infer_result['question_toks']
    gold = eval_result['gold']
    pred = eval_result['predicted']
    ## Schema and memory
    tables = [' '.join(item) for item in infer_result['schema']['tables']]
    columns = [' '.join(item[:-1]) for item in infer_result['schema']['columns']]
    memory = nl_toks+columns+tables
    ## Schema linking
    sc_link = infer_result['sc_link']
    cv_link = infer_result['cv_link']
    ## align_mat
    mc_align_matrix = torch.tensor(debug_result['mc_align_matrix'])
    mt_align_matrix = torch.tensor(debug_result['mt_align_matrix'])
    ## action scores
    decode_history = debug_result['history']
    ## encoder attention weight for each layers
    atts_list = debug_result['att_list']

    # Print Basic info
    print(f"DB_ID: {db_id}")
    print(f"NL: {nl}")
    print(f"GOLD: {gold}")
    print(f"PRED: {pred}")
    
    # Analyze Linking
    ## question - column linking
    draw_question_schema_link(nl_toks, columns, sc_link['q_col_match'], title="Question - Column linking")
    ## question - table linking
    draw_question_schema_link(nl_toks, tables, sc_link['q_tab_match'], title="Question - Table linking")
    ## question - column cell linking
    draw_question_schema_link(nl_toks, columns, dict(cv_link['cell_match'], **cv_link['num_date_match']),
                             title="Question - Cell linking") 

    # Analyze align matrix
    visualize_attention(mc_align_matrix, memory, columns, title="Memory - Column alignment")
    visualize_attention(mt_align_matrix, memory, tables, title="Memory - Table alignment")
    
    # Analyze attention flow in encoder
    att_flow = cal_attention_flow(torch.tensor(atts_list).squeeze(1))
    visualize_attention(att_flow, memory, memory, title="Encoder attention flow")
    
    # Analyze decoding steps
    for step, step_info in enumerate(decode_history):
        if inspect_step_idx != None and inspect_step_idx != step:
            continue
        print(f"Step: {step}")
        # For easy reference
        rule_left = step_info['rule_left']
        choices = step_info['choices']
        probs = step_info['probs']

        # Decoder: action choices and probs
        print(f"rule_left: {rule_left}")
        print(f"choices: {choices}")
        print(f"probs: {['{:.2f}'.format(prob*100) for prob in probs]}")

        # Decoder: hidden state - memory attention
        dec_att = torch.tensor(step_info['att_probs'])
        visualize_attention(dec_att.transpose(0,1), memory, ['hidden_state'],
                                title="Hidden state - Memory attention")

        # More info for column/table
        if rule_left in ['column', 'table']:
            # Decoder: memory-pointer probs
            memory_pointer_probs = torch.tensor(step_info['memory_pointer_probs'])
            visualize_attention(memory_pointer_probs.transpose(0, 1), 
                                memory, ['hidden_state'], title='Memory pointer probs')
            
            ## Decoder: attention flow of inputs to schema candidates
            # 1. alignment matrix and encoder attention flow
            if rule_left == 'column':
                target = columns
                align_mat = mc_align_matrix
                schema_att_flow = att_flow[len(nl_toks):len(nl_toks+columns)]
            else:
                target = tables
                align_mat = mt_align_matrix
                schema_att_flow = att_flow[len(nl_toks+columns):]
            align_mat = align_mat.unsqueeze(-1).repeat(1,1,len(memory))
            schema_att_flow = schema_att_flow.unsqueeze(0).repeat(len(memory), 1, 1)
            output1 = align_mat * schema_att_flow # (len_memory x len_schema x len_memory)
        
            # 2. hidden attention and output from (1)
            dec_att = dec_att.squeeze(0).reshape(len(memory), 1, 1).repeat(1, len(target), len(memory))
            att_flow_to_target_schema = torch.sum(dec_att * output1, dim=0)
            visualize_attention(att_flow_to_target_schema, target, memory,
                                   title=f"Attention flow to output {rule_left}")

def api(wrong_indices, eval_list, debug_list, infer_list, model_idx=0, wrong_idx=0, step_idx=None):
        item_idx = wrong_indices[wrong_idx]
        print(len(eval_list))
        eval_results = get(eval_list, item_idx)
        debug_results = get(debug_list, item_idx)
        infer_results = get(infer_list, item_idx)
        analyze(eval_results, debug_results, infer_results, model_idx=model_idx, inspect_step_idx=step_idx)


if __name__ == "__main__":
    wrong_indices, eval_list, debug_list, infer_list = get_info_list()
    print(f"Consistenly wrong count:{len(wrong_indices)}")
    api(wrong_indices, eval_list, debug_list, infer_list, model_idx=0, wrong_idx=0)
