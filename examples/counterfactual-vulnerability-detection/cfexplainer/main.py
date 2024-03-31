import os
import gc
import json
import random
import argparse
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup

from models.vul_detector import Detector
from helpers import utils
from line_extract import get_dep_add_lines_bigvul
from graph_dataset import VulGraphDataset, collate
from models.gnnexplainer import XGNNExplainer
from models.cfexplainer import CFExplainer
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM


warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary'), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary'), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary'), 4),
    }
    return results
     

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
    
def train(args, train_dataloader, valid_dataloader, test_dataloader, model):

    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    checkpoint_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device))
    
    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataloader)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Total optimization steps = {}".format(args.max_steps))
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0

    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch_data in enumerate(bar):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            # labels = global_max_pool(batch_data._VULN, batch).long()
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            model.train()
            probs = model(x, edge_index, batch)
            labels = F.one_hot(1 - labels, 2)
            loss = F.binary_cross_entropy(probs, labels.float())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    results = evaluate(args, valid_dataloader, model)
                    print(f"  Valid acc:{results['eval_acc']}")
                    
                    # Save model checkpoint
                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        print("  " + "*" * 20)  
                        print("  Best acc:{}".format(round(best_acc, 4)))
                        print("  " + "*" * 20)
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to {}".format(output_dir))

                        test_result = evaluate(args, test_dataloader, model)
                        for key, value in test_result.items():
                            print("  {} = {}".format(key, round(value, 4)))
        bar.close()


def evaluate(args, eval_dataloader, model):

    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataloader)))
    print("  Batch size = {}".format(args.batch_size))

    model.eval()
    all_probs = [] 
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            # labels = global_max_pool(batch_data._VULN, batch).long()
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            probs = model(x, edge_index, batch)
            probs = F.one_hot(torch.argmax(probs, dim=-1), 2)[:, 0]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)
    eval_acc = np.mean(all_labels == all_probs)

    result = {
        "eval_acc": round(eval_acc, 4),
    }
    
    eval_results = calculate_metrics(all_labels, all_probs)
    result.update(eval_results)
    
    return result


def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]
    
    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)
    
    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8
    
    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in
    
    ret = sorted(
        list(
            zip(
                line_importance.squeeze(-1).cpu().numpy(),
                lines,
            )
        ),
        reverse=True,
    )
    
    filtered_ret = []
    for i in ret:
        if i[0] > 0:
            filtered_ret.append(int(i[1]))

    return filtered_ret


def eval_exp(exp_saved_path, model, correct_lines, args):
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))
    
    accuracy = 0
    precisions = []
    recalls = []
    F1s = []
    pn = []
    for graph in graph_exp_list:
        graph.to(args.device)
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        exp_label_lines = correct_lines[int(sampleid)]
        # exp_label_lines = list(exp_label_lines["removed"]) + list(exp_label_lines["depadd"])
        exp_label_lines = list(exp_label_lines["removed"])
        if len(edge_weight) > args.KM:
            value, index = torch.topk(edge_weight, k=args.KM)
        else:
            index = torch.arange(edge_weight.shape[0])
        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = temp != 0
        
        lines = graph._LINE.cpu().numpy()
        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)
        
        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                accuracy += 1
                break
            
        hit = 0
        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                hit += 1
        if hit != 0:
            precision = hit / len(exp_lines)
            recall = hit / len(exp_label_lines)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)
        
        fac_edge_index = edge_index[:, index]
        fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])  # add self-loop
        fac_logits = model(x, fac_edge_index, batch)
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][0]
        
        cf_edge_index = edge_index[:, cf_index]
        cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])  # add self-loop
        cf_logits = model(x, cf_edge_index, batch)
        cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][0]
        
        pn.append(int(cf_pred != pred))
                
        if args.case_sample_ids and str(sampleid) in args.case_sample_ids:
            case_saving_dir = str(utils.cache_dir() / f"cases")
            case_graph_saving_path = f"{case_saving_dir}/{args.gnn_model}_{args.ipt_method}_{sampleid}.pt"
            torch.save(graph, case_graph_saving_path)
            print(f"Saving {str(sampleid)} in {case_graph_saving_path}!")

    accuracy = round(accuracy / len(graph_exp_list), 4)
    print("Accuracy:", accuracy)
    precision = round(np.mean(precisions), 4)
    print("Precision:", precision)
    recall = round(np.mean(recalls), 4)
    print("Recall:", recall)
    f1 = round(np.mean(F1s), 4)
    print("F1:", f1)
    PN = round(sum(pn) / len(pn), 4)
    print("Probability of Necessity:", PN)
    
    if args.hyper_para:
        para_saving_dir = str(utils.cache_dir() / f"parameter_analysis")
        if not os.path.exists(para_saving_dir):
            os.makedirs(para_saving_dir)
        if args.ipt_method == "cfexplainer":
            if args.cfexp_L1:
                para_saving_path = os.path.join(para_saving_dir, f"{args.ipt_method}_L1_{args.cfexp_alpha}.res")
            else:
                para_saving_path = os.path.join(para_saving_dir, f"{args.ipt_method}_{args.cfexp_alpha}.res")
        KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
        if os.path.isfile(para_saving_path):
            result = json.load(open(para_saving_path, "r"))
        else:
            GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
            metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
            result = {}
            for GNN_model in GNN_models:
                result[GNN_model] = {}
                for metric in metrics:
                    result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
        result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
        result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
        result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
        result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
        json.dump(result, open(para_saving_path, "w"))
    else:
        results_saving_dir = str(utils.cache_dir() / f"results")
        if not os.path.exists(results_saving_dir):
            os.makedirs(results_saving_dir)
        results_saving_path = os.path.join(results_saving_dir, f"{args.ipt_method}.res")
        KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
        if os.path.isfile(results_saving_path):
            result = json.load(open(results_saving_path, "r"))
        else:
            GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
            metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
            result = {}
            for GNN_model in GNN_models:
                result[GNN_model] = {}
                for metric in metrics:
                    result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
        result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
        result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
        result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
        result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
        json.dump(result, open(results_saving_path, "w"))


def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = XGNNExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05, 
        coff_edge_size=0.001, coff_edge_ent=0.001
    )
    explainer.device = args.device
    
    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def cfexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = CFExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05, alpha=args.cfexp_alpha, L1_dist=args.cfexp_L1
    )
    explainer.device = args.device
    
    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None, num_classes=args.num_classes)
        edge_weight = 1 - edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    input_dim = args.gnn_hidden_size * 2
    
    pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device, explain_graph=True, epochs=100, lr=0.005,
                            coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
    pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
    if os.path.isfile(pgexplainer_saving_path) and not args.ipt_update:
        print("Load saved PGExplainer model...")
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    else:
        pgexplainer.train_explanation_network(train_dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    
    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model)
    pgexplainer_edges.device = pgexplainer.device
    
    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = pgexplainer_edges(x, edge_index, num_classes=args.num_classes, sparsity=0.5)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def subgraphx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    
    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                        verbose=False, c_puct=10.0, rollout=5, high2low=False, min_atoms=5, expand_atoms=14,
                        reward_method='gnn_score', subgraph_building_method='zero_filling',
                        save_dir=explanation_saving_dir)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        saved_MCTSInfo_list = None
        prediction = prob.argmax(-1).item()
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        explain_result = subgraphx.explain(x, edge_index, label=prediction, node_idx=0, saved_MCTSInfo_list=saved_MCTSInfo_list)
        torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            node_weight[item['coalition']] += item['P']
        node_weight = node_weight / len(explain_result)
        edge_index, _ = remove_self_loops(edge_index.detach().cpu())
        edge_weight = node_weight[edge_index[0]] + node_weight[edge_index[1]]
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def gnn_lrp_run(args, model, test_dataset, correct_lines):
    # for name, parameter in model.named_parameters():
    #     print(name)
    
    graph_exp_list = []
    visited_sampleids = set()
    
    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/gnn_lrp")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    gnnlrp_explainer = GNN_LRP(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)
        
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            edge_masks, self_loop_edge_index = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        else:
            walks, edge_masks, related_preds, self_loop_edge_index = gnnlrp_explainer(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
            torch.save((edge_masks, self_loop_edge_index), os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph.detach().clone().cpu())
        visited_sampleids.add(sampleid)
        
        del graph
        gc.collect()
        
    return graph_exp_list


def deeplift_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    deep_lift = DeepLIFT(model, explain_graph=True)
    
    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)
        
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = deep_lift(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def gradcam_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    gc_explainer = GradCAM(model, explain_graph=True)
    
    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)
        
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = gc_explainer(x, edge_index, sparsity=0.5, num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index
        
        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)
        
    return graph_exp_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    # GNN Model
    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--gnn_model", default="GCNConv", type=str,
                        help="GNN core.")
    parser.add_argument("--gnn_hidden_size", default=256, type=int,
                        help="hidden size of gnn.")
    parser.add_argument("--gnn_feature_dim_size", default=768, type=int,
                        help="feature dim size of gnn.")
    parser.add_argument("--residual", action='store_true',
                        help="Whether to obtain residual representations.")
    parser.add_argument("--graph_pooling", default="mean", type=str,
                        help="The operator of graph pooling.")
    parser.add_argument("--num_gnn_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_ggnn_steps", default=3, type=int,
                        help="The sequence length for GGNN.")
    parser.add_argument("--ggnn_aggr", default="add", type=str,
                        help="The aggregation scheme to use for GGNN.")
    parser.add_argument("--gin_eps", default=0., type=float,
                        help="Eps value for GIN.")
    parser.add_argument("--gin_train_eps", action='store_true',
                        help="If set to True, eps will be a trainable parameter.")
    parser.add_argument("--gconv_aggr", default="mean", type=str,
                        help="The aggregation scheme to use.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout rate.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    
    # Training
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_explain", action='store_true',
                        help="Whether to run explaining.")
    
    # Explainer
    parser.add_argument("--ipt_method", default="gnnexplainer", type=str,
                        help="The save path of interpretations.")
    parser.add_argument("--ipt_update", action='store_true',
                        help="Whether to update interpretations.")
    parser.add_argument("--KM", default=8, type=int,
                        help="The size of explanation subgraph.")
    parser.add_argument("--cfexp_L1", action='store_true',
                        help="Whether to use L1 distance item.")
    parser.add_argument("--cfexp_alpha", default=0.9, type=float,
                        help="CFExplainer.")
    parser.add_argument("--hyper_para", action='store_true',
                        help="Whether to tune the hyper-parameters.")
    parser.add_argument("--case_sample_ids", nargs='+',
                        help="Ids of samples to extract for case study.")

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)
    set_seed(args.seed)
    
    args.start_epoch = 0
    args.start_step = 0

    model = Detector(args)
    model.to(args.device)
    
    train_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    print(train_dataset)
    
    valid_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='val')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
    print(valid_dataset)
    
    test_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(test_dataset)
    
    if args.do_train:
        train(args, train_dataloader, valid_dataloader, test_dataloader, model)
    
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))                  
        model.to(args.device)
        test_result = evaluate(args, test_dataloader, model)

        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))
            
        if args.do_explain:
            correct_lines = get_dep_add_lines_bigvul()
            ipt_save_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}")
            if not os.path.exists(ipt_save_dir):
                os.makedirs(ipt_save_dir)
            if args.hyper_para:
                if args.ipt_method == "cfexplainer":
                    if args.cfexp_L1:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_L1_{args.cfexp_alpha}.pt")
                    else:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_{args.cfexp_alpha}.pt")
            else:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}.pt")
            print("Size of test dataset:", len(test_dataset))
            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if not os.path.exists(ipt_save) or args.ipt_update:
                graph_exp_list = []
                if args.ipt_method == "pgexplainer":
                    eval_model = Detector(args)
                    eval_model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
                    eval_model.to(args.device)
                    graph_exp_list = pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines)
                elif args.ipt_method == "subgraphx":
                    graph_exp_list = subgraphx_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnn_lrp":
                    graph_exp_list = gnn_lrp_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "deeplift":
                    graph_exp_list = deeplift_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gradcam":
                    graph_exp_list = gradcam_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "cfexplainer":
                    graph_exp_list = cfexplainer_run(args, model, test_dataset, correct_lines)
                    
                torch.save(graph_exp_list, ipt_save)
            
            eval_exp(ipt_save, model, correct_lines, args)
            

if __name__ == "__main__":
    main()
