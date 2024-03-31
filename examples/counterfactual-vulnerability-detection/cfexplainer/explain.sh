#!/bin/bash


for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gnnexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method cfexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method pgexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method subgraphx --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gnn_lrp --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method deeplift --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gradcam --KM $KM --cuda_id 0
done


for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gnnexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method cfexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method pgexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method subgraphx --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gnn_lrp --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method deeplift --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gradcam --KM $KM --cuda_id 0
done


for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gnnexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method cfexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method pgexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method subgraphx --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gnn_lrp --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method deeplift --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gradcam --KM $KM --cuda_id 0
done


for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gnnexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method cfexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method pgexplainer --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method subgraphx --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gnn_lrp --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method deeplift --KM $KM --cuda_id 0
done

for KM in {2..20..2}  
do  
    python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gradcam --KM $KM --cuda_id 0
done
