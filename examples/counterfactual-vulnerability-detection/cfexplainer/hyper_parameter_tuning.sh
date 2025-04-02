#!/bin/bash


alphas=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )

for alpha in "${alphas[@]}"
do
    for KM in {2..20..2}  
    do  
        python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method cfexplainer --KM $KM --cuda_id 0 --hyper_para --cfexp_alpha $alpha
    done
done

for alpha in "${alphas[@]}"
do
    for KM in {2..20..2}  
    do  
        python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method cfexplainer --KM $KM --cuda_id 0 --hyper_para --cfexp_alpha $alpha
    done
done

for alpha in "${alphas[@]}"
do
    for KM in {2..20..2}  
    do  
        python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method cfexplainer --KM $KM --cuda_id 0 --hyper_para --cfexp_alpha $alpha
    done
done

for alpha in "${alphas[@]}"
do
    for KM in {2..20..2}  
    do  
        python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method cfexplainer --KM $KM --cuda_id 0 --hyper_para --cfexp_alpha $alpha
    done
done
