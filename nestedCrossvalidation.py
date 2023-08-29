import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import logging as log
import sys
# test
import probMoE as pm
import utils as ut
import uframe as uf

if __name__ == "__main__":
    
    dataset = "diabetes"
    data_path = r"D:\Github_Projects\Datasets"
    result_path = r"D:\Evaluation_MoE\Evaluation"
    # select setting
    missing = 0.01
    bandwidth = 0.1
    n_folds_outer = 5
    n_folds_inner = 3
    n_experts_max = 5
    lr = 0.01
    reg_lambda = 0.002
    batch_size_experts = 10
    batch_size_gate = 20
    n_epochs = 100
    threshold_samples = 0.6 # our method
    n_samples = 160 # our method
    local_mode = True # our method
    
    # Load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)

    # Scale data
    data_sc = MinMaxScaler().fit_transform(data)
    target = target
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p = missing, bandwidth = bandwidth, mice_iterations=2)
    kwargs = {
          'stepsize':0.3,
          'niter':20,
    }  
    X_object = X.mode(**kwargs)
    #################################################################################################################################
    ################################################## Nested Crossvalidation ########################################################
    #################################################################################################################################
    # Set up logger
    log.basicConfig(
        level=log.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        handlers=[
            log.FileHandler(result_path + '/LOGS.log','w'),
            log.StreamHandler(sys.stdout)
        ])
    logger = log.getLogger()
    logger.info(f"### START OF NESTED CROSSVALIDATION FOR DATASET: {dataset}, MISSING: {missing}, BANDWIDTH: {bandwidth}, THRESHOLD: {threshold_samples} ###")
    # save results for outer
    score_moe_list_total_local = []
    score_moe_list_total_global = []
    indices_outter = np.arange(len(data_sc))
    kf_outer = KFold(n_splits=n_folds_outer, shuffle=True, random_state=42)
    for fold, (train_indices_outer, test_indices_outer) in enumerate(kf_outer.split(X)):
        val_size_outer = len(train_indices_outer) // 3  # 20% of train data for outer validation
        val_indices_outer = train_indices_outer[:val_size_outer]
        train_indices_outer = train_indices_outer[val_size_outer:]
        # Outer split
        X_train_outer = X[train_indices_outer]
        target_train_outer = target[train_indices_outer]
        X_val_outer = X[val_indices_outer]
        target_val_outer = target[val_indices_outer]
        X_test_outer = data_sc[test_indices_outer]
        target_test_outer = target[test_indices_outer]

        # save results for inner Cv
        opt_expert_size_local = None
        opt_expert_size_global = None

        ########################################## Inner Crossvalidation ################################################
        val_loss_local_list_total_inner = []
        val_loss_global_list_total_inner = []
        cluster_accuracies_local_list_total = []
        cluster_accuracies_global_list_total = []
        # Hyperparameter Tuning
        expert_range = range(2, n_experts_max + 1)
        logger.info(f"### START OF INNER CROSSVALIDATON FOR {fold} .FOLD ###")
        for n in expert_range:
            val_loss_local_list = []
            val_loss_global_list = []
            cluster_accuracies_local_list = []
            cluster_accuracies_global_list = []
            kf_inner = KFold(n_splits=n_folds_inner, shuffle=True, random_state=42)  # Inner cross-validation
            for inner_fold, (train_indices_inner, val_indices_inner) in enumerate(kf_inner.split(X_train_outer)):
                X_train_inner = X_train_outer[train_indices_inner]
                target_train_inner = target_train_outer[train_indices_inner]
                X_val_inner = X_train_outer[val_indices_inner]
                target_val_inner = target_train_outer[val_indices_inner]
                ################################################## LOCAL #############################################################
                try:
                    moe = pm.MoE(n, inputsize = input_size, outputsize = output_size)
                    moe.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                            n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                    #################################################  GLOBAL ############################################################
                    moe_glob = pm.MoE(n, inputsize = input_size, outputsize = output_size)
                    moe_glob.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = False, weighted_experts=True, 
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                            n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                    # analyze
                    cluster_accuracies_local, cluster_accuracies_global, loss_weighted_local = moe.analyze(data_sc[train_indices_inner])   
                    _ , _ , loss_weighted_glob = moe_glob.analyze(data_sc[train_indices_inner])     
                    val_loss_global_list.append(loss_weighted_glob)
                    val_loss_local_list.append(loss_weighted_local)
                except:
                    cluster_accuracies_global, cluster_accuracies_local, loss_weighted_local, loss_weighted_glob = np.inf, np.inf, np.inf, np.inf
                val_loss_global_list.append(loss_weighted_glob)
                val_loss_local_list.append(loss_weighted_local)

                # cluster accuracy
                cluster_accuracies_local_list.append(cluster_accuracies_local)
                cluster_accuracies_global_list.append(cluster_accuracies_global)
            
            val_loss_local_list_total_inner.append(np.mean(val_loss_local_list))
            val_loss_global_list_total_inner.append(np.mean(val_loss_global_list))        
            cluster_accuracies_local_list_total.append(np.mean(cluster_accuracies_local_list))
            cluster_accuracies_global_list_total.append(np.mean(cluster_accuracies_global_list))
        ######################################################################################################################  
        ###################################################### Evaluation ###################################################
        ######################################################################################################################
        opt_expert_size_local = ut.find_best_expert(expert_range, val_loss_local_list_total_inner)
        opt_expert_size_global = ut.find_best_expert(expert_range, val_loss_global_list_total_inner)
        logger.info(f"### FINISHED HYPERPARAMETER SEARCH FOR {fold} .FOLD - BEST EXPERT SIZE LOCAL {opt_expert_size_local}, BEST EXPERT SIZE GLOBAL {opt_expert_size_global}  ###")
        logger.info(f"### VAL LOSS LOCAL {val_loss_local_list_total_inner}, VAL LOSS GLOBAL {val_loss_global_list_total_inner} ###")
        ################################################## LOCAL #############################################################
        try:
            moe = pm.MoE(opt_expert_size_local, inputsize = input_size, outputsize = output_size)
            moe.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            moe = pm.MoE(2, inputsize = input_size, outputsize = output_size)
            moe.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = moe.predict(X_test_outer)
        score_moe = moe.evaluation(predictions, target_test_outer) 
        logger.info(f"Prob MoE score Local: {score_moe}")
        score_moe_list_total_local.append(score_moe)
        #################################################  GLOBAL ############################################################
        try:
            moe_glob = pm.MoE(opt_expert_size_global, inputsize = input_size, outputsize = output_size)
            moe_glob.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = False, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            moe_glob = pm.MoE(2, inputsize = input_size, outputsize = output_size)
            moe_glob.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = False, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = moe_glob.predict(X_test_outer)
        score_moe = moe_glob.evaluation(predictions, target_test_outer) 
        logger.info(f"Prob MoE score Global: {score_moe}")
        score_moe_list_total_global.append(score_moe)
       
    logger.info("#############################################################################################")
    logger.info("#############################################################################################")
    logger.info(f"LOCAL TOTAL SCORE: {np.mean(score_moe_list_total_local)}")
    logger.info(f"GLOBAL TOTAL SCORE: {np.mean(score_moe_list_total_global)}")
    logger.info("#############################################################################################")


