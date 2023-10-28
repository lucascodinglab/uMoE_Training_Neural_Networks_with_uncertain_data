import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import logging as log
import sys
import argparse
from Model import uMoE as umoe
from Evaluation import refMoE as moe
from Model import utils as ut
import uframe as uf


"""
This File serves as evaluation file for training uMoE and Baseline Models (MLP and MoE) .
The trainingsprocess includes the Nested Cross-Validation to find the 
optimale number of subspaces/Experts for uMoE and MoE.
"""

# Create an argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Experimental Results with Nested Cross-Validation")
# Define and add command-line arguments
parser.add_argument("--dataset", type=str, default="wine_quality", help="Dataset name")
parser.add_argument("--data_path", type=str, default="D:\Github_Projects\MOE_Training_under_Uncertainty\Datasets", help="Path to data")
parser.add_argument("--result_path", type=str, default="D:\Evaluation", help="Path to results")
parser.add_argument("--missing", type=float, default=0.4, help="Missing data percentage")
parser.add_argument("--bandwidth", type=float, default=0.1, help="Bandwidth value")
parser.add_argument("--verbose", type=bool, default=False, help="Switch for plotting train and val loss")
parser.add_argument("--n_experts_max", type=int, default=6, help="Maximum number of subspaces/experts")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--reg_lambda", type=float, default=0.002, help="Regularization lambda")
parser.add_argument("--batch_size_experts", type=int, default=16, help="Batch size for experts")
parser.add_argument("--batch_size_gate", type=int, default=24, help="Batch size for gating unit")
parser.add_argument("--n_epochs", type=int, default=150, help="Number of epochs")
parser.add_argument("--threshold_samples", type=float, default=0.8, help="Threshold samples")
parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
parser.add_argument("--local_mode", type=bool, default=True, help="Local mode")
parser.add_argument("--inner_fold", type=int, default=3, help="Number of inner Folds for NCV")
parser.add_argument("--outer_fold", type=int, default=5, help="Number of outer Folds for NCV")

if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset = args.dataset
    data_path = args.data_path
    result_path = args.result_path
    missing = args.missing
    bandwidth = args.bandwidth
    verbose = args.verbose
    n_experts_max = args.n_experts_max
    lr = args.lr
    reg_lambda = args.reg_lambda
    batch_size_experts = args.batch_size_experts
    batch_size_gate = args.batch_size_gate
    n_epochs = args.n_epochs
    threshold_samples = args.threshold_samples
    n_samples = args.n_samples
    local_mode = args.local_mode
    n_folds_inner = args.inner_fold
    n_folds_outer = args.outer_fold

    # Load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)

    # Scale data
    data_sc = MinMaxScaler().fit_transform(data)
    target = target
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p = missing, bandwidth = bandwidth, mice_iterations=2)
    kwargs = {
          'stepsize':0.2,
          'niter':25,
    }  
    X_object = X.mode(**kwargs)
    #################################################################################################################################
    ################################################## Nested Crossvalidation ########################################################
    #################################################################################################################################
    # Set up logger
    log.basicConfig(
        level=log.INFO,
        format='%(message)s',  # Remove %(asctime)s and use only custom information
        handlers=[
            log.FileHandler(result_path + '/LOGS.log', 'w'),
            log.StreamHandler(sys.stdout)
        ]
    )
    logger = log.getLogger()
    logger.info(f"### START OF NESTED CROSSVALIDATION FOR DATASET: {dataset}, MISSING: {missing}, BANDWIDTH: {bandwidth}, THRESHOLD: {threshold_samples} ###")
    # save results for outer
    score_umoe_list_total = []
    score_ref_moe_ev_list_total = []
    score_ref_moe_mode_list_total = []
    score_ref_nn_mode_list_total = []
    score_ref_nn_ev_list_total = []
    # Start of NCV
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
        # global mode value for reference method
        ref_train_outer_mode = uf.uframe()
        ref_train_outer_mode.append(X_train_outer.mode())
        # expected value for reference method
        ref_train_outer_ev = uf.uframe()
        ref_train_outer_ev.append(X_train_outer.ev())
        # val
        ref_val_outer_ev = uf.uframe()
        ref_val_outer_ev.append(X_val_outer.ev()) 
        ref_val_outer_mode = uf.uframe()
        ref_val_outer_mode.append(X_val_outer.mode())
        # save results of best expert size for outer Fold
        opt_expert_size_umoe = None
        opt_expert_size_moe_ev = None
        opt_expert_size_moe_mode = None
        ########################################## Inner Crossvalidation ################################################
        val_loss_umoe_list_total_inner = []
        val_loss_ref_moe_ev_list_total_inner = []
        val_loss_ref_moe_mode_list_total_inner = []
        # Tuning subspace/Expert size
        expert_range = range(2, n_experts_max + 1)
        logger.info(f"### START OF INNER CROSSVALIDATON FOR {fold} .FOLD ###")
        for n in expert_range:
            print(n)
            val_loss_umoe_list = []
            val_loss_moe_ev_list = []
            val_loss_moe_mode_list = []
            kf_inner = KFold(n_splits=n_folds_inner, shuffle=True, random_state=42)  # Inner cross-validation
            for inner_fold, (train_indices_inner, val_indices_inner) in enumerate(kf_inner.split(X_train_outer)):
                X_train_inner = X_train_outer[train_indices_inner]
                target_train_inner = target_train_outer[train_indices_inner]
                X_val_inner = X_train_outer[val_indices_inner]
                target_val_inner = target_train_outer[val_indices_inner]
                # global mode value for reference method
                ref_train_inner_mode = uf.uframe()
                ref_train_inner_mode.append(X_train_inner.mode())
                # expected value for reference method
                ref_train_inner_ev = uf.uframe()
                ref_train_inner_ev.append(X_train_inner.ev())
                # val
                ref_val_inner_ev = uf.uframe()
                ref_val_inner_ev.append(X_val_inner.ev()) 
                ref_val_inner_mode = uf.uframe()
                ref_val_inner_mode.append(X_val_inner.mode())
                ################################################## uMoE #############################################################
                try:
                    umoe_model = umoe.MoE(n, inputsize = input_size, outputsize = output_size)
                    umoe_model.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                            verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                            n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)   
                    
                    val_loss_umoe_list.append(umoe_model.get_val_loss())
                except:
                    # if uMoE can not distribute dataset across all experts, this expert size n is then discarded for outer loop
                    val_loss_umoe_list.append(np.inf)
                ################################################## MoE - Exp. Value #############################################################
                try:
                    ref_moe_ev = moe.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
                    ref_moe_ev.fit(ref_train_inner_ev, target_train_inner, ref_val_inner_ev, target_val_inner,
                                verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                                n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                      
                    val_loss_moe_ev_list.append(ref_moe_ev.get_val_loss())
                except:
                    val_loss_moe_ev_list.append(np.inf)
                ############################################# MoE - Global Mode Value #############################################################
                try:
                    ref_moe_mode = moe.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [32, 32], hidden_gate = [32, 32])
                    ref_moe_mode.fit(ref_train_inner_mode, target_train_inner, ref_val_inner_mode, target_val_inner,
                                verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                                n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
     
                    val_loss_moe_mode_list.append(ref_moe_mode.get_val_loss())
                except:
                    val_loss_moe_mode_list.append(np.inf)
                
                    
            val_loss_umoe_list_total_inner.append(np.mean(val_loss_umoe_list))
            val_loss_ref_moe_ev_list_total_inner.append(np.mean(val_loss_moe_ev_list))
            val_loss_ref_moe_mode_list_total_inner.append(np.mean(val_loss_moe_mode_list))
        ######################################################################################################################  
        ###################################################### Outer Fold Eval ###################################################
        ######################################################################################################################
        opt_expert_size_umoe = ut.find_best_expert(expert_range, val_loss_umoe_list_total_inner)
        opt_expert_size_moe_ev = ut.find_best_expert(expert_range, val_loss_ref_moe_ev_list_total_inner)
        opt_expert_size_moe_mode = ut.find_best_expert(expert_range, val_loss_ref_moe_mode_list_total_inner)
        logger.info(f"### FINISHED HYPERPARAMETER SEARCH FOR {fold} .FOLD - BEST EXPERT SIZE uMoE {opt_expert_size_umoe} ###")
        logger.info(f"### VAL LOSS uMoE {val_loss_umoe_list_total_inner} ###")
        logger.info(f"### FINISHED HYPERPARAMETER SEARCH FOR {fold} .FOLD - BEST EXPERT SIZE MoE-EV {opt_expert_size_moe_ev} ###")
        logger.info(f"### VAL LOSS MoE-EV {val_loss_ref_moe_ev_list_total_inner} ###")
        logger.info(f"### FINISHED HYPERPARAMETER SEARCH FOR {fold} .FOLD - BEST EXPERT SIZE MoE-Mode {opt_expert_size_moe_mode} ###")
        logger.info(f"### VAL LOSS MoE-Mode {val_loss_ref_moe_mode_list_total_inner} ###")
        ################################################## uMoE #############################################################
        try:
            umoe_model = umoe.MoE(opt_expert_size_umoe, inputsize = input_size, outputsize = output_size)
            umoe_model.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            # if uMoE can not distribute dataset across all experts, this expert size n is set to two (as default value)
            umoe_model = umoe.MoE(2, inputsize = input_size, outputsize = output_size)
            umoe_model.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = umoe_model.predict(X_test_outer)
        score_umoe = umoe_model.evaluation(predictions, target_test_outer) 
        logger.info(f"uMoE score: {score_umoe}")
        score_umoe_list_total.append(score_umoe)
        ################################################## MOE - EV #############################################################
        try:
            ref_moe_ev = moe.MoE(opt_expert_size_moe_ev, inputsize = input_size, outputsize = output_size)
            ref_moe_ev.fit(ref_train_outer_ev, target_train_outer, ref_val_outer_ev, target_val_outer,
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            ref_moe_ev = moe.MoE(2, inputsize = input_size, outputsize = output_size)
            ref_moe_ev.fit(ref_train_outer_ev, target_train_outer, ref_val_outer_ev, target_val_outer,
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = ref_moe_ev.predict(X_test_outer)
        score_ref_moe_ev = ref_moe_ev.evaluation(predictions, target_test_outer) 
        logger.info(f"MoE-EV score: {score_ref_moe_ev}")
        score_ref_moe_ev_list_total.append(score_ref_moe_ev)
        ################################################## MOE - Mode #############################################################
        try:
            ref_moe_mode = moe.MoE(opt_expert_size_moe_mode, inputsize = input_size, outputsize = output_size)
            ref_moe_mode.fit(ref_train_outer_mode, target_train_outer, ref_val_outer_mode, target_val_outer,  
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            ref_moe_mode = moe.MoE(2, inputsize = input_size, outputsize = output_size)
            ref_moe_mode.fit(ref_train_outer_mode, target_train_outer, ref_val_outer_mode, target_val_outer,  
                    verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = ref_moe_mode.predict(X_test_outer)
        score_ref_moe_mode = ref_moe_mode.evaluation(predictions, target_test_outer) 
        logger.info(f"MoE-Mode score: {score_ref_moe_mode}")
        score_ref_moe_mode_list_total.append(score_ref_moe_mode)

        ############################################### NN Mode #############################################################  
        nn_mode = moe.MoE(1, inputsize = input_size, outputsize = output_size)
        nn_mode.fit(ref_train_outer_mode, target_train_outer, ref_val_outer_mode, target_val_outer,
                verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = nn_mode.predict(X_test_outer)
        score_nn_mode = nn_mode.evaluation(predictions, target_test_outer) 
        logger.info(f"Ref NN Mode score: {score_nn_mode}")
        score_ref_nn_mode_list_total.append(score_nn_mode)
        ############################################### NN-EV #############################################################  
        nn_ev = moe.MoE(1, inputsize = input_size, outputsize = output_size)
        nn_ev.fit(ref_train_outer_ev, target_train_outer, ref_val_outer_ev, target_val_outer, 
                verbose=verbose, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = nn_ev.predict(X_test_outer)
        score_nn_ev = nn_ev.evaluation(predictions, target_test_outer) 
        logger.info(f"Ref NN Exp. Value score: {score_nn_ev}")
        score_ref_nn_ev_list_total.append(score_nn_ev)
    logger.info("#############################################################################################")
    logger.info("#############################################################################################")
    logger.info("#############################################################################################")
    logger.info(f"uMoE TOTAL SCORE: {np.mean(score_umoe_list_total)}")
    logger.info(f"REF MOE MODAL TOTAL SCORE: {np.mean(score_ref_moe_mode_list_total)}")
    logger.info(f"REF MOE EXP V. TOTAL SCORE: {np.mean(score_ref_moe_ev_list_total)}")
    logger.info(f"REF NN MODAL TOTAL SCORE: {np.mean(score_ref_nn_mode_list_total)}")
    logger.info(f"REF NN EXP V. TOTAL SCORE: {np.mean(score_ref_nn_ev_list_total)}")
    logger.info("#############################################################################################")


