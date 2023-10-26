import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import logging as log
import sys
import argparse
import uMoE as umoe
import utils as ut
import uframe as uf


"""
This File serves as isolated main file for training an uMoE model. 
The trainingsprocess includes the Nested Cross-Validation to find the 
optimale number of subspaces/Experts
"""

# Create an argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Nested Crossvalidation for uMoE")
# Define and add command-line arguments
parser.add_argument("--dataset", type=str, default="wine_quality", help="Dataset name")
parser.add_argument("--data_path", type=str, default="D:\Github_Projects\MOE_Training_under_Uncertainty\Datasets", help="Path to data")
parser.add_argument("--result_path", type=str, default="D:\Evaluation", help="Path to results")
parser.add_argument("--missing", type=float, default=0.01, help="Missing data percentage")
parser.add_argument("--bandwidth", type=float, default=0.1, help="Bandwidth value")
parser.add_argument("--n_folds", type=int, default=2, help="Number of folds")
parser.add_argument("--n_experts_max", type=int, default=4, help="Maximum number of subspaces/Experts")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--reg_lambda", type=float, default=0.002, help="Regularization lambda")
parser.add_argument("--batch_size_experts", type=int, default=16, help="Batch size for experts")
parser.add_argument("--batch_size_gate", type=int, default=24, help="Batch size for gating unit")
parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--threshold_samples", type=float, default=0.6, help="Threshold samples")
parser.add_argument("--n_samples", type=int, default=150, help="Number of samples")
parser.add_argument("--local_mode", type=bool, default=True, help="Local mode")
parser.add_argument("--inner_fold", type=int, default=2, help="Number of inner Folds for NCV")
parser.add_argument("--outer_fold", type=int, default=2, help="Number of outer Folds for NCV")

if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset = args.dataset
    data_path = args.data_path
    result_path = args.result_path
    missing = args.missing
    bandwidth = args.bandwidth
    n_folds = args.n_folds
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
          'niter':30,
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
    score_umoe_list_total_local = []
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
        ########################################## Inner Crossvalidation ################################################
        val_loss_local_list_total_inner = []
        # Tuning Expert size
        expert_range = range(2, n_experts_max + 1)
        logger.info(f"### START OF INNER CROSSVALIDATON FOR {fold} .FOLD ###")
        for n in expert_range:
            val_loss_local_list = []
            kf_inner = KFold(n_splits=n_folds_inner, shuffle=True, random_state=42)  # Inner cross-validation
            for inner_fold, (train_indices_inner, val_indices_inner) in enumerate(kf_inner.split(X_train_outer)):
                X_train_inner = X_train_outer[train_indices_inner]
                target_train_inner = target_train_outer[train_indices_inner]
                X_val_inner = X_train_outer[val_indices_inner]
                target_val_inner = target_train_outer[val_indices_inner]
                ################################################## uMoE Local #############################################################
                try:
                    umoe_model = umoe.MoE(n, inputsize = input_size, outputsize = output_size)
                    umoe_model.fit(X_train_inner, target_train_inner, X_val_inner, target_val_inner, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                            verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                            n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)   
                    umoe_val_loss = umoe_model.get_val_loss()
                    val_loss_local_list.append(umoe_val_loss)
                except:
                    # if uMoE can not distribute dataset across all experts, this expert size n is then discarded for outer loop
                    val_loss_local_list.append(np.inf)
                    
            val_loss_local_list_total_inner.append(np.mean(val_loss_local_list))
        ######################################################################################################################  
        ###################################################### Outer Fold Eval ###################################################
        ######################################################################################################################
        opt_expert_size_local = ut.find_best_expert(expert_range, val_loss_local_list_total_inner)
        logger.info(f"### FINISHED HYPERPARAMETER SEARCH FOR {fold} .FOLD - BEST EXPERT SIZE uMoE {opt_expert_size_local} ###")
        logger.info(f"### VAL LOSS uMoE {val_loss_local_list_total_inner} ###")
        ################################################## uMoE Local #############################################################
        try:
            umoe_model = umoe.MoE(opt_expert_size_local, inputsize = input_size, outputsize = output_size)
            umoe_model.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        except:
            # if uMoE can not distribute dataset across all experts, this expert size n is set to two (as default value)
            umoe_model = umoe.MoE(2, inputsize = input_size, outputsize = output_size)
            umoe_model.fit(X_train_outer, target_train_outer, X_val_outer, target_val_outer, threshold_samples=threshold_samples, local_mode = True, weighted_experts=True, 
                    verbose=True, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                    n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
        predictions = umoe_model.predict(X_test_outer)
        score_umoe = umoe_model.evaluation(predictions, target_test_outer) 
        logger.info(f"uMoE score: {score_umoe}")
        score_umoe_list_total_local.append(score_umoe)

    logger.info("#############################################################################################")
    logger.info("#############################################################################################")
    logger.info(f"uMoE SCORE: {np.mean(score_umoe_list_total_local)}")
    logger.info("#############################################################################################")


