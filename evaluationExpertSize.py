import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import argparse
import uMoE as pm
import refMoE as rm
import utils as ut
import uframe as uf

# Create an argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Experimental Results for number of subspaces")
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

    # Load data
    data, target, input_size, output_size, score_type = ut.preprocess_data(data_path = data_path, dataset=dataset)

    # Scale data
    data_sc = MinMaxScaler().fit_transform(data)
    target = target
    
    # Uframe
    X = uf.uframe_from_array_mice_2(data_sc, kernel = "gaussian" , p = missing, bandwidth = bandwidth)
    kwargs = {
          'stepsize':0.2,
          'niter':30,
    }  
    X_object = X.mode(**kwargs)

    ################################################## Crossvalidation ########################################################
    indices = np.arange(len(data_sc))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # save results
    expert_range = range(2, n_experts_max + 1)
    score_umoe_list_total = []
    score_ref_moe_mode_list_total = []
    score_ref_moe_ev_list_total = []
    score_ref_nn_mode_list_total = []
    score_ref_nn_ev_list_total = []
    # start crossvalidation
    for n in expert_range:
        score_umoe_list = []
        score_ref_moe_mode_list = []
        score_ref_moe_ev_list = []
        score_ref_nn_mode_list = []
        score_ref_nn_ev_list = []
        # Print the indices for each fold
        for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
            val_size = len(train_indices) // 3
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            # split Uframe
            X_train = X[train_indices]
            target_train = target[train_indices]
            X_val = X[val_indices]
            target_val = target[val_indices]
            data_test = data_sc[test_indices]
            target_test = target[test_indices]
            # global mode value for reference methods
            ref_train_mode = uf.uframe()
            ref_train_mode.append(X_train.mode()) 
            ref_train_ev = uf.uframe()
            ref_train_ev.append(X_train.ev())  
            # val
            ref_val_mode = uf.uframe()
            ref_val_mode.append(X_val.mode()) 
            ref_val_ev = uf.uframe()
            ref_val_ev.append(X_val.ev())  
              
            ########################### uMoE #############################################################
            try:
                umoe = pm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                umoe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                        verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                        n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                umoe = pm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                umoe.fit(X_train, target_train, X_val, target_val, threshold_samples=threshold_samples, local_mode = local_mode, weighted_experts=True, 
                        verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, n_epochs=n_epochs, 
                        n_samples=n_samples, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            predictions = umoe.predict(data_test)
            score_moe = umoe.evaluation(predictions, target_test) 
            print(f"uMoE score: {score_moe}")
            score_umoe_list.append(score_moe)

            ############################ MoE MODE #############################################################
            try:
                ref_moe = rm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_mode, target_train, ref_val_mode, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                ref_moe = rm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_mode, target_train, ref_val_mode, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
          
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            print(f"Ref Mode MoE score: {score_ref_moe}")
            score_ref_moe_mode_list.append(score_ref_moe)
   
            ############################ MoE EV #############################################################
            try:
                ref_moe = rm.MoE(n, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_ev, target_train, ref_val_ev, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            except:
                ref_moe = rm.MoE(2, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16], hidden_gate = [16, 16])
                ref_moe.fit(ref_train_ev, target_train, ref_val_ev, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
            # predictions / eval
            predictions_ref = ref_moe.predict(data_test)
            score_ref_moe = ref_moe.evaluation(predictions_ref, target_test)
            print(f"Ref EV MoE score: {score_ref_moe}")
            score_ref_moe_ev_list.append(score_ref_moe)    
            
            ############################ NN Mode #############################################################
            if n == 2:
               # NN
                nn_mode = rm.MoE(1, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16],  hidden_gate=[1])
                # val
                nn_mode.fit(ref_train_mode, target_train, ref_val_mode, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                  
                predictions_ref = nn_mode.predict(data_test)
                score_ref_nn_mode = nn_mode.evaluation(predictions_ref, target_test)
                score_ref_nn_mode_list_total.append(score_ref_nn_mode)
                print(f"Ref NN Mode score: {score_ref_nn_mode}")
                
                
            ############################ NN EV #############################################################
            if n == 2:
               # NN
                nn_ev = rm.MoE(1, inputsize = input_size, outputsize = output_size, hidden_experts = [16, 16],  hidden_gate=[1])
                # val
                nn_ev.fit(ref_train_ev, target_train, ref_val_ev, target_val,
                            verbose=False, batch_size_experts=batch_size_experts, batch_size_gate=batch_size_gate, 
                            n_epochs=n_epochs, lr = lr, reg_lambda=reg_lambda, reg_alpha = 0.5)
                  
                predictions_ref = nn_ev.predict(data_test)
                score_ref_nn_ev = nn_ev.evaluation(predictions_ref, target_test)
                score_ref_nn_ev_list_total.append(score_ref_nn_ev)
                print(f"Ref NN EV score: {score_ref_nn_ev}")

           ########################################################################      
        # calculate average for every fold
        score_umoe_list_total.append(np.mean(score_umoe_list))
        score_ref_moe_mode_list_total.append(np.mean(score_ref_moe_mode_list))
        score_ref_moe_ev_list_total.append(np.mean(score_ref_moe_ev_list))
    # average of NN models
    score_ref_nn_ev_list_total = np.mean(score_ref_nn_ev_list_total)
    score_ref_nn_mode_list_total = np.mean(score_ref_nn_mode_list_total)
    # plot
    ut.compare_scores(score_umoe_list_total, score_ref_moe_mode_list_total, score_ref_moe_ev_list_total, score_ref_nn_mode_list_total, score_ref_nn_ev_list_total, expert_range, result_path, dataset, missing, score_type, bandwidth, threshold_samples)
