import os
import re
import numpy as np
import hjson
np.set_string_function(lambda a: str(a.shape), repr=False)
from numpy import array
import pandas as pd 
from datetime import datetime
from UtilityFuncs import *


if __name__ == "__main__":
    # SET PARAMETERS
    config = hjson.load(open('config.hjson','rb'))
    match = re.search(r"^f\d+_ma(\d+)_", config['feat_options'])
    DS = int(match.group(1)) # Down sampling rate
    features_use_list=[r"B\d*_\w*__(?!T$)",r"B\d*_\w*_PTT",r"(B\d*_\w*__IBI\d|B\d*_\w*__TR)",r"(B\d*_\w*__A$|.*__AR$)",r"(B\d*_\w*__IPA$|B\d*_\w*__IPAR$)",r"B\d*_D2S__\w*",r"B\d*_D2I__\w*",r"B\d*_D2M__\w*",r"(IBI_\d|D2D_\d)",r"(IBI_\d|PTT|D2\w_\d)"]
    features_pattern_list=[r"^BM[1234](_BM[1234])*_[A-Z2]*__((?!T$))",r"^BM[1234](_BM[1234])*_[A-Z2]*__((?!IBI)(?!T$))",r"^BAEM[12](_BM[12])*_[A-Z2]*__((?!IBI)(?!T$))",r"(B\d*_\w*__IBI\d|B\d*_\w*__TR)",r"(B\d*_\w*__A$|.*__AR$)",r"(B\d*_\w*__IPA$|B\d*_\w*__IPAR$)",r"B\d*_D2S__\w*",r"B\d*_D2I__\w*",r"B\d*_D2M__\w*",r"(IBI_\d|D2D_\d)",r"(IBI_\d|PTT|D2\w_\d)"]
    cv = config['cv_list'][config['training_select']]
    shuffle = config['shuffle_list'][config['training_select']] 
    now=datetime.now()
    timestamp = now.strftime("%Y-%m-%d")
    feature_path = config['root_path'] + "features/" + config['feature_data'] + '/' + config['feat_options']
    outputs_file = feature_path
    features_file = feature_path
    if config['features_mean'] == 1:
        features_file += "_mean_all.csv"
    else:
        features_file += "_all.csv"
    data_tab_all = pd.read_csv(features_file, quotechar='"', skipinitialspace=True) # Reading the main feature file
    data_sum_all_subjects = pd.DataFrame() # Summary of results dataframe
    # MAIN LOOP
    for subject_id in config['subject_id_arr']:
        print("Subject = " + str(subject_id))
        predictions_path = config['root_path'] + "predictions/" + timestamp + "_" + config['training_name_list'][config['training_select']] + "_" + config['feature_name_list'][config['feature_select']] + "/subject_id_" + str(subject_id) + "/"
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        # GET SUBJECT'S DATA
        data_df = data_tab_all[data_tab_all['subject_id'] == subject_id]
        data_copy = data_df.copy()       
        features_names = array(data_df.columns)
        # CHOOSE RELEVANT SETUPS BASED ON THE SELECTED TRAINING
        data_filt_tab = filter_setups(data_copy, config['setups_name_match_str'][config['training_select']])
        # SET TRAINING PARAMETERS
        train_index_arr = []
        test_index_arr = []
        if shuffle == 1:
            data_row_tab = data_filt_tab[::int(DS/2)] # DOWNSAMPLING; 50% OVERLAP
            kf = KFold(n_splits=config['n_fold'], shuffle=True)
            kf_gs = KFold(n_splits=config['n_fold'] - 1, shuffle=False)
            print("Data after downsampling=" + str(data_row_tab.shape[0]))
        else:
            data_row_tab = data_filt_tab
            kf = KFold(n_splits=config['n_fold'], shuffle=False)
            kf_gs = KFold(n_splits=config['n_fold'] - 1, shuffle=False)
        # REMOVE MISSING DATA (ROWS WITH MORE THAN data_row_tab*100% OF FEATURES MISSING)
        data_filt_row_tab = data_row_tab[pd.isnull(data_row_tab).sum(axis=1) <= config['row_remove_nan_perc']*data_row_tab.shape[1]]
        DBP = array(data_filt_row_tab['DBP'])
        SBP = array(data_filt_row_tab['SBP'])
        if len(data_filt_row_tab)<24:
            print('Not enough Data, Skipped')
            continue
    
        print("Data after cleaning=" + str(data_filt_row_tab.shape[0]) + "/" + str(
                data_row_tab.shape[0]) + "=" + str(
                data_filt_row_tab.shape[0] / data_row_tab.shape[0] * 100) + "%")
        # GET TRAIN AND TEST INDICES
        if cv == 1:
            kf.get_n_splits(data_filt_row_tab)
            n = 0
            for train_index, test_index in kf.split(data_filt_row_tab):
                train_index_arr.append(train_index)
                test_index_arr.append(test_index)
                n = n + 1
        else:
            for testing_match_str_i in config['testing_match_str'][config['training_select']]:
                train_index_arr.append(array([j for j, item in enumerate(data_filt_row_tab['exp_setup_name']) if
                     re.search(config['training_match_str'][config['training_select']], item)]))
                test_index_arr_ini = array([j for j, item in enumerate(data_filt_row_tab['exp_setup_name']) if
                                            re.search(testing_match_str_i, item)])
                if len(test_index_arr_ini) > 0:
                    test_index_arr.append(test_index_arr_ini)
        # FILTER FEATURES TO BE USED BASED ON THE feature_select PARAMETER
        data_filt_col_tab, features_names_select = filter_features(data_filt_row_tab, features_pattern_list[config['feature_select']])
        # Fix Remaining Missing Data
        data_imp = impute_nans(data_filt_col_tab)
        X_raw = X_raw_imp = data_imp
        X_raw_names = features_names_select
        N=len(X_raw_imp)
        D = np.shape(X_raw_imp)[1]
        # INITIALIZE TRAIN AND TEST ARRAYS FOR SUBJECT
        y_train_all_data = []
        y_test_all_data = []
        kfold_train_all_data = []
        kfold_test_all_data = []
        info_arr_test_all_data = np.empty((N, 2))
        info_arr_train_all_data = np.empty((N, 2))
        sample_index_test_all_data = []
        y_train_err_GraBoosting_data = []
        y_test_err_GraBoosting_data = []
        y_train_err_XGBoosting_data = []
        y_test_err_XGBoosting_data = []
        y_train_err_lm_data = []
        y_test_err_lm_data = []
        y_train_err_tree_data = []
        y_test_err_tree_data = []
        y_train_err_randforest_data = []
        y_test_err_randforest_data = []
        y_train_err_ada_data = []
        y_test_err_ada_data = []
        y_train_err_svr_data = []
        y_test_err_svr_data = []
        y_train_err_nn_data = []
        y_test_err_nn_data = []
        model_param_test_ada_data = []
        model_feat_imp_test_ada_data = []
        model_param_test_svr_data = []
        model_feat_imp_test_svr_data = []
        model_param_test_lm_data = []
        model_feat_imp_test_lm_data = []
        model_param_test_GraBoosting_data = []
        model_feat_imp_test_GraBoosting_data = []
        model_param_test_XGBoosting_data = []
        model_feat_imp_test_XGBoosting_data = []
        model_param_test_nn_data = []
        model_param_test_tree_data = []
        model_param_test_randforest_data = []
        data_sum_all = pd.DataFrame(columns=['training_name', 'feature_name', 'model name', 'Subject ID', 'training DBP NN', 'CC', 'ME',
                             'RMSE', 'testing DBP N', 'CC', 'ME', 'RMSE', 'training SBP N', 'CC', 'ME', 'RMSE','testing SBP N', 'CC', 'ME', 'RMSE'])
        for m in [0,1]: # Iterate over DBP=0 and SBP=1
            # INITIALIZE TRAIN AND TEST ARRAYS FOR SBP AND DBP
            i = 0
            y_train_all = []
            y_test_all = []
            train_index_all = []
            test_index_all = []
            kfold_train_all = []
            kfold_test_all = []
            info_arr_test_all = np.empty((0, 6), float)
            info_arr_train_all = np.empty((0, 6), float)
            sample_index_test_all = []
            y_train_err_GraBoosting = []
            y_test_err_GraBoosting = []
            y_train_err_XGBoosting = []
            y_test_err_XGBoosting = []
            y_test_err_lm = []
            y_train_err_lm = []
            y_train_err_tree = []
            y_test_err_tree = []
            y_train_err_ada = []
            y_test_err_ada = []
            y_train_err_randforest = []
            y_test_err_randforest = []
            y_train_err_svr = []
            y_test_err_svr = []
            y_train_err_nn = []
            y_test_err_nn = []
            model_param_test_ada = np.empty((0, 2), int)
            model_feat_imp_test_ada = np.empty((0, D), int)
            model_param_test_svr = np.empty((0, 2), int)
            model_feat_imp_test_svr = np.empty((0, D), int)
            model_param_test_lm = np.empty((0, 2), int)
            model_feat_imp_test_lm = np.empty((0, D), int)
            model_param_test_GraBoosting = np.empty((0, 2), int)
            model_feat_imp_test_GraBoosting = np.empty((0, D), int)
            model_param_test_XGBoosting = np.empty((0, 2), int)
            model_feat_imp_test_XGBoosting = np.empty((0, D), int)
            # ITERATE OVER TESTING FOLDS
            for testing_match_str_i in test_index_arr:
                i = i + 1
                if m == 0:
                    y_all = DBP
                    print("Model = DBP,", end="")
                else:
                    y_all = SBP
                    print("Model = SBP,", end="")
                print("Fold = ", i, end="")
                # GET FOLD'S TRAIN DATA, TEST DATA, AND META INFORMATION
                train_index = train_index_arr[i - 1]
                test_index = test_index_arr[i - 1]
                print(" TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_raw[train_index], X_raw[test_index]
                y_train, y_test = y_all[train_index], y_all[test_index]
                info_arr_test = array(
                    data_filt_row_tab[['subject_id', 'setup_n', 'trial_n','P_MS__T', 'Sample_c', 'exp_setup_name']].iloc[
                        test_index])
                info_arr_train = array(
                    data_filt_row_tab[['subject_id', 'setup_n', 'trial_n','P_MS__T', 'Sample_c', 'exp_setup_name']].iloc[
                        train_index])
                y_train_all = np.append(y_train_all, y_train)
                y_test_all = np.append(y_test_all, y_test)
                info_arr_test_all = np.append(info_arr_test_all, info_arr_test, axis=0)
                info_arr_train_all = np.append(info_arr_train_all, info_arr_train, axis=0)
                train_index_all = np.append(train_index_all, train_index)
                test_index_all = np.append(test_index_all, test_index)
                kfold_train_all = np.append(kfold_train_all, i * np.ones([train_index.size, ]))
                kfold_test_all = np.append(kfold_test_all, i * np.ones([test_index.size, ]))
                # START TRAINING PROCESS FOR EACH SELECTED MODEL
                ###################### GradientBoosting ######################################
                if config['GraBoosting_EN'] == 1:
                    y_train_err,y_test_err,model_feat_imp2,model_param2 = model_graboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs,features_names_select)
                    model_param_test_GraBoosting = np.append(model_param_test_GraBoosting, model_param2, axis=0)
                    model_feat_imp_test_GraBoosting = np.append(model_feat_imp_test_GraBoosting, model_feat_imp2, axis=0)
                    y_train_err_GraBoosting = np.append(y_train_err_GraBoosting, y_train_err)
                    y_test_err_GraBoosting = np.append(y_test_err_GraBoosting, y_test_err)
                ###################### XGBoosting ######################################
                if config['XGBoosting_EN'] == 1:
                    y_train_err,y_test_err,model_feat_imp2,model_param2 = model_xgboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs,features_names_select)
                    model_param_test_XGBoosting = np.append(model_param_test_XGBoosting, model_param2, axis=0)
                    model_feat_imp_test_XGBoosting = np.append(model_feat_imp_test_XGBoosting, model_feat_imp2, axis=0)
                    y_train_err_XGBoosting = np.append(y_train_err_XGBoosting, y_train_err)
                    y_test_err_XGBoosting = np.append(y_test_err_XGBoosting, y_test_err)
                # ###################### Linear Regression #####################################
                # Create linear regression object
                if config['lm_EN'] == 1:
                    y_train_err,y_test_err,model_feat_imp2,model_param2 = model_lm(X_train, y_train, X_test, y_test, cv, i, D)
                    model_param_test_lm = np.append(model_param_test_svr, model_param2, axis=0)
                    model_feat_imp_test_lm = np.append(model_feat_imp_test_svr, model_feat_imp2, axis=0)
                    y_train_err_lm = np.append(y_train_err_lm, y_train_err)
                    y_test_err_lm = np.append(y_test_err_lm, y_test_err)
                ###################### DiscisionTree Regression #####################################
                if config['tree_EN'] == 1:
                    y_train_err,y_test_err= model_tree(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs)
                    y_train_err_tree = np.append(y_train_err_tree, y_train_err)
                    y_test_err_tree = np.append(y_test_err_tree, y_test_err)
                ###################### Ada Boosting Regression #######################################
                if config['ada_EN'] == 1:
                    y_train_err,y_test_err,model_feat_imp2,model_param2 = model_adaboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs,features_names_select)
                    model_param_test_ada = np.append(model_param_test_ada, model_param2, axis=0)
                    model_feat_imp_test_ada = np.append(model_feat_imp_test_ada, model_feat_imp2, axis=0)
                    y_train_err_ada = np.append(y_train_err_ada, y_train_err)
                    y_test_err_ada = np.append(y_test_err_ada, y_test_err)
                ###################### RandomForest Regression #######################################
                if config['randforest_EN'] == 1:
                    y_train_err,y_test_err= model_RF(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs)
                    y_train_err_randforest = np.append(y_train_err_randforest, y_train_err)
                    y_test_err_randforest = np.append(y_test_err_randforest, y_test_err)
                # # ###################### SVM Regression #######################################
                if config['svr_EN'] == 1:
                    y_train_err,y_test_err,model_feat_imp2,model_param2 = model_svr(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs)
                    model_param_test_svr = np.append(model_param_test_svr, model_param2, axis=0)
                    model_feat_imp_test_svr = np.append(model_feat_imp_test_svr, model_feat_imp2, axis=0)
                    y_train_err_svr = np.append(y_train_err_svr, y_train_err)
                    y_test_err_svr = np.append(y_test_err_svr, y_test_err)
                    # # ###################### NN Regression #######################################
                if config['nn_EN'] == 1:
                    y_train_err,y_test_err= model_nn(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs)
                    y_train_err_nn = np.append(y_train_err_nn, y_train_err)
                    y_test_err_nn = np.append(y_test_err_nn, y_test_err)
            if config['GraBoosting_EN'] == 1:
                y_train_err_GraBoosting_data.append(y_train_err_GraBoosting)
                y_test_err_GraBoosting_data.append(y_test_err_GraBoosting)
                model_param_test_GraBoosting_data.append(model_param_test_GraBoosting)
                model_feat_imp_test_GraBoosting_data.append(model_feat_imp_test_GraBoosting)
            if config['XGBoosting_EN'] == 1:
                y_train_err_XGBoosting_data.append(y_train_err_XGBoosting)
                y_test_err_XGBoosting_data.append(y_test_err_XGBoosting)
                model_param_test_XGBoosting_data.append(model_param_test_XGBoosting)
                model_feat_imp_test_XGBoosting_data.append(model_feat_imp_test_XGBoosting)
            if config['lm_EN'] == 1:
                y_train_err_lm_data.append(y_train_err_lm)
                y_test_err_lm_data.append(y_test_err_lm)
                model_param_test_lm_data.append(model_param_test_lm)
                model_feat_imp_test_lm_data.append(model_feat_imp_test_lm)
            if config['tree_EN'] == 1:
                y_train_err_tree_data[:, m] = y_train_err_tree
                y_test_err_tree_data[:, m] = y_test_err_tree
            if config['randforest_EN'] == 1:
                y_train_err_randforest_data[:, m] = y_train_err_randforest
                y_test_err_randforest_data[:, m] = y_test_err_randforest
            if config['ada_EN'] == 1:
                y_train_err_ada_data.append(y_train_err_ada)
                y_test_err_ada_data.append(y_test_err_ada)
                model_param_test_ada_data.append(model_param_test_ada)
                model_feat_imp_test_ada_data.append(model_feat_imp_test_ada)
            if config['svr_EN'] == 1:
                y_train_err_svr_data.append(y_train_err_svr)
                y_test_err_svr_data.append(y_test_err_svr)
                model_param_test_svr_data.append(model_param_test_svr)
                model_feat_imp_test_svr_data.append(model_feat_imp_test_svr)
            if config['nn_EN'] == 1:
                y_train_err_nn_data[:, m] = y_train_err_nn
                y_test_err_nn_data[:, m] = y_test_err_nn
            y_train_all_data.append(y_train_all)
            y_test_all_data.append(y_test_all)
            kfold_train_all_data.append(kfold_train_all)
            kfold_test_all_data.append(kfold_test_all)
            info_arr_test_all_data = info_arr_test_all
            info_arr_train_all_data = info_arr_train_all      
        # SUMMARIZE THE RESULTS FOR EACH SUBJECT
        data = dict()
        data['y_train_all_data'] = np.array(y_train_all_data).transpose()
        data['y_test_all_data'] = np.array(y_test_all_data).transpose()
        data['feature_name'] = config['feature_name_list'][config['feature_select']]
        data['training_name'] = config['training_name_list'][config['training_select']]
        data['info_arr_test_all_data'] = info_arr_test_all_data
        data['info_arr_train_all_data'] = info_arr_train_all_data
        data['subject_id'] = subject_id
        data['kfold_train_all_data'] = np.array(kfold_train_all_data).transpose()
        data['kfold_test_all_data'] = np.array(kfold_test_all_data).transpose()
        data['features_names']=features_names
        data['config']=config
        data['X_raw_names']=X_raw_names
        data['predictions_path']=predictions_path
        if config['GraBoosting_EN'] == 1:
            data['y_train_err'] = np.array(y_train_err_GraBoosting_data).transpose()
            data['y_test_err'] = np.array(y_test_err_GraBoosting_data).transpose()
            data['model_param_test'] = model_param_test_GraBoosting_data
            data['model_feat_imp_test'] = model_feat_imp_test_GraBoosting_data
            data['model_name'] = "GraBoosting"
            data_sum_GraBoosting = data_summary(data)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_GraBoosting,index=data_sum_all.columns),ignore_index=True)
        if config['XGBoosting_EN'] == 1:
            data['y_train_err'] = np.array(y_train_err_XGBoosting_data).transpose()
            data['y_test_err'] = np.array(y_test_err_XGBoosting_data).transpose()
            data['model_param_test'] = model_param_test_XGBoosting_data
            data['model_feat_imp_test'] = model_feat_imp_test_XGBoosting_data
            data['model_name'] = "XGBoosting"
            data_sum_XGBoosting = data_summary(data)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_XGBoosting,index=data_sum_all.columns),ignore_index=True)
        if config['lm_EN'] == 1:
            data['y_train_err'] = np.array(y_train_err_lm_data).transpose()
            data['y_test_err'] = np.array(y_test_err_lm_data).transpose()
            data['model_param_test'] = model_param_test_lm_data
            data['model_name'] = "lm         "
            data_sum_lm = data_summary(data)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_lm,index=data_sum_all.columns),ignore_index=True)
        if config['tree_EN'] == 1:
            y_train_err = y_train_err_tree_data
            y_test_err = y_test_err_tree_data
            model_param_test = model_param_test_tree_data
            model_name = "tree       "
            data_sum_tree = data_summary(y_train_all_data, y_train_err, y_test_all_data, y_test_err, model_name,
                                         config['features_mean'], info_arr_test_all_data, sample_index_test_all_data,
                                         model_param_test)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_tree,index=data_sum_all.columns),ignore_index=True)
        if config['ada_EN'] == 1:
            data['y_train_err'] = np.array(y_train_err_ada_data).transpose()
            data['y_test_err'] = np.array(y_test_err_ada_data).transpose()
            data['model_param_test'] = model_param_test_ada_data
            data['model_feat_imp_test'] = model_feat_imp_test_ada_data
            data['model_name'] = "ada        "
            data_sum_ada = data_summary(data)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_ada,index=data_sum_all.columns),ignore_index=True )
        if config['randforest_EN'] == 1:
            y_train_err = y_train_err_randforest_data
            y_test_err = y_test_err_randforest_data
            model_param_test = model_param_test_randforest_data
            model_name = "randforest "
            data_sum_randforest = data_summary(y_train_all_data, y_train_err, y_test_all_data, y_test_err,
                                               model_name, config['features_mean'], info_arr_test_all_data,
                                               sample_index_test_all_data, model_param_test)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_randforest,index=data_sum_all.columns),ignore_index=True)
        if config['svr_EN'] == 1:
            data['y_train_err'] = np.array(y_train_err_svr_data).transpose()
            data['y_test_err'] = np.array(y_test_err_svr_data).transpose()
            data['model_param_test'] = model_param_test_svr_data
            data['model_name'] = "svr        "
            data_sum_svr = data_summary(data)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_svr,index=data_sum_all.columns),ignore_index=True)
        if config['nn_EN'] == 1:
            y_train_err = y_train_err_nn_data
            y_test_err = y_test_err_nn_data
            model_param_test = model_param_test_nn_data
            model_name = "nn         "
            data_sum_nn = data_summary(y_train_all_data, y_train_err, y_test_all_data, y_test_err, model_name,
                                       config['features_mean'], info_arr_test_all_data, sample_index_test_all_data,
                                       model_param_test)
            data_sum_all = data_sum_all.append(pd.Series(data_sum_nn,index=data_sum_all.columns),ignore_index=True)
        # WRITE SUBJECT'S SUMMARY TO FILE
        test_file = predictions_path
        test_file += "data_summary"
        if config['features_mean'] == 1:
            test_file += "_mean"
        test_file += ".csv"
        data_sum_all.to_csv(test_file)
        # SUMMARY FOR ALL SUBJECTS
        data_sum_all_subjects = data_sum_all_subjects.append(data_sum_all,ignore_index=True)#np.vstack([data_sum_all_subjects, data_sum_all[1::]])
    # WRITE OVERALL SUMMARY TO FILE
    test_file = predictions_path + "../../"
    test_file += "data_summary_subjects_"+now.strftime("%Y-%m-%d_%H-%M")
    if config['features_mean'] == 1:
        test_file += "_mean"
    test_file += ".csv"
    data_sum_all_subjects.to_csv(test_file)
        
        
        
        
        
