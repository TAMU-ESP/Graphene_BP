from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor
import re
import pandas as pd
import numpy as np
from numpy import array
from sklearn.impute import SimpleImputer
import math
import os
try:
    import tensorflow as tf
except Exception as e:
    print('Error in importing tensorflow ')
    print(e)
    print('\nPlease fix tensorflow installation or continue without neural network model\n')
    
def model_analysis(y_train,y_train_predict,y_test,y_test_predict):
    y_train_err = y_train_predict - y_train
    y_train_mae = np.mean(abs(y_train_err))
    y_train_rmse = np.sqrt(np.mean((y_train_err)**2, axis=0))
    y_train_me = np.mean((y_train_err))
    y_test_err = y_test_predict - y_test
    y_test_mae = np.mean(abs(y_test_err))
    y_test_rmse = np.sqrt(np.mean((y_test_err)**2, axis=0))
    y_test_me = np.mean((y_test_err))
    print('training RMSE, ME: %2.3f, %2.3f ' %(y_train_rmse,y_train_me), end="")
    print('testing RMSE, ME: %2.3f, %2.3f ' %(y_test_rmse,y_test_me))
    return y_train_err,y_test_err


def data_summary(data):
    y_train_predict = data['y_train_all_data'] + data['y_train_err']
    y_train_n = len(data['y_train_all_data'])
    y_train_cc = np.array([np.corrcoef(data['y_train_all_data'], y_train_predict,rowvar=False)[0, 2],np.corrcoef(data['y_train_all_data'], y_train_predict,rowvar=False)[1, 3]])
    y_train_me = np.mean((data['y_train_err']), axis=0)
    y_train_mae = np.mean(abs(data['y_train_err']), axis=0)
    y_train_rmse = np.sqrt(np.mean((data['y_train_err'])**2, axis=0))
    y_test_predict = data['y_test_all_data'] + data['y_test_err']
    y_test_n = len(data['y_test_all_data'])
    y_test_cc = np.array([np.corrcoef(data['y_test_all_data'], y_test_predict, rowvar=False)[0, 2],np.corrcoef(data['y_test_all_data'], y_test_predict, rowvar=False)[1, 3]])
    y_test_me = np.mean((data['y_test_err']), axis=0)
    y_test_mae = np.mean(abs(data['y_test_err']), axis=0)
    y_test_rmse = np.sqrt(np.mean((data['y_test_err'])**2, axis=0))
    print('%s Subject ID,training DBP N, CC, ME, RMSE:\t %g\t %g\t %2.3f\t%2.3f\t%2.3f \t\tSBP N, CC, ME, RMSE:\t %g\t %2.3f\t%2.3f\t%2.3f' % (
    data['model_name'], data['subject_id'], y_train_n, y_train_cc[0], y_train_me[0], y_train_rmse[0], y_train_n,y_train_cc[1], y_train_me[1], y_train_rmse[1]))
    print('%s Subject ID, testing  DBP N, CC, ME, RMSE:\t %g\t %g\t %2.3f\t%2.3f\t%2.3f \t\tSBP N, CC, ME, RMSE:\t %g\t %2.3f\t%2.3f\t%2.3f' % (
    data['model_name'], data['subject_id'], y_test_n, y_test_cc[0], y_test_me[0], y_test_rmse[0], y_test_n,y_test_cc[1], y_test_me[1], y_test_rmse[1]))
    model_name = data['training_name']+'_'+data['feature_name']+'_'+data['model_name'].replace(" ", "")
    if data['config']['features_mean']==1:
        model_name += "_mean"
    train_array = np.concatenate((data['info_arr_train_all_data'],data['kfold_train_all_data'], data['y_train_all_data'], y_train_predict), axis=1)
    train_header = np.concatenate((data['features_names'][1:5], array(
        ['Sample', 'ExpName', 'Fold index DBP', 'Fold Index SBP', 'Ref_data_DBP', 'Ref_data_SBP', 'Predict_data_DBP',
         'Predict_data_SBP'])), axis=0)
    df = pd.DataFrame(train_array)
    train_file = data['predictions_path']
    train_file += model_name
    train_file += "_train.csv"
    if not os.path.exists(data['predictions_path']):
        os.makedirs(data['predictions_path'])
    df.to_csv(train_file, header=train_header)
    if "ada" in model_name:
        test_header = np.concatenate((data['features_names'][1:5],array(['Sample','ExpName','Fold index DBP','Fold Index SBP','Ref_data_DBP','Ref_data_SBP','Predict_data_DBP','Predict_data_SBP','n_estimators_DBP','max_depth_DBP','n_estimators_SBP','max_depth_SBP']),data['X_raw_names'],data['X_raw_names']), axis=0)
        test_array = np.concatenate((data['info_arr_test_all_data'],data['kfold_test_all_data'], data['y_test_all_data'], y_test_predict,data['model_param_test'][0],data['model_param_test'][1],data['model_feat_imp_test'][0],data['model_feat_imp_test'][1]), axis=1)
    else:
        test_array = np.concatenate((data['info_arr_test_all_data'],data['kfold_test_all_data'], data['y_test_all_data'], y_test_predict), axis=1)
    df = pd.DataFrame(test_array)
    test_file = data['predictions_path']
    test_file += model_name
    test_file += "_test.csv"
    if "ada" in model_name:
        df.to_csv(test_file, header=test_header)
    else:
        df.to_csv(test_file, header=None)
    data_sum_dbp = np.stack((data['training_name'],data['feature_name'],data['model_name'],data['subject_id'],y_train_n ,y_train_cc[0], y_train_me[0], y_train_rmse[0],y_test_n, y_test_cc[0], y_test_me[0], y_test_rmse[0]), axis=-1)
    data_sum_sbp = np.stack((y_train_n, y_train_cc[1], y_train_me[1], y_train_rmse[1],y_test_n, y_test_cc[1], y_test_me[1], y_test_rmse[1]), axis=-1)
    data_sum = np.concatenate((data_sum_dbp,data_sum_sbp),axis=0)
    return data_sum

def filter_setups(data, setups_name_match_str):
    setup_name_list = list(data['exp_setup_name'])
    data_idx = [i for i, item in enumerate(setup_name_list) if re.search(setups_name_match_str, item)]
    data_setup_tab = data.iloc[data_idx]
    print("Data after setup select=" + str(np.size(data_setup_tab, 0)) + "/" + str(np.size(data, 0)) + "=" + str(
        np.size(data_setup_tab, 0) / np.size(data, 0) * 100) + "%")  
    return data_setup_tab

def filter_features(data, features_pattern):
    r = re.compile(features_pattern)
    features_names_select = list(filter(r.match, data.columns))
    data_filt_col_tab = data[features_names_select]
    print("Features size=" + str(len(features_names_select)))
    print(features_names_select)
    return data_filt_col_tab, features_names_select

def impute_nans(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)
    data_imp = imp.transform(data)
    return data_imp

def model_adaboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs, features_names_select, trained_model=None):
    if (cv == 0 and i == 1) or (cv == 1):
        ada_grid_final={}
        if config['ada_grid']['n_estimators']:
            ada_grid_final['n_estimators']= config['ada_grid']['n_estimators']
        else:
            ada_grid_final['n_estimators']= [x * pow(2,max(int(math.log(len(X_train)/10,2))-3,3)) for x in [1,2]]
    
        if config['ada_grid']['base_estimator__max_depth']:
            ada_grid_final['base_estimator__max_depth']= config['ada_grid']['base_estimator__max_depth']
        else:
            ada_grid_final['base_estimator__max_depth']= [x * pow(2,int(math.log(D/10,2))+2) for x in [1,2]]
    
        print("Ada Boosting training starts, n_estimators=" + str(
            ada_grid_final['n_estimators']) + ", max_depth=" + str(ada_grid_final['base_estimator__max_depth']), end="")
    
        rng = np.random.RandomState(1)  
        regr_ada = AdaBoostRegressor(DecisionTreeRegressor(),
                                     random_state=rng)  
        regr_ada_gs = GridSearchCV(estimator=regr_ada, param_grid=ada_grid_final, cv=kf_gs,
                                   verbose=config['verbose_param'])
    
        regr_ada_gs.fit(X_train, y_train)
    
        # Feature Importance
        feature_importance = regr_ada_gs.best_estimator_.feature_importances_
        # make importances relative to max importance
        sorted_idx = np.argsort(feature_importance)
    
        print(
            'Best n_estimators, max_depth= %d, %d' % (regr_ada_gs.best_params_['n_estimators'],
                                                      regr_ada_gs.best_estimator_.base_estimator.max_depth),
            end="")
        print(' important features= %s=%1.2f,%s=%1.2f,%s=%1.2f' % (
            array(features_names_select)[sorted_idx][-1], feature_importance[sorted_idx][-1],
            array(features_names_select)[sorted_idx][-2], feature_importance[sorted_idx][-2],
            array(features_names_select)[sorted_idx][-3], feature_importance[sorted_idx][-3]))
    else:
        regr_ada_gs = trained_model
        feature_importance = regr_ada_gs.best_estimator_.feature_importances_
    # Make predictions using the testing set
    y_train_predict = regr_ada_gs.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_ada_gs.predict(X_test)
    
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    model_param = np.append(regr_ada_gs.best_params_['n_estimators'],
                            regr_ada_gs.best_estimator_.base_estimator.max_depth)
    model_feat_imp = (feature_importance[:])
    model_param2 = np.repeat(model_param.reshape(1, model_param.shape[0]), np.shape(X_test)[0],
                             0)
    model_feat_imp2 = np.repeat(model_feat_imp.reshape(1, model_feat_imp.shape[0]),
                                np.shape(X_test)[0], 0)
    return y_train_err, y_test_err, model_feat_imp2, model_param2, regr_ada_gs

def model_graboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs, features_names_select, trained_model=None):
    if (cv == 0 and i == 1) or (cv == 1):
        GraBoosting_grid_final = {}
        if config['GraBoosting_grid']['n_estimators']:
            GraBoosting_grid_final['n_estimators'] = config['GraBoosting_grid']['n_estimators']
        else:
            GraBoosting_grid_final['n_estimators'] = [
                x * pow(2, max(int(math.log(len(X_train) / 10, 2)) -2, 3)) for x in [1, 2]]
    
        if config['GraBoosting_grid']['max_depth']:
            GraBoosting_grid_final['max_depth'] = config['GraBoosting_grid']['max_depth']
        else:
            GraBoosting_grid_final['max_depth'] = [
                x * pow(2, int(math.log(D / 10, 2)) + 2) for x in [1, 2]]
        print(" Gradient Boosting training starts, n_estimators=" + str(
            GraBoosting_grid_final['n_estimators']) + ", max_depth=" + str(
            GraBoosting_grid_final['max_depth']), end="")
    
        regr_GraBoosting = GradientBoostingRegressor()
        regr_GraBoosting_gs = GridSearchCV(estimator=regr_GraBoosting, param_grid=GraBoosting_grid_final,
                                           cv=kf_gs, verbose=config['verbose_param'])
        regr_GraBoosting_gs.fit(X_train, y_train)
    
        feature_importance = regr_GraBoosting_gs.best_estimator_.feature_importances_
        # make importances relative to max importance
        feature_importance_s = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
    
        print('best n_estimators, max_depth= %d, %d' % (
        regr_GraBoosting_gs.best_params_['n_estimators'],
        regr_GraBoosting_gs.best_params_['max_depth']),
              end="")
        print(' important features= %s=%1.2f,%s=%1.2f,%s=%1.2f' % (
            array(features_names_select)[sorted_idx][-1], feature_importance[sorted_idx][-1],
            array(features_names_select)[sorted_idx][-2], feature_importance[sorted_idx][-2],
            array(features_names_select)[sorted_idx][-3], feature_importance[sorted_idx][-3]))
        # compute test set deviance
        test_score = np.zeros((regr_GraBoosting_gs.best_params_['n_estimators'],),
                              dtype=np.float64)
    
    else:
        regr_GraBoosting_gs = trained_model
        feature_importance = regr_GraBoosting_gs.best_estimator_.feature_importances_
    
    # Make predictions using the testing set
    y_train_predict = regr_GraBoosting_gs.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_GraBoosting_gs.predict(X_test)
    
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test,
                                             y_test_predict)
    model_param = np.append(regr_GraBoosting_gs.best_params_['n_estimators'],
                            regr_GraBoosting_gs.best_params_['max_depth'])
    model_feat_imp = (feature_importance[:])
    model_param2 = np.repeat(model_param.reshape(1, model_param.shape[0]),
                             np.shape(X_test)[0],
                             0)
    model_feat_imp2 = np.repeat(model_feat_imp.reshape(1, model_feat_imp.shape[0]),
                                np.shape(X_test)[0], 0)
    return y_train_err, y_test_err, model_feat_imp2, model_param2, regr_GraBoosting_gs

def model_xgboost(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs, features_names_select, trained_model=None):
    if (cv == 0 and i == 1) or (cv == 1):
        XGBoosting_grid_final = {}
        if config['XGBoosting_grid']['n_estimators']:
            XGBoosting_grid_final['n_estimators'] = config['XGBoosting_grid']['n_estimators']
        else:
            XGBoosting_grid_final['n_estimators'] = [
                x * pow(2, max(int(math.log(len(X_train) / 10, 2)) -2, 3)) for x in [1, 2]]
    
        if config['XGBoosting_grid']['max_depth']:
            XGBoosting_grid_final['max_depth'] = config['XGBoosting_grid']['max_depth']
        else:
            XGBoosting_grid_final['max_depth'] = [
                x * pow(2, int(math.log(D / 10, 2)) + 2) for x in [1, 2]]
        print(" XG-Boosting training starts, n_estimators=" + str(
            XGBoosting_grid_final['n_estimators']) + ", max_depth=" + str(
            XGBoosting_grid_final['max_depth']), end="")
    
        regr_XGBoosting = XGBRegressor()
        regr_XGBoosting_gs = GridSearchCV(estimator=regr_XGBoosting, param_grid=XGBoosting_grid_final,
                                           cv=kf_gs, verbose=config['verbose_param'])
        regr_XGBoosting_gs.fit(X_train, y_train)
    
        feature_importance = regr_XGBoosting_gs.best_estimator_.feature_importances_
        # make importances relative to max importance
        feature_importance_s = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
    
        print('best n_estimators, max_depth= %d, %d' % (
        regr_XGBoosting_gs.best_params_['n_estimators'],
        regr_XGBoosting_gs.best_params_['max_depth']),
              end="")
        print(' important features= %s=%1.2f,%s=%1.2f,%s=%1.2f' % (
            array(features_names_select)[sorted_idx][-1], feature_importance[sorted_idx][-1],
            array(features_names_select)[sorted_idx][-2], feature_importance[sorted_idx][-2],
            array(features_names_select)[sorted_idx][-3], feature_importance[sorted_idx][-3]))
        # compute test set deviance
    else:
        regr_XGBoosting_gs = trained_model
        feature_importance = regr_XGBoosting_gs.best_estimator_.feature_importances_
    # Make predictions using the testing set
    y_train_predict = regr_XGBoosting_gs.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_XGBoosting_gs.predict(X_test)
    
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test,
                                             y_test_predict)
    model_param = np.append(regr_XGBoosting_gs.best_params_['n_estimators'],
                            regr_XGBoosting_gs.best_params_['max_depth'])
    model_feat_imp = (feature_importance[:])
    model_param2 = np.repeat(model_param.reshape(1, model_param.shape[0]),
                             np.shape(X_test)[0],
                             0)
    model_feat_imp2 = np.repeat(model_feat_imp.reshape(1, model_feat_imp.shape[0]),
                                np.shape(X_test)[0], 0)
    return y_train_err, y_test_err, model_feat_imp2, model_param2, regr_XGBoosting_gs

def model_lm(X_train, y_train, X_test, y_test, cv, i, D): 
    if (cv == 0 and i == 1) or (cv == 1):
        print(" Linear Regression starts ")
        regr_lm = linear_model.LinearRegression()
        # Train the model using the training sets
        regr_lm.fit(X_train, y_train)
        feature_importance = np.zeros((D, 1))
    # Make predictions using the testing set
    y_train_predict = regr_lm.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_lm.predict(X_test)
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    model_param = np.append(0,0)
    model_feat_imp = (feature_importance[:])
    model_param2 = np.repeat(model_param.reshape(1, model_param.shape[0]),
                             np.shape(X_test)[0],
                             0)
    model_feat_imp2 = np.repeat(model_feat_imp.reshape(1, model_feat_imp.shape[0]),
                                np.shape(X_test)[0], 0)
    return y_train_err, y_test_err, model_feat_imp2, model_param2

def model_tree(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs): 
    print("DiscisionTree Regression starts ", end="")
    # Create DiscisionTree regression object
    regr_tree = DecisionTreeRegressor()
    regr_tree_gs = GridSearchCV(estimator=regr_tree, param_grid=config['tree_grid'], cv=kf_gs,
                                verbose=config['verbose_param'])
    regr_tree_gs.fit(X_train, y_train)
    
    # Make predictions using the testing set
    y_train_predict = regr_tree_gs.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_tree_gs.predict(X_test)
    print("max_depth= ", regr_tree_gs.best_params_['max_depth'], end="")
    print(" important features= ",
          np.argsort(regr_tree_gs.best_estimator_.feature_importances_)[:-5:-1][0] + 1)
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    return y_train_err, y_test_err

def model_RF(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs): 
    print("RandomForest Regression starts ", end="")
    regr_randforest = RandomForestRegressor(random_state=0)
    regr_randforest_gs = GridSearchCV(estimator=regr_randforest, param_grid=config['randforest_grid'],
                                      cv=kf_gs, verbose=config['verbose_param'])
    regr_randforest_gs.fit(X_train, y_train)
    
    # Make predictions using the testing set
    y_train_predict = regr_randforest_gs.predict(X_train)
    # Make predictions using the testing set
    y_test_predict = regr_randforest_gs.predict(X_test)
    print("max_depth= ", regr_randforest_gs.best_params_['max_depth'], end="")
    print(" important features= ",
          np.argsort(regr_randforest_gs.best_estimator_.feature_importances_)[:-5:-1][0] + 1)
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    return y_train_err, y_test_err

def model_svr(X_train, y_train, X_test, y_test, cv, i, config, D, kf_gs): 
    if (cv == 0 and i == 1) or (cv == 1):
        print("svm fitting starts ", end="")
        # Fit regression model 3 different kernel functions are using
        regr_svr = SVR(kernel='rbf', verbose=0)
        regr_svr_gs = GridSearchCV(estimator=regr_svr, param_grid=config['svr_grid'], cv=kf_gs,
                                   verbose=config['verbose_param'])
    
        # Make predictions using the testing set
        regr_svr_gs.fit(X_train, y_train)
        # Feature Importance
        feature_importance = np.zeros((D, 1))
    
        print("C, epsilon= ", regr_svr_gs.best_params_['C'],
              regr_svr_gs.best_params_['epsilon'])
    
    y_train_predict = regr_svr_gs.predict(X_train)
    y_test_predict = regr_svr_gs.predict(X_test)
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    model_param = np.append(regr_svr_gs.best_params_['C'],
                            regr_svr_gs.best_params_['epsilon'])
    model_feat_imp = (feature_importance[:])
    model_param2 = np.repeat(model_param.reshape(1, model_param.shape[0]), np.shape(X_test)[0],
                             0)
    model_feat_imp2 = np.repeat(model_feat_imp.reshape(1, model_feat_imp.shape[0]),
                                np.shape(X_test)[0], 0)
    return y_train_err, y_test_err, model_feat_imp2, model_param2

def model_nn(X_train, y_train, X_test, y_test, cv, i, config, D): 
    n_dim = X_train.shape[1]
    n_classes = 1
    n_hidden_units_one = n_dim
    n_hidden_units_two = 1
    sd0 = 1 / np.sqrt(n_dim)
    sd1 = 1 / np.sqrt(n_hidden_units_one)
    sd2 = 1 / np.sqrt(n_hidden_units_two)
    y_train.shape += (1,)
    y_test.shape += (1,)
     
    print("NN ", end="")
    # Make predictions using the testing set
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])
     
    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd0))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd0))
    h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)
    
    W = tf.Variable(tf.random_normal([n_hidden_units_one, n_classes], mean=0, stddev=sd1))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd2))
    y_ = tf.matmul(h_1, W) + b
     
     # Loss Function
    mse = tf.losses.mean_squared_error(labels=Y,
                                        predictions=y_)  # simple mean squared error loss function
     
     # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(
         mse)  # BGD with Adam efficient version
    cost_history = np.empty(shape=[1], dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())  # v1.0 changes
         for epoch in range(config['training_epochs']):
             _, cost = sess.run([optimizer, mse], feed_dict={X: X_train, Y: y_train})
             cost_history = np.append(cost_history, cost)
             if config['verbose_param'] >= 1:
                 print('Epoch', epoch, 'completed out of', config['training_epochs'], 'loss:', cost)
     
         print('Epoch', epoch, 'completed out of', config['training_epochs'], 'loss:', cost)
         y_test_predict = sess.run(y_, feed_dict={X: X_test})
         y_train_predict = sess.run(y_, feed_dict={X: X_train})
         y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test,y_test_predict)
    y_train_err, y_test_err = model_analysis(y_train, y_train_predict, y_test, y_test_predict)
    return y_train_err, y_test_err