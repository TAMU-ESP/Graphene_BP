file = open('config.hjson','r')
lines = file.readlines()
file.close()

comment = {}
comment['subject_id_arr'] = "List of subjects to be analyzed"
comment['training_select']= "0:ShuffleCV_hgcp - 1:NoShuffleCV_hgcp - 2:SingleTrain_hgcp - 3:SingleTrain_WithBaseValsalva \
- 4:SingleTrain_S1All_TestNextDay - 5:ShuffleCV_valsalva - 6:NoShuffleCV_valsalva]\n\
    Choose one of the above trainings..."
comment['feature_select']= "0:WithIBI - 1:NoIBI \nChoose one of the above feature sets..."
comment['features_mean'] = "Use average features instead of beat-to-beat?"
comment['row_remove_nan_perc']="Remove sample (heart beat) if rate of missing features is larger than row_remove_nan_perc*100%"
comment['root_path'] = 'The root path that contains the folder with features data (puth the path in \"\"'
comment['feat_options'] = 'Feature file name'
comment['feature_data']= 'Folder to read features data'
comment['prediction_path'] = 'Path to store prediction results'
comment['GraBoosting_EN'] = 'Use gradient boosting model? (true/false)'
comment['XGBoosting_EN'] = 'Use xgboost model? (true/false)'
comment['lm_EN'] = 'Use linear regression model? (true/false)'
comment['tree_EN'] = 'Use decision tree model? (true/false)'
comment['ada_EN'] = 'Use adaboost model? (true/false)'
comment['randforest_EN'] = 'Use random forest model? (true/false)'
comment['svr_EN'] = 'Use support vector regression model? (true/false)'
comment['nn_EN'] = 'Use support neural network model? (true/false)'
comment['n_fold'] =  'Number of folds for cross validation'
comment['verbose_param'] =  'Print training details?'
comment['XGBoosting_grid'] =  'Parameteres of XGBoost model'
comment['ada_grid'] =  'Parameteres of ADABoost model'
comment['svr_grid'] =  'Parameteres of SVR model'
comment['GraBoosting_grid'] =  'Parameteres of gradient boosting model'
comment['tree_grid'] =  'Parameteres of decision tree model'
comment['randforest_grid'] =  'Parameteres of random forest model'
comment['training_epochs'] =  'Number of training epochs for neural network'
comment['learning_rate'] = 'Learning rate'


for i,line in enumerate(lines):
    if line[:2]=='//' or i==0 or i==len(lines)-1 or ':' not in line:
        continue
    tmp_line=line.replace(" ", "")
    tmp_line=tmp_line.replace("\n", "")
    key = tmp_line.split(':')[0]
    if key not in list(comment.keys()):
        continue
    value = tmp_line.split(':')[1]
    tmp = input(comment[key]+'\n'+key + ' ['+value+']: ')
    if tmp:
        lines[i] = key+':'+tmp+'\n'

file = open('config.hjson','w')
file.writelines(lines)
file.close()
    