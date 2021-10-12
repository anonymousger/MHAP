# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:54:41 2020

@author: 

Read a dataset and train a model for classify the data.
"""
from Data_Preprocessing import ReadData
from ConvNet_Model import ConvNet
import numpy as np
import tensorflow.keras as keras
import sys, getopt
from High_Activated_Filters_FCN import HighlyActivated
import pandas
from itertools import *
from  functools import *
from Clustering import Clustering
from matplotlib import pyplot
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt
np.random.seed(0)

def readData(data_name,dir_name):
    dir_path = dir_name + data_name+'/'
    dataset_path = dir_path + data_name +'.mat'
    
    ##read data and process it
    prepare_data = ReadData()
    prepare_data.data_preparation(dataset_path, dir_path)
    datasets_dict = prepare_data.read_dataset(dir_path,data_name)
    x_train = datasets_dict[data_name][0]
    y_train = datasets_dict[data_name][1]
    x_test = datasets_dict[data_name][2]
    y_test = datasets_dict[data_name][3]
    x_train, x_test = prepare_data.z_norm(x_train, x_test)
    nb_classes = prepare_data.num_classes(y_train,y_test)
    y_train, y_test, y_true = prepare_data.on_hot_encode(y_train, y_test)
    x_train, x_test, input_shape = prepare_data.reshape_x(x_train, x_test)  
    #create train validation subvalidation sub set
    x_training, x_validation = x_train[:90,:], x_train[90:,:]
    y_training, y_validation = y_train[:90,:], y_train[90:,:]
    
    x_training = x_train
    y_training = y_train
    
    return x_training, x_validation, x_test, y_training, y_validation, y_true, input_shape,nb_classes

def trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes):
    ##train the model
    train_model = ConvNet()
    #ResNet
    #model = train_model.networkResNet(input_shape,nb_classes)
    #FCN 
    model = train_model.network_fcN(input_shape,nb_classes)
    #cnn
    #model = train_model.network(input_shape,nb_classes)
    print(model.summary())
    train_model.trainNet(model,x_training,y_training,x_validation,y_validation,16,2000)
    return model,train_model

def predect(y_true,x_test,model,train_model,dimention_deactivated):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    keras.backend.clear_session()
    file = open('../Results/file_name.csv','a')
    file.write(str(dimention_deactivated))
    file.close()
    df_metrics = train_model.calculate_metrics(y_true, y_pred, 0.0)
    df = pandas.DataFrame(df_metrics).transpose()
    df.to_csv('../Results/file_name.csv', mode='a')
    return y_pred
    
def visulize_active_filter(model,x_test,y_true,nb_classes,train_model,cluster_centers,netLayers=3):
    ##visulize activated filters for the original testing dataset
    dimention_deactivated = 'all'
    y_pred = predect(y_true,x_test,model,train_model,dimention_deactivated)
    visulization = HighlyActivated(model,x_test,y_pred,nb_classes,netLayers=3)
    activation_layers = visulization.Activated_filters(example_id=1)
    visulization.get_high_activated_filters(activation_layers,dimention_deactivated)
    activated_class_cluster = visulization.show_high_activated_period(activation_layers,dimention_deactivated,cluster_centers)
    visulization.print_high_activated_combunation(activated_class_cluster)
    ##visulize activated filters when set all dimention of the data to zero, and just one with its original data
    x = []
    combination_id = []
    for i in range (x_test.shape[2]):
        x.append(i)
        tu = []
        tu.append(i)
        combination_id.append(tu)
    
    r = []
    for i in range(2,x_test.shape[2]):
        r.append(list(combinations(x, i)))
        
    for h in (r):
        for l in h:
            combination_id.append(l)
    print(combination_id)
    multivariate_variables = [[] for i in range(len(combination_id))]
    for i in range(len(combination_id)):
        multivariate_variables[i] = np.copy(x_test)
        for j in range(len(multivariate_variables[i])):      
            for k in range(len(multivariate_variables[i][j])):
                for n in range(x_test.shape[2]):
                    if (n not in combination_id[i]):
                        multivariate_variables[i][j][k][n] = 0
                    else:
                        dimention_deactivated =  ''.join(map(str,combination_id[i])) 
        y_pred = predect(y_true,multivariate_variables[i],model,train_model,dimention_deactivated)
        visulization = HighlyActivated(model,multivariate_variables[i],y_pred,nb_classes,netLayers=3)
        activation_layers = visulization.Activated_filters(example_id=1)
        visulization.get_high_activated_filters(activation_layers,dimention_deactivated)
        activated_class_cluster = visulization.show_high_activated_period(activation_layers,dimention_deactivated,cluster_centers)
        visulization.print_high_activated_combunation(activated_class_cluster)
        
def cluster_data_compenation(model,x_training,y_training,nb_classes):
    visulization_traning = HighlyActivated(model,x_training,y_training,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_indexes, filter_number = visulization_traning.get_high_active_period(activation_layers)
    cluster_periods = visulization_traning.extract_dimention_active_period(period_indexes)
    ##clustring the periods
    cluster_data = []
  
    cluster_number = [12,11,13]
    #clustering = Clustering(cluster_periods)
    #print(new_data)
    kshape = KShape(n_clusters=12, verbose=True, random_state=42)
    trans_x = np.nan_to_num(cluster_periods[0])
    kshape.fit(trans_x)
    cluster_centers = kshape.cluster_centers_

    cluster_data.append(cluster_centers)

    x = []
    combination_id = []
    for i in range (x_training.shape[2]):
        x.append(i)
        tu = []
        tu.append(i)
        combination_id.append(tu)
    
    r = []
    for i in range(2,x_training.shape[2]):
        r.append(list(combinations(x, i)))
        
    for h in (r):
        for l in h:
            combination_id.append(l)
    multivariate_variables = [[] for i in range(len(combination_id))]
    for i in range(len(combination_id)):
        multivariate_variables[i] = np.copy(x_training)
        for j in range(len(multivariate_variables[i])):      
            for k in range(len(multivariate_variables[i][j])):
                for n in range(x_training.shape[2]):
                    if (n not in combination_id[i]):
                        multivariate_variables[i][j][k][n] = 0
                    else:
                        dimention_deactivated =  ''.join(map(str,combination_id[i])) 
        visulization_traning = HighlyActivated(model,x_training,y_training,nb_classes,netLayers=3)
        activation_layers = visulization_traning.Activated_filters(example_id=1)
        period_indexes, filter_number = visulization_traning.get_high_active_period(activation_layers)
        cluster_periods = visulization_traning.extract_dimention_active_period(period_indexes)
       
        kshape = KShape(n_clusters=12, verbose=True, random_state=42)
        trans_x = np.nan_to_num(cluster_periods[0])
        kshape.fit(trans_x)
        cluster_centers = kshape.cluster_centers_

        cluster_data.append(cluster_centers)
    
   #save the cluster center for each layer in different array
    cluser_center1 = []
    cluser_center2 = []
    cluser_center3 = []
    cluser_center = []
    l = 0
    for i in cluster_data: 
        for j in i:
            if(l == 0):
                cluser_center1.append(j)
            elif(l == 1):
                cluser_center2.append(j)
            else:
                cluser_center3.append(j)
        l +=1
    cluser_center.append(cluser_center1)
    cluser_center.append(cluser_center2)
    cluser_center.append(cluser_center3)
    #return cluser_center

    return cluser_center


def normilization(data):
        i = 0
        datt = []
        maxi = max(data)
        mini = abs(min(data))
        while (i< len(data)):
            
            if(data[i] >=0):
                val = data[i]/maxi
            else:
                val = data[i]/mini
         
            datt.append(val)
            i += 1
            
        return datt

#compare to cluster
def fitted_cluster(data,cluster):
        data = normilization(data)
        cluster[0] = normilization(cluster[0])
        mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = normilization(i)
            dist = distance.euclidean(data,clu_nor)
            if(dist < mini):
                cluster_id = count
                mini = dist
            count+=1
            
        return cluster_id


def run(argv):
    data_name = ''
    dir_name = ''
    try:
      opts, args = getopt.getopt(argv,"hf:d:",["file_name=","directory_name="])
    except getopt.GetoptError:
      print ('Train_example_dataset.py -f <file name> -d <directory name>')
      sys.exit(2)
    print (opts)
    for opt, arg in opts:
      if opt == '-h':
         print ('Train_example_dataset.py -f <file name> -d <directory name>')
         sys.exit()
      elif opt in ("-f", "--file"):
         data_name = arg
      elif opt in ("-d", "--directory"):
         dir_name = arg
     
    #data_sets = ['ArabicDigits','AUSLAN','CharacterTrajectories','CMUsubject16','ECG','JapaneseVowels','KickvsPunch','Libras','NetFlow','PEMS','UWave','Wafer','WalkvsRun']
    #for i in data_sets:
    #x_training, x_validation, x_test, y_training, y_validation, y_true,input_shape, nb_classes = readData(i,dir_name)
    
    data_name = 'UWave'
    dir_name = '../Data/mtsdata/'
    x_training, x_validation, x_test, y_training, y_validation, y_true,input_shape, nb_classes = readData(data_name,dir_name)
    model,train_model = trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes)
    y_pred = predect(y_true,x_test,model,train_model,'alls')

    #clustring the heigh active period for the traning data
    cluster_centers = cluster_data_compenation(model,x_training,y_training,nb_classes)
    print(cluster_centers)
   
    """
    count = 0
    for j in cluster_centers:
        count += 1
        count1 = 1
        for k in j:
            fig, ax = pyplot.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.plot(k)
            name = 'images/%d_%d.png' %(count,count1)
            fig.savefig(name)   # save the figure to file
            pyplot.close(fig)    # close the figure window
            count1 +=1
    """
    
    cluster_periods = cluster_centers
    #get the count of clusters for first label
    print((cluster_periods[0][0]))
    # seprate HAP to dimentions(3 array)
    d1 = []
    d2 = []
    d3 = []
    i = 0
    while(i< len(cluster_periods[0])):
        d1.append(cluster_periods[0][i][0])
        d2.append(cluster_periods[0][i][1])
        d3.append(cluster_periods[0][i][2])
        i+=1
        
    cluster_ = []
    for i in d1:
        cluster_.append(fitted_cluster(i,new_data))
    x = []
    for i in range(0,11):
        x.append(cluster_.count(i))
    
    print('1 label first')
    print(x)
    
    cluster_ = []
    for i in d2:
        cluster_.append(fitted_cluster(i,new_data_2))
    x = []
    for i in range(0,11):
        x.append(cluster_.count(i))
    
    print('1 label second')
    print(x)
    
    cluster_ = []
    for i in d3:
        cluster_.append(fitted_cluster(i,new_data_3))
    x = []
    for i in range(0,11):
        x.append(cluster_.count(i))
    print('1 label third')
    print(x)
    #visulize heatmap
    from High_Activated_Filters import HighlyActivated
    data_vis1 = visulize_active_filter(model,x_test,y_true,nb_classes,train_model,cluster_centers,netLayers=3)
    import seaborn as sns
    ix = 1
    fig=pyplot.figure(figsize=(16, 9))
    fig.tight_layout(pad=7.0)
    for j in range(7):
        ax1 = fig.add_subplot(2,4, ix)
        #image_name = '../Results/'+str(dimention_deactivated) + ','+ str(j) +'.png'      
        if(j == 0):
            title_name = 'Signal I'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[1][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        elif(j == 1):
            title_name = 'Signal II'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[2][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        elif(j == 2):
            title_name = 'Signal III'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[3][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
    
        elif(j == 3):
            title_name = 'All Signal'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[0][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        elif(j == 4):
            title_name = 'I & II'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[4][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        elif(j == 5):
            title_name = 'I & III'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[5][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        elif(j == 6):
            title_name = 'II & III'
            ax1.set_title(title_name)
            svm = sns.heatmap(data_vis1[6][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
        ix += 1
    
    #title_name = 'ALL Signal'
    #ax1.set_title(title_name)
    #svm = sns.heatmap(data_vis1[0][2],ax=ax1,cbar_kws={"orientation": "horizontal"})
    #figure = svm.get_figure()    
    name = 'conv_heatmap_uWave.pdf' 
    fig.savefig(name,transparent=True)   # save the figure to file
    pyplot.show()  
    
    #get MHAP for test data to compare cluster
    # x_test, y_training, y_validation, y_true
    visulization_traning = HighlyActivated(model,x_test,y_true,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_indexes, filter_number = visulization_traning.get_high_active_period(activation_layers)
    cluster_periods = visulization_traning.extract_dimention_active_period(period_indexes)
if __name__ == '__main__':
    run(sys.argv[1:])