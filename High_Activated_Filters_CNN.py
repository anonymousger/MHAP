# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:24:41 2020

@author: 

This class will return the highly activated filters(feature map) for each layer in anetwork
"""
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.spatial import distance

np.random.seed(0)

class HighlyActivated:
    def __init__(self, model,test_data,y_pred,nb_classes,netLayers):
        self.model = model
        self.x_test = test_data
        self.y_pred = y_pred
        self.nb_classes = nb_classes
        self.netLayers = netLayers
        
    def Activated_filters(self,example_id):
        # we add 6 as the BatchNormalization, and activation is retured as layer
        layer_outputs = [layer.output for layer in self.model.layers[:self.netLayers+6]] 
        # Extracts the outputs of the top n layers
        # Creates a model that will return these outputs, given the model input
        activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs) 
        activations = activation_model.predict(self.x_test)
        #shows the activated filters for each layer for an example

        for i in range(0,self.netLayers+6):
            flag = False
            if i == 0:
            #or i == 1:
                activated_nodes = activations[i]
                flag = True
            #elif(i%2 == 1 and i >1):
            #    activated_nodes = activations[i]
            #    flag = True
            if(flag):
                n_filters, ix = activated_nodes.shape[2], 1
                for j in range(0,n_filters):
                        # specify subplot and turn of axis
                        ax = pyplot.subplot(n_filters, 3, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel
                        pyplot.plot(activated_nodes[example_id, :, j])
                        ix += 1
                pyplot.show()     
        return activations
                
    def get_best_distribution(sel,data):
        dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
        dist_results = []
        params = {}
        for dist_name in dist_names:
            dist = getattr(stats, dist_name)
            param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        dist_results.append((dist_name, p))
        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
        return best_dist, best_p, params[best_dist]

    def initilizatio_visualization_narrays(self,activations):
        classes_lists = [[] for i in range(self.nb_classes)]
        for i in range(len(classes_lists)):
            #we add 5 as we have the first layer input layer and last is Batch nor so we skip them
            for l in range(1,self.netLayers+6):
                #get the id of the filter layer
                #first conv start from 1
                if(l == 1):
                    activated_nodes = activations[l]  
                    x = []
                    for ii in range(activated_nodes.shape[2]):
                        x.append(0)
                    classes_lists[i].append(x)
                            
                elif(l== 4):
                    activated_nodes = activations[l]  
                    x = []
                    for ii in range(activated_nodes.shape[2]):
                        x.append(0)
                    classes_lists[i].append(x)
                
                elif(l== 7):
                    activated_nodes = activations[l]  
                    x = []
                    for ii in range(activated_nodes.shape[2]):
                        x.append(0)
                    classes_lists[i].append(x)

        return classes_lists
    
    def get_flters_activation_distribution(self, activations):
         # loop through the test dataset
        classes_lists = self.initilizatio_visualization_narrays(activations)
        for j in range(len(self.x_test)):
            #loop through each layer in the network
            activated_id = 1
            for l in range(1,self.netLayers+6):
                data = []
                flag = False
                #get the id of the filter layer
                if(l == 1):
                    activated_id = l
                    flag = True                    
                elif(l==4):
                    activated_id = l
                    flag = True   
                elif(l==7):
                    activated_id = l
                    flag = True 
                if(flag):
                    activated_nodes = activations[activated_id]
                    #get the distrubution of the values and choose the node if active or not
                    for i in range(activated_nodes.shape[2]):
                            # cheack highly activated filter where its 1-norm is larger than a threshold values
                            #kde = stats.gaussian_kde(activated_nodes[j, :, i])
                            # Calculate the integral of the kde between 10 and 20:
                            #xmin, xmax = min(activated_nodes[j, :, i]), max(activated_nodes[j, :, i])
                            #integral, err = integrate.quad(kde, xmin, xmax)
                            integral = np.sum(activated_nodes[j, :, i])
                            data.append(integral)
                
                    #x,y,z = (self.get_best_distribution(data))
                    y = np.mean(data)
                    #the threshold value fo a layer
                    #test all the filter nodes if they are larger than threshold then they are active
                    n = 0                 
                    while(n < len(data)):
                        if(np.sum(data[n]) > y):
                            if(l == 1):#the first conv layer
                                classes_lists[self.y_pred[j]][0][n] += 1
                            elif(l==4): # the second, third etc conv layer (we subtract -2 from l becoouse each conv follow by pooling layer)
                                classes_lists[self.y_pred[j]][1][n] += 1
                            elif(l==7): # the second, third etc conv layer (we subtract -2 from l becoouse each conv follow by pooling layer)
                                classes_lists[self.y_pred[j]][2][n] += 1
                            
                        n += 1      

        return classes_lists
    
    def get_high_activated_filters(self,activations,dimention_deactivated):
        #plot the heat map of activated filters
        classes_activated = self.get_flters_activation_distribution(activations)
        classes_lists = [[] for i in range(self.netLayers)]
        ix = 1
        for j in range(self.netLayers):
            ax1 = pyplot.subplot(self.netLayers, 3, ix)
            for n in range(self.nb_classes):
                classes_lists[j].append(classes_activated[n][j])
            image_name = '../Results/'+str(dimention_deactivated) + ','+ str(j) +'.png'
            svm = sns.heatmap(classes_lists[j],ax=ax1)
            #figure = svm.get_figure()    
            #figure.savefig(image_name, dpi=400)
            ix += 1
        
        pyplot.show()        
                
                
    def get_high_active_period(self, activations):
        # loop through the test dataset
        cluster_lists = [[] for i in range(self.netLayers)]
        filter_number = []
        period_x_test = []
        for j in range(len(self.x_test)):
            #if(j == 0):
                #loop through each layer in the network
                activated_id = 1
                p_sample = []
                for l in range(1,self.netLayers+6):
                    data = []
                    flag = False
                    #get the id of the filter layer
                    if(l == 1):
                        activated_id = l
                        flag = True                    
                    elif(l==4):
                        activated_id = l
                        flag = True         
                    elif(l==7):
                        activated_id = l
                        flag = True 
                    if(flag):
                        activated_nodes = activations[activated_id]
                        #get the distrubution of the values and choose the node if active or not
                        filter_number.append(activated_nodes.shape[2])

                        #loop through activartion channels
                        for i in range(activated_nodes.shape[2]):
                            for k in (activated_nodes[j, :, i]):
                                data.append(k)
                            #x,y,z = (get_best_distribution(data))
                            mean_ = np.mean(data)
                            active_period = []
                            count = 0
                            #data in channel filter
                            for k in (activated_nodes[j, :, i]):
                                if(k >= mean_):
                                    active_period.append(count)
                                count+=1

                            if(l == 1):
                                cluster_lists[0].append(active_period)
                            elif(l == 4):
                                cluster_lists[1].append(active_period)
                            elif(l == 7):
                                cluster_lists[2].append(active_period)
                period_x_test.append(cluster_lists)            
        print(len(cluster_lists))
        return cluster_lists,filter_number,period_x_test
    
    def extract_dimention_active_period(self,active_node_index):
        #the lest is order for each layer for each test data instatnt there will be (channels) 
        cluster_lists = [[] for i in range(self.netLayers)]
        #loop through layers
        l =0
        while(l < len(active_node_index)):
            filter_len = 0
            if(l == 0):
                filter_len = 31
            elif( l == 1):
                filter_len = 63
            elif(l == 2):
                filter_len = 127
            j = 0
            start_count = 0
            count = start_count
            index_x = 0
            d1 =[]
            d2 =[]
            d3 =[]
            #if(index_x == 0):
            for xe in(self.x_test[index_x]):
                        d1.append(xe[0])
                        d2.append(xe[1])
                        d3.append(xe[2])
            m = 0
            while(m < len(active_node_index[l])):
                if(count == start_count+filter_len):
                    count = start_count
                    index_x +=1
                    d1 =[]
                    d2 =[]
                    d3 =[]
                    if(index_x <=179 ):
                    #if(index_x ==0 ):
                        for xe in(self.x_test[index_x]):
                            d1.append(xe[0])
                            d2.append(xe[1])
                            d3.append(xe[2])
                else:
                    count+=1
                #print(len(active_node_index[l][m]))
                for k in (active_node_index[l][m]):
                        sample_index= k
                        j = sample_index
                        period1 = []
                        period2 = []
                        period3 = []
                        period = []
                        if(sample_index < (len(d1))-50):
                            while(j < sample_index +50):
                                period1.append(d1[j])
                                period2.append(d2[j])
                                period3.append(d3[j])
                                j+=1
                            period.append(period1)
                            period.append(period2)
                            period.append(period3)
                            #pyplot.plot(period1)
                            #pyplot.plot(period2)
                            #pyplot.plot(period3)
                            #pyplot.show()
                            cluster_lists[l].append(period)
                            
                m+=1
            l+=1

        #print(len(cluster_lists))    
        #print(len(cluster_lists[0]))
        #print(len(cluster_lists[0][0][0]))
        
        return(cluster_lists)
    
    def transform_to_same_length(self,data, max_length):
        curr_length = len(data)
        idx = np.array(range(curr_length))
        try:
            idx_new = np.linspace(0, idx.max(), max_length)
        except ValueError:  #raised if `y` is empty.
            pass
        # linear interpolation
        f = interp1d(idx, data, kind='cubic')
        new_ts = f(idx_new)
      
        return new_ts

    def normilization(self,data):
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

    def fitted_cluster(self,data,cluster):
        data = self.normilization(data)
        cluster[0] = self.normilization(cluster[0])
        mini = distance.euclidean(data,cluster[0])
        cluster_id = 0
        count = 0
        for i in (cluster):
            clu_nor = self.normilization(i)
            dist = distance.euclidean(data,clu_nor)
            if(dist < mini):
                cluster_id = count
                mini = dist
            count+=1
            
        return cluster_id
    
    def initilizatio_cluster_narrays(self,cluster_num):
        classes_lists = [[] for i in range(self.nb_classes)]
        for i in range(len(classes_lists)):
            for l in range(len(cluster_num)):
                    x = []
                    for ii in range(len(cluster_num[l])):
                        x.append(0)
                    classes_lists[i].append(x)
        return classes_lists
    
    def show_high_activated_period(self,activations,dimention_deactivated,cluster_centers):
        classes_lists = self.initilizatio_cluster_narrays(cluster_centers)
        for j in range(len(self.x_test)):
            #loop through each layer in the network
            activated_id = 1
            for l in range(1,self.netLayers+6):
                data = []
                flag = False
                #get the id of the filter layer
                cluster_active = 0
                if(l == 1):
                    activated_id = l
                    cluster_active = 0
                    flag = True                    
                elif(l==4):
                    activated_id = l
                    cluster_active = l-2
                    flag = True  
                elif(l==7):
                    activated_id = l
                    cluster_active = l-2
                    flag = True
                if(flag):
                    activated_nodes = activations[activated_id]
                    Len_period = len(cluster_centers[cluster_active][0])
                    #get the distrubution of the values and choose the node if active or not
                    for i in range(activated_nodes.shape[2]):
                        for k in (activated_nodes[j, :, i]):
                            data.append(k)
                        #x,y,z = (get_best_distribution(data))
                        mean_ = np.mean(data)
                        active_period = []
                        for k in (activated_nodes[j, :, i]):
                            if(k >= mean_):
                                active_period.append(k)
                        ##Get the activated perid in layer l, filter i
                        ##compare it with the cluster center of layer l, added to cluster_class array
                        
                        if(len(active_period) >4):
                            if(len(active_period) != Len_period):
                                active_period = self.transform_to_same_length(active_period,Len_period)
                            
                            n = self.fitted_cluster(active_period,cluster_centers[cluster_active])
                            if(l == 1):#the first conv layer
                                classes_lists[self.y_pred[j]][0][n] += 1
                            elif(l == 4): # the second, third etc conv layer (we subtract -2 from l becoouse each conv follow by pooling layer)
                                classes_lists[self.y_pred[j]][1][n] += 1  
                            elif(l == 7): # the second, third etc conv layer (we subtract -2 from l becoouse each conv follow by pooling layer)
                                classes_lists[self.y_pred[j]][2][n] += 1  
        
        #plot the activation map period-class
        classes_activated = classes_lists
        classes_lists = [[] for i in range(self.netLayers)]
        ix = 1
        for j in range(self.netLayers):
            ax1 = pyplot.subplot(self.netLayers, 3, ix)
            for n in range(self.nb_classes):
                classes_lists[j].append(classes_activated[n][j])
            svm = sns.heatmap(classes_lists[j],ax=ax1)
            ix += 1
        pyplot.show()       
        
        #plot the activation map period-channel    
        classes_lists = [[] for i in range(self.netLayers)]
        classes_lists[0] = [x + y for x, y in zip(classes_activated[0][0], classes_activated[1][0])]
        classes_lists[1] = [x + y for x, y in zip(classes_activated[0][1], classes_activated[1][1])]
        for j in classes_lists:
           #pyplot.hist(j,bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
           i = 0
           while (i < len(j)):
               pyplot.bar(i+1,j[i])
               i +=1
           pyplot.show() 
           
        return classes_activated
    
    def print_high_activated_combunation(self,period_lists_class):
        #for each class see what are most frequent compentatio of period
        #get the top 4 from first channel and top 6 from second channel
        data_cp = period_lists_class
        classes_lists = [[] for i in range(self.nb_classes)]
        count  = 0
        for i in period_lists_class:
            for j in range(len(i)):
                 #ind = np.argpartition(i[j], -4)[-4:]
                 if(j == 0):
                     ind = np.argpartition(i[j], -4)[-4:]
                     classes_lists[count].append(ind)
                 else:
                     ind = np.argpartition(i[j], -6)[-6:]
                     classes_lists[count].append(ind)
            
            count +=1
        print(classes_lists)
		
	def get_period_index(activations,x_test,netLayers=3):
        cluster_lists = [[] for i in range(netLayers)]
        period_x_test = []
        
        for j in range(len(x_test)):
                #loop through each layer in the network
                activated_id = 1
                for l in range(1,netLayers+6):
                    data = []
                    flag = False
                    #get the id of the filter layer
                    if(l == 1):
                        activated_id = l
                        flag = True                    
                    elif(l==4):
                        activated_id = l
                        flag = True         
                    elif(l==7):
                        activated_id = l
                        flag = True 
                    if(flag):
                        activated_nodes = activations[activated_id]
                        #loop through activartion channels
                        for i in range(activated_nodes.shape[2]):
                            #go through each time sample of data
                            for k in (activated_nodes[j, :, i]):
                                data.append(k)
                            mean_ = np.mean(data)
                            active_period = []
                            count = 0
                            #index of max activated
                            per= 0
                            #data in channel filter
                            max_active = activated_nodes[j, :, i][0]
                            for k in (activated_nodes[j, :, i]):
                                if(k >= mean_):
                                    if(k > max_active):
                                        max_active = k
                                        per = count
                                count+=1
                            active_period.append(per)

                            if(l == 1):
                                cluster_lists[0].append(active_period)
                            elif(l == 4):
                                cluster_lists[1].append(active_period)
                            elif(l == 7):
                                cluster_lists[2].append(active_period)
                period_x_test.append(cluster_lists)            
        return period_x_test
		
	def interpret_input_x_data(x_test,period_x_test, Guss_factor,filter_len_1):
    x_test_new = x_test
    x_test_1 = [[]]
    filter_len_1 = 450
    #first layer filter len = 32, second = 64, third = 128
    series = [gauss(0.0, Guss_factor) for i in range(filter_len_1)]
    i = 0
    while i < len(x_test):
        series = [gauss(0.0, Guss_factor) for i in range(filter_len_1)]
        #get the index of test sample i for layer j
        index = 700
        k = 0
        test_ = []
        j = 0
        while (k < len(x_test[i])):
            if(index <= k < (index+filter_len_1)):
                d = 0
                arr = [0] * x_test.shape[2]
                while (d < x_test.shape[2]):
                    #print(x_test[i][k][d])
                    arr[d] = x_test[i][k][d] + series[j]
                    d +=1
                j +=1
            else:
                d = 0
                arr = [0] * x_test.shape[2]
                while (d < x_test.shape[2]):
                    #print(x_test[i][k][d])
                    arr[d] = x_test[i][k][d]
                    d +=1
            test_.append(arr)
            k +=1
        
        x_test_1.append(test_)
       
        i +=1
        
    return x_test_1
                