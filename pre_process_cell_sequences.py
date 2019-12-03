import os
directory = 'F:\Cell_data_public\Albany\C2C12'

from os import listdir
from os.path import isfile, join

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import find_contours
from skimage import feature
from copy import deepcopy
from scipy.signal import argrelextrema

from skimage.filters import threshold_otsu, threshold_adaptive, threshold_yen, threshold_isodata, threshold_li

"""import cv2
import sys
#sys.path.append("../pySaliencyMap")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pySaliencyMap
"""
from scipy.ndimage import binary_dilation


def save_img(img, path):
    misc.imsave(path, img)
    
def enhance_images():
    save_path = os.path.join(directory, 'enhanced')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_subdir = os.path.join(directory, 'C2C12_seq1')
    file_list = [f for f in listdir(file_subdir) if isfile(join(file_subdir, f))]
    print len(file_list)
    count = 0
    for f in file_list:
        path = os.path.join(file_subdir, f)
        new_path = os.path.join(save_path, f)
        count += 1
        if count>0:
            img = np.array(misc.imread(path)).astype(np.float32)
            min = np.amin(img)
            max = np.amax(img)
            img = img*255/max
            
            save_img(img, new_path)
            """#img = int(255*(img-min)/(max-min))
            print img
            #img = (img*255)/np.amax(img)
            print img.shape, 'img shape'
            print img.dtype
            print np.amax(img)
            plt.imshow(img)
            plt.show()
            """

def jump_interval(directory):
    from shutil import copyfile
    read_path = os.path.join(directory, 'enhanced')
    save_path = os.path.join(directory, 'intervalized_3')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_list = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    print len(file_list)
    for i in range(0,len(file_list)):
        if i%3 == 0:
            print i
            copyfile(join(read_path, file_list[i]), join(save_path, file_list[i]))
            
def show_patch():
    #path = 'F:\Cell_data_public\Albany\C2C12\intervalized_3\\0397.tif'
    path = 'F:\Cell_data_public\Albany\BAEC\Mitosis-BAEC-F0001\BAEC_seq1\\007.tif'
    img = np.array(misc.imread(path))
    patch = img[110:170, 180:240]
    print np.sum(patch)
    plt.imshow(patch, cmap='Greys_r')
    plt.show()
    
    
def show_patch_sequence():
    path = 'F:\Cell_data_public\processed_data_for_training\\big_division\\progress.npz'
    data = np.load(path)
    seq = data['input_raw_data']
    fig = plt.figure()
    for i in range(0, 18):
        start_frame = 20
        im = seq[start_frame+i,0,:,:]
        print im.shape
        print i, np.amin(im), np.amax(im), np.mean(im)
        a = fig.add_subplot(3,6,i+1)
        imgplot = plt.imshow(im)
    plt.show()
    

def show_output_sequence():
    i, b = (5, 0)
    
    input_path = 'F:\Cell_data_public\processed_data_for_training\\from server\\division\\progress.npz'
    input = np.load(input_path)
    seq = np.reshape(input['input_raw_data'], (16,20,64,64))
    seq = seq[b,:,:,:]
    if True:
        print seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(5,15):
            im = seq[j]
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(im, cmap='Greys_r')
    #plt.show()
    
    
    path = 'F:\Cell_data_public\processed_data_for_training\\from server\\division-32-32_32-32_pred_10\\epoch_14.npz'
    data = np.load(path)
    
    if True:
        seq = data['values'][0,:,b]
        fig = plt.figure()
        print seq[0].shape, 'khut'
        for j in range(seq.shape[0]):
            im = seq[j]
            im = np.reshape(im, (2,2,32,32))
            im = np.transpose(im, (2,0,3,1))
            im = np.reshape(im, (64,64))
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(3,6,j+1)
            imgplot = plt.imshow(im)

        seq = data['values'][1,:,b]
        print seq.shape, 'shape'
        fig = plt.figure()
        
        for j in range(seq.shape[0]):
            im = seq[j,:,:]
            im = np.reshape(im, (2,2,32,32))
            im = np.transpose(im, (2,0,3,1))
            im = np.reshape(im, (64,64))
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(3,6,j+1)
            imgplot = plt.imshow(im)
        
        seq = data['values'][2,:,b]
        fig = plt.figure()
        for j in range(seq.shape[0]):
            im = seq[j,:,:]
            im = np.reshape(im, (2,2,32,32))
            im = np.transpose(im, (2,0,3,1))
            im = np.reshape(im, (64,64))
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(3,6,j+1)
            imgplot = plt.imshow(im)
    plt.show()
        

def show_output_sequence_2():
    i, b = (6, 8) # 2
    
    input_path = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame_backup_0\\progress.npz'
    input = np.load(input_path)
    print input['input_raw_data'].shape
    seq = np.reshape(input['input_raw_data'], (input['input_raw_data'].shape[0]/20,20,64,64))
    seq = seq[b,:,:,:]
    
    raw_seq = np.zeros_like(seq)
    if True:
        
        print seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(5, 15):
            im = seq[j]*255
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(im, cmap='Greys_r')
            plt.axis('off')
            raw_seq[j] = im
    
    
    path = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame_backup_0\\epoch_100.npz'
    data = np.load(path)
    
    if True:
        #print data['outputs'].shape
        #seq = data['values'][0][:,b,:,:,:]
        #seq = np.vstack((seq, data['values'][1][:,b,:,:,:]))
        
        #seq = data['values'][:,b,:,:,:]
        #seq = np.vstack((seq, data['values'][:,b,:,:,:]))
        
        
        seq2 = data['values1'][:,b,:,:,:]
        #seq2 = data['outputs'][8,:,b,:,:,:]
        
        seq3 = deepcopy(seq2)
        print seq2.shape, 'seq2'
        seq2 = np.sum(seq2, axis=1)
        seq3 = np.amax(seq3, axis=1)
        print seq2.shape, seq3.shape, 'seq2 3'
        seq2 = np.reshape(seq2, (seq2.shape[0], 64,64))
        seq3 = np.reshape(seq3, (seq3.shape[0], 64,64))
        print seq2.shape, seq3.shape, 'seq2 3'
        
        #seq2 = np.vstack((seq2, seq3))
        fig = plt.figure()
        for j in range(seq2.shape[0]):
            #seq2[j] = gaussian_filter(seq2[j], sigma=4)
            seq2[j] = seq2[j]
            print np.max(seq2[j]), np.min(seq2[j]), np.mean(seq2[j]), 'jiji'
            a = fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(seq2[j], cmap='Greys_r')
        
        
        seq = data['values'][:,b,:,:,:]
        #seq = np.vstack((seq, data['values1'][i,b,:20,:,:]))
        
        print seq.shape, 'shapehspae'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        sum_im = None
        
        seq1 = np.array([seq[0]])
        for x in range(1,seq.shape[0]):
            seq1 = np.vstack((seq1, np.array([seq[x]])))
        
        
        print seq1.shape, '1dis'
        seq2 = deepcopy(seq1)
        

        seq2 = np.reshape(seq2, (seq2.shape[0], 64,64))
        for j in range(seq2.shape[0]):
            #seq2[j] = gaussian_filter(seq2[j], sigma=4)
            seq2[j] = seq2[j]
            print np.max(seq2[j]), np.min(seq2[j]), np.mean(seq2[j]), 'jiji--'
            a = fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(seq2[j], cmap='Greys_r')
            plt.axis('off')
        
        
        if False:
            seq3 = np.zeros_like(seq2)
            masked_seq = np.zeros_like(seq2)
            for j in range(seq2.shape[0]):
                j += 1
                im = seq2[j-1]
                im = im*255
                
                global_thresh = threshold_otsu(im)
                binary_global = im < global_thresh
                
                block_size = 21
                binary_adaptive = threshold_adaptive(im, block_size, method='gaussian', offset=4)
                if np.amax(binary_adaptive)==np.amin(binary_adaptive):
                    binary_adaptive = np.ones_like(binary_adaptive)
                
                #im = binary_adaptive*-1 + 1
                #print j, np.amin(im), np.amax(im), im[3,3], 'eoeoeoe'
            
            
            if False:
                im = binary_adaptive*-1 + 1
                contours = find_contours(im, 0)
                masked_raw_im = np.zeros_like(im, dtype='float64')
                #print contours
                for c in contours:
                    mask = np.zeros_like(im)
                    c_1 = np.array(c)
                    c_y = int((np.max(c_1[:,0]) + np.min(c_1[:,0]))/2)
                    c_x = int((np.max(c_1[:,1]) + np.min(c_1[:,1]))/2)
                    for pair in c:
                        pair_2 = c[c[:,1]==pair[1]]
                        p_min = np.min(pair_2[:,0])
                        p_max = np.max(pair_2[:,0])
                        if c.shape[0]>10:
                            im[p_min:p_max+1,pair[1]] = c.shape[0]
                            mask[p_min:p_max+1,pair[1]] = 1
                        else:
                            im[p_min:p_max+1,pair[1]] = 0
                    #print c.shape[0]
                    crop = raw_seq[j-1+5]*mask
                    average = crop.sum() / mask.sum()
                    #print average,'--', mask.sum(), 'average', j-1
                    crop = average*mask
                    if mask.sum()>0 and mask.sum()>50:
                        masked_raw_im += crop
                    
                        """if j-2>=0 and masked_seq[j-2, c_y, c_x]==0:
                            crop_before = raw_seq[j-1+4]*mask
                            average = crop_before.sum() / mask.sum()
                            masked_seq[j-2] += average*mask
                            print average, 'average==========================='
                            """
                masked_seq[j-1] = masked_raw_im
                    
                #print np.unique(im)
                
                im[im>0] = 1
                min1 = j-3 if j-3>0 else 0
                min2 = j-4 if j-4>0 else 0
                max1 = j-1 if j-1<seq2.shape[0] else seq2.shape[0]
                max2 = j+2 if j+2<seq2.shape[0] else seq2.shape[0]
                im_1 = np.sum(np.vstack((seq3[min1:max1], [im])), axis=0)
                im_2 = np.sum(np.vstack((seq3[min2:max1], [im])), axis=0)
                #print np.amax(im_1), np.amin(im_1), 'im_1'
                im_1[im_1!=3] = 0
                im_1[im_2!=3] = 0
                seq3[j-1] = im
                
                #im = deepcopy(im_1)
                im = masked_raw_im
            
            
            a = fig.add_subplot(4,5,j)
            imgplot = plt.imshow(im, cmap='Greys_r')
            #plt.axis('off')
            
    
    if False:
        for i in range(masked_seq.shape[1]):
            for j in range(masked_seq.shape[2]):
                line = masked_seq[:,i,j]
                if np.max(line)>0:
                    binary_line = line>0
                    line[line==0] = 10000
                    maxima = argrelextrema(line, np.greater)
                    
                    minima = argrelextrema(line, np.less)
                    if len(maxima[0])==1 and masked_seq[:,i,j][maxima[0]][0] != 10000:
                        """min =  masked_seq[:,i,j][minima]
                        max = masked_seq[:,i,j][maxima[0]]
                        perc = (np.max(max) - np.min(min))/np.max(max)*100
                        i_min = i-15 if i-15>=0 else 0
                        i_max = i+15 if i+15<=masked_seq.shape[1] else masked_seq.shape[1]
                        j_min = j-15 if j-15>=0 else 0
                        j_max = j+15 if j+15<=masked_seq.shape[2] else masked_seq.shape[2]
                        """
                        """"a = raw_seq[maxima[0][0]+5]
                        mask_reversed = deepcopy(masked_seq[maxima[0][0]])
                        mask_reversed[mask_reversed>0] = -1
                        mask_reversed += 1
                        a = a*mask_reversed
                        """
                        #print masked_seq[:,i,j][maxima[0]], maxima[0], i, j#, '--', np.percentile(a[i_min:i_max, j_min:j_max], 50)
                        pass
    
    plt.show()
    
def show_short_seq(z,y,x):
    raw_seq = np.load('F:\Cell_data_public\processed_data_for_training\\big_division\\raw_sequence.npz')
    raw_seq = raw_seq['sequence']
    #raw_seq = np.reshape(raw_seq, (raw_seq.shape[0], raw_seq.shape[2], raw_seq.shape[3]))
    print raw_seq.shape, 'raw'
    fig = plt.figure()
    count = 0
    for i in range(8):
        count += 1
        im = raw_seq[z-4+i, y-50:y+50, x-50:x+50]
        print im.shape, 'im'
        a = fig.add_subplot(4,5,count)
        imgplot = plt.imshow(im, cmap='Greys_r')
    plt.show()
    

def get_kmeans_index(folder, train_data, predict_data, n_clusters=5, cluster_index=[1], kmeans_file=False):
    print np.max(train_data), np.min(train_data), 'train'
    print np.max(predict_data), np.min(predict_data), 'predict'
    
    from sklearn.cluster import KMeans
    import cPickle
    if not kmeans_file:
        kmeans = KMeans(n_clusters=n_clusters).fit(train_data)
        cPickle.dump(kmeans, open(folder+'\\kmeans.p', 'wb'))
    else:
        kmeans = cPickle.load(open(kmeans_file, 'rb'))
    train_labels = kmeans.predict(train_data)
    labels = kmeans.predict(predict_data)
    labels += 1; train_labels += 1
    labels_copy = deepcopy(labels)
    unique_labels = np.unique(labels)
    print labels
    print np.max(labels), np.min(labels), labels.shape, 'labels'
    max=0; min=1; max_ind=0; min_ind=0
    ind_value = []
    for i in unique_labels:
        a = train_data[train_labels==i]
        if len(a)>0:
            ind_value.append(np.max(a))
            if np.max(a)>max:
                max = np.max(a)
                max_ind = i
            if np.min(a)<min:
                min = np.min(a)
                min_ind = i
            #print np.max(a), np.min(a), a.shape, 'unique ))))', i
    print min_ind, 'min_ind', max_ind, 'max_ind'
    order = np.argsort(ind_value)
    ind_value = np.array(ind_value)
    ind_value = ind_value[order]
    if not kmeans_file:
        cPickle.dump(ind_value, open(folder+'\\ind_value.p', 'wb'))
    else:
        ind_value = cPickle.load(open(folder+'\\ind_value.p', 'rb'))
    
    print ind_value, 'ind_value'
    
    """
    for v in range(predict_data.shape[0]):
        a = predict_data[v][0]
        diff = ind_value - a
        diff[diff<0] = 100
        ind = np.argmin(diff)
        predict_data[v][0] = ind
        if ind in cluster_index:
            print a, ind
    labels_1 = predict_data + 1
    
    
    #unique_labels = unique_labels[order]
    #print unique_labels, labels_copy.shape, 'abc'
    #cluster_index = unique_labels[cluster_index]
    print cluster_index, 'cluster_index ===='
    print np.unique(labels_1), 'labels_copy'
    for i in cluster_index:
        labels_1[labels_1==i] = 100
    labels_1[labels_1<100] = 0
    labels_1[labels_1==100] = 1
    """
    
    if np.min(cluster_index)==0:
        min = 0
    else:
        min = ind_value[np.min(cluster_index)]
    max = ind_value[np.max(cluster_index)]
    predict_data[predict_data<min] = 0
    predict_data[predict_data>=max] = 0
    predict_data[predict_data>0] = 1
    return predict_data
    
    
def show_classes_ordered_seq():
    i, b = (6, 20)
    
    input_path = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_1e-3_supervised\\ordered_seq.npz'
    input = np.load(input_path)
    #print input['input_raw_data'].shape
    print input.keys()
    data = input['output_raw_data']
    print data.shape
    raw_seq = np.reshape(data, (data.shape[0]/20,20,64,64))
    raw_seq = raw_seq[b,:,:,:]
    if True:
        print raw_seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, 20):
            im = raw_seq[j]#*255
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            plt.imshow(im, cmap='Greys_r')
            plt.axis('off')
            
    data = input['input_raw_data']
    raw_seq = np.reshape(data, (data.shape[0]/20,20,64,64))
    raw_seq = raw_seq[b,:,:,:]
    if True:
        print raw_seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, 20):
            im = raw_seq[j]#*255
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            plt.imshow(im, cmap='Greys_r')
            plt.axis('off')
    
    
    
    folder = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_1e-3_supervised'
    
    path = folder+'\\map_epoch_ordered_sequence_epoch-99-111_supervised.npz'
    data = np.load(path)
    #print data['values'].shape
    #outputs = data['outputs'][0,:,b,:,:,:]
    
    a = data['outputs']
    print a.shape
    a = np.reshape(a, (66,10,16,64,64))
    a = np.transpose(a, (0,2,1,3,4))
    a = np.reshape(a, (66*16, 10, 64,64))
    seq2 = a[b,:,:,:]
    seq2 = np.reshape(seq2, (seq2.shape[0], 64,64)); #seq2[seq2<0.3] = 0
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for j in range(10):#seq2.shape[0]):
            fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(seq2[j], cmap='Greys_r')
            plt.axis('off')
            print j, np.amin(seq2[j]), np.amax(seq2[j]), np.mean(seq2[j])
    
    plt.show()
    

def show_classes():
    i, b = (6, 3)
    frame_no = 20
    
    input_path = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_extra\\progress.npz'
    input = np.load(input_path)
    print input['input_raw_data'].shape, 'input'
    print input.keys()
    data = input['output_raw_data']
    print data.shape, 'data'
    raw_seq = np.reshape(input['input_raw_data'], (data.shape[0]/frame_no,frame_no,64,64))
    print raw_seq.shape, 'raw_seq'
    raw_seq = raw_seq[b,:,:,:]
    if True:
        print raw_seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, frame_no):
            im = raw_seq[j]#*255
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            plt.imshow(im, cmap='Greys_r')
            # plt.axis('off')
    
    """
    data = input['input_raw_data']
    raw_seq = np.reshape(data, (data.shape[0]/frame_no,frame_no,64,64))
    raw_seq = raw_seq[b,:,:,:]
    if True:
        print raw_seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, frame_no):
            im = raw_seq[j]#*255
            print j, np.amin(im), np.amax(im), np.mean(im)
            a = fig.add_subplot(4,5,j+1)
            plt.imshow(im, cmap='Greys_r')
            #plt.axis('off')
    """
    
    
    folder = 'F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_extra'
    """input_path = folder+'\\progress.npz'
    input = np.load(input_path)
    seq = np.reshape(input['output_raw_data'], (16,20,64,64))
    seq = seq[b,:,:,:]
    """
    
    
    path = folder+'\\epoch_99.npz'
    data = np.load(path)
    print data['values'].shape
    print data.keys()
    # outputs = data['outputs'][0,:,b,:,:,:]
    
    seq2 = data['values'][:,b,:,:,:]
    print seq2.shape, 'seq2.shape'
    seq2 = np.reshape(seq2, (seq2.shape[0], 64,64)); #seq2[seq2<0.3] = 0
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for j in range(seq2.shape[0]):
            a = fig.add_subplot(4,5,j+1)
            imgplot = plt.imshow(seq2[j], cmap='Greys_r')
            #plt.axis('off')
            print j, np.amin(seq2[j]), np.amax(seq2[j]), np.mean(seq2[j])
    
    
    if True:
        print data['values1'].shape, "data['values1']"
        for i in range(16): #range(16):
            #seq2 = data['values1'][i-2:i+3,b,4,:,:] # F0001 - 4, F0005 - 7
            
            seq2 = data['values1'][:,b,i,:,:]
            #seq2[seq2<float(12)/16] = 0
            
            # seq2 = outputs[:,i,:,:]
            print seq2.shape, 'aiku'
            seq2 = np.reshape(seq2, (seq2.shape[0], 64,64))
            
            #seq = data['values'][:,b,:,:,:]
            
            if i==9 and False:
                shape = data['values1'].shape
                train_data = deepcopy(data['values1'][:,:,i,:,:])
                train_data = np.reshape(train_data, (shape[0]*shape[1]*shape[3]*shape[4], 1))
                predict_data = np.reshape(seq2, (seq2.shape[0]*seq2.shape[1]*seq2.shape[2], 1))
                
                labels = get_kmeans_index(folder, train_data, predict_data, n_clusters=30, cluster_index=[0,1], kmeans_file=folder+'\\kmeans.p') # folder+'\\kmeans.p'
                print labels.shape, 'labels.shape'
                labels = np.reshape(labels, seq2.shape)
                seq2 = seq2 * labels
            
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            for j in range(seq2.shape[0]):
                im = seq2[j]*255
                
                im[im>0] = 1
                im = binary_dilation(im, iterations=4)
                im = im*raw_seq[j+5]
                
                """event = deepcopy(seq2[j])
                event[event>0] = 1
                event = gaussian_filter(event, sigma=5)
                #event[event>0.3] = 1
                block_size = 21
                #event = threshold_adaptive(event, block_size, method='gaussian', offset=2)
                im = im*event
                """
                
                print np.max(seq2[j]), np.min(seq2[j]), np.mean(seq2[j]), 'jiji--', i, j
                #print np.max(im), np.min(im), np.mean(im), 'cucu'
                a = fig.add_subplot(4,5,j+1)
                #seq2[j] = binary_dilation(seq2[j], iterations=2)
                imgplot = plt.imshow(seq2[j], cmap='Greys_r')
                #a = fig.add_subplot(4,5,j+11)
                #imgplot = plt.imshow(im, cmap='Greys_r')
                plt.axis('off')
    plt.show()
    
    
def show_index_map():
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk, opening, square
    from skimage.feature import canny, hog
    
    raw_seq = np.load('F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_1e-3_supervised\\raw_sequence_end.npz')
    raw_seq = raw_seq['sequence']
    data = np.load('F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_1e-3_supervised\\big_map.npz')
    data = data['big_map']; #data = np.transpose(data, (0,2,1))
    
    #gt = np.load('F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame-F0001_1e-3_supervised\\big_map.npz')
    #gt = gt['big_map']
    
    mask = disk(4)
    threshold = np.percentile(raw_seq, 95)
    #raw_seq[raw_seq<threshold] = 0
    
    for i in range(120,130):
        #print i, np.max(data[i-111,:,:]), np.min(data[i-111,:,:]), np.mean(data[i-111,:,:])
        
        #a = binary_dilation(data[i-111,:,:], iterations=13)
        a = binary_dilation(data[i-111,:,:], iterations=10)
        #a = data[i-111,:,:]
        print a.shape, raw_seq.shape
        #a = deepcopy(data[i-111,:,:])
        im = a * raw_seq[i-111,0:data.shape[1]*4:4,0:data.shape[2]*4:4]
        #im = raw_seq[i-111,0:80,0:80]
        #im = im[::4,::4]
        plt.figure()
        plt.imshow(im, cmap='Greys_r')
        
        """
        newarr, im1 = hog(im, visualise=True, pixels_per_cell=(4,4), cells_per_block=(1,1))
        print newarr
        print newarr.shape
        plt.figure()
        plt.imshow(im1, cmap='Greys_r')
        """
        
        """
        a = binary_dilation(gt[i-111,:,:], iterations=12)
        im = a * raw_seq[i-111,0:data.shape[1]*4:4,0:data.shape[2]*4:4]
        plt.figure()
        plt.imshow(im, cmap='Greys_r')
        """
        
        """
        im1 = deepcopy(im)
        im1[im1<threshold] = 0
        im1 = opening(im1, square(3))
        plt.figure()
        plt.imshow(im1, cmap='Greys_r')
        
        a = binary_dilation(data[i-111,:,:], iterations=6)
        print i, np.max(a), np.min(a), np.mean(a)
        im = a * raw_seq[i-111,0:data.shape[1]*4:4,0:data.shape[2]*4:4]
        im[:9,:9] = mask
        plt.figure()
        plt.imshow(im, cmap='Greys_r')
        """
        
    plt.show()


import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def test_filter():
    from skimage.filters import scharr, threshold_adaptive, threshold_otsu
    from skimage.morphology import opening
    from skimage.transform import hough_circle
    from skimage.feature import canny, hog, local_binary_pattern, ORB, daisy
    from skimage.morphology import square
    
    raw_seq = np.load('F:\Cell_data_public\processed_data_for_training\\from server\\divbi-classification-single-frame\\raw_sequence_111.npz')
    raw_seq = raw_seq['sequence']
    for i in range(155,156):
        im = deepcopy(raw_seq[i-111,::4,::4] - raw_seq[i+1-111,::4,::4])
        #im = raw_seq[i-111,::4,::4]
        #im = canny(im, sigma=1)
        #threshold = threshold_otsu(im)
        #threshold = np.percentile(raw_seq[i-111], 90)
        #im[im<threshold] = 0
        plt.figure()
        plt.imshow(raw_seq[i-111,::4,::4], cmap='Greys_r')
        
        """
        plt.figure()
        im = raw_seq[i-111] - raw_seq[i+1-111]
        threshold = threshold_otsu(im)
        im[im<threshold] = 0
        plt.imshow(im, cmap='Greys_r')
        """
        
        
        """
        edges = canny(raw_seq[i-111], sigma=3, low_threshold=10, high_threshold=50)
        hough_radii = np.arange(2, 8, 2)
        hough_res = hough_circle(im, hough_radii)
        plt.figure()
        plt.imshow(hough_res[0,:], cmap='Greys_r')
        plt.figure()
        plt.imshow(hough_res[1,:], cmap='Greys_r')
        plt.figure()
        plt.imshow(hough_res[2,:], cmap='Greys_r')
        #plt.imshow(edges, cmap='Greys_r')
        print hough_res[0,:]
        """
        
        
        """
        hog_im, img = hog(im, visualise=True, pixels_per_cell=(4,4), cells_per_block=(1,1))
        plt.figure()
        plt.imshow(img, cmap='Greys_r')
        print hog_im
        """
        
        """
        det_ext = ORB(n_keypoints=5)
        det_ext.detect_and_extract(im)
        orb_ft = det_ext.descriptors
        print orb_ft
        print orb_ft.shape, np.sum(orb_ft)
        """
        
        """
        descs, im1 = daisy(im, radius=4, visualize=True)
        plt.figure()
        plt.imshow(im1, cmap='Greys_r')
        print descs
        print descs.shape, im1.shape
        """
        
        
        """
        lbp = local_binary_pattern(im, 2, 4)
        plt.figure()
        plt.imshow(lbp, cmap='Greys_r')
        print lbp
        print lbp.shape
        """
        
        
        """
        plt.figure()
        im1 = deepcopy(im)
        im1 = opening(im1, square(1))
        plt.imshow(im1, cmap='Greys_r')
        
        plt.figure()
        im1 = deepcopy(im)
        im1 = opening(im1, square(3))
        plt.imshow(im1, cmap='Greys_r')
        
        plt.figure()
        im1 = deepcopy(im)
        im1 = opening(im1, square(4))
        plt.imshow(im1, cmap='Greys_r')
        """
        
    plt.show()


def show_progress_seq():
    i, b = (6, 3)
    frame_no = 10

    input_path = 'F:\Cell_data_public\\Albany\\from_server\\progress.npz'
    input = np.load(input_path)
    print input['input_raw_data'].shape, 'input'
    print input.keys()
    output = input['output_raw_data']
    print output.shape, 'output'
    raw_seq = input['input_raw_data']
    print raw_seq.shape, 'raw_seq'
    raw_seq = raw_seq[b, :, :, :]
    output_seq = output[b, :, :, :]
    if True:
        print raw_seq.shape[0], 'dis'
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, frame_no):
            im = raw_seq[j]  # *255
            print j, np.amin(im), np.amax(im), np.mean(im), im.shape
            a = fig.add_subplot(4, 5, j + 1)
            plt.imshow(im, cmap='Greys_r')

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, frame_no):
            im = output_seq[j]  # *255
            print np.where(im == 1)
            print j, np.amin(im), np.amax(im), np.mean(im), im.shape
            a = fig.add_subplot(4, 5, j + 1)
            plt.imshow(im, cmap='Greys_r')
    plt.show()

#test_filter()

#enhance_images()
#jump_interval(directory)
#show_patch()
#show_patch_sequence()
#show_output_sequence()

#show_output_sequence_2()

# show_classes()

#show_classes_ordered_seq()

#show_short_seq(33,216,460)

# show_index_map()
show_progress_seq()