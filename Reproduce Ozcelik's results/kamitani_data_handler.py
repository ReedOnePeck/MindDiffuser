from scipy.io import loadmat
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import preprocessing


class kamitani_data_handler():
    """Generate batches for FMRI prediction
    frames_back - how many video frames to take before FMRI frame
    frames_forward - how many video frames to take after FMRI frame
    """

    def __init__(self, matlab_file ,test_img_csv = 'KamitaniData/imageID_test.csv',train_img_csv = 'KamitaniData/imageID_training.csv',voxel_spacing =3,log = 0 ):
        mat = loadmat(matlab_file)
        self.data = mat['dataSet'][:,3:]
        self.sample_meta = mat['dataSet'][:,:3]
        meta = mat['metaData']


        self.meta_keys = list(l[0] for l in meta[0][0][0][0])
        self.meta_desc = list(l[0] for l in meta[0][0][1][0])
        self.voxel_meta = np.nan_to_num(meta[0][0][2][:,3:])
        test_img_df = pd.read_csv(test_img_csv, header=None)
        train_img_df =pd.read_csv(train_img_csv, header=None)
        self.test_img_id = test_img_df[0].values
        self.train_img_id = train_img_df[0].values
        self.sample_type = {'train':1 , 'test':2 , 'test_imagine' : 3}
        self.voxel_spacing = voxel_spacing

        self.log = log

    def get_meta_field(self,field = 'DataType'):
        index = self.meta_keys.index(field)
        if(index <3): # 3 first keys are sample meta
            return self.sample_meta[:,index]
        else:
            return self.voxel_meta[index]


    def print_meta_desc(self):
        print(self.meta_desc)

    def get_labels(self, imag_data = 0,test_run_list = None):
        le = preprocessing.LabelEncoder()

        img_ids = self.get_meta_field('Label')
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        imag = (type == self.sample_type['test_imagine'])

        img_ids_train = img_ids[train]
        img_ids_test = img_ids[test]
        img_ids_imag = img_ids[imag]


        train_labels  = []
        test_labels  =  []
        imag_labels = []
        for id in img_ids_test:
            idx = (np.abs(id - self.test_img_id)).argmin()
            test_labels.append(idx)

        for id in img_ids_train:
            idx = (np.abs(id - self.train_img_id)).argmin()
            train_labels.append(idx)

        for id in img_ids_imag:
            idx = (np.abs(id - self.test_img_id)).argmin()
            imag_labels.append(idx)

        if (test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_labels = test_labels[select]

        #imag_labels = le.fit_transform(img_ids_imag)
        if(imag_data):
            return np.array(train_labels), np.array(test_labels), np.array(imag_labels)
        else:
            return np.array(train_labels),np.array(test_labels)





    def get_data(self,normalize =1 ,roi = 'ROI_VC',imag_data = 0,test_run_list = None):   # normalize 0-no, 1- per run , 2- train/test seperatly
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        test_imag = (type == self.sample_type['test_imagine'])
        test_all  = np.logical_or(test,test_imag)

        roi_select = self.get_meta_field(roi).astype(bool)
        data = self.data[:,roi_select]

        if(self.log ==1):
            data = np.log(1+np.abs(data))*np.sign(data)


        if(normalize==1):

            run = self.get_meta_field('Run').astype('int')-1
            num_runs = np.max(run)+1
            data_norm = np.zeros(data.shape)

            for r in range(num_runs):
                data_norm[r==run] = sklearn.preprocessing.scale(data[r==run])
            train_data = data_norm[train]
            test_data  = data_norm[test]
            test_all = data_norm[test_all]
            test_imag = data_norm[test_imag]

        else:
            train_data = data[train]
            test_data  =  data[test]
            if(normalize==2):
                train_data = sklearn.preprocessing.scale(train_data)
                test_data = sklearn.preprocessing.scale(test_data)


        if(self.log ==2):
            train_data = np.log(1+np.abs(train_data))*np.sign(train_data)
            test_data = np.log(1+np.abs(test_data))*np.sign(test_data)
            train_data = sklearn.preprocessing.scale(train_data)
            test_data = sklearn.preprocessing.scale(test_data)



        test_labels =  self.get_labels()[1]
        imag_labels = self.get_labels(1)[2]
        num_labels = max(test_labels)+1
        test_data_avg = np.zeros([num_labels,test_data.shape[1]])
        test_imag_avg = np.zeros([num_labels,test_data.shape[1]])

        if(test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]

            select = np.in1d(run, test_run_list)
            test_data = test_data[select,:]
            test_labels = test_labels[select]

        for i in range(num_labels):
            test_data_avg[i] = np.mean(test_data[test_labels==i],axis=0)
            test_imag_avg[i] = np.mean(test_imag[imag_labels == i], axis=0)
        if(imag_data):
            return train_data, test_data, test_data_avg,test_imag,test_imag_avg

        else:
            return train_data, test_data, test_data_avg

    def get_voxel_loc(self):
        x = self.get_meta_field('voxel_x')
        y = self.get_meta_field('voxel_y')
        z = self.get_meta_field('voxel_z')
        dim = [int(x.max() -x.min()+1),int(y.max() -y.min()+1), int(z.max() -z.min()+1)]
        return [x,y,z] , dim