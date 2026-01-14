import polars as pl
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import numpy as np

class RLoader:
    def __init__(self,path):
        """Initialize RLoader with a given path."""
        self.path = path
    
    def load_data(self,num_classes,spk_raw = False):
        '''This is the main function to load the data'''
        # load whole session data
        data_session = self.load_whole_session(spk_raw)

        # load train data from R(I smoothed spike with sd=5 in R)
        train_data = self.load_train_data()

        # pick current train based on my_train label
        no_null_train_data = [df.filter(pl.col('y')!=6) for df in train_data]
        no_E_train_data = [df.filter(pl.col('y')!=5) for df in no_null_train_data]
        trian_data_switch = {
            6: train_data, # 6 states
            5: no_null_train_data, # 5 odors 
            4: no_E_train_data # 4 odors
        }

        train = trian_data_switch.get(num_classes)

        # load test data from R
        test_data = self.load_test_data()

        return data_session, train, test_data


    def load_whole_session(self,spk_raw = False):
        # 1. load the whole session data and compute the firing rate for the whole session
        if spk_raw:
            r_file = self.path + "EnsembleMat_raw.RData"
        else:
            r_file = self.path + "EnsembleMat_sd5.RData"
        # print(r_file)

        robjects.r(f'load("{r_file}")')  # Loads the objects into R's environment

        # Access the list object  
        r_list = robjects.r['data']
        data_session = []  # Store results as a list of tuples

        # Ensure R to Pandas conversion is active
        with localconverter(robjects.default_converter + pandas2ri.converter):
            for i, inner_list in enumerate(r_list):

                # Convert each element of the inner list to a Pandas DataFrame
                df_r = inner_list['EnsembleMatrix']
                df_pandas = robjects.conversion.rpy2py(df_r)
                df_np = df_pandas.to_numpy()
                data_session.append(df_np)
        return data_session

    def load_train_data(self):
        # Load train data .RData file
        r_file = self.path + "train_data.RData"
        robjects.r(f'load("{r_file}")')  # Loads the objects into R's environment

        # Access the list object  
        r_list = robjects.r['train_data']

        train_data = []  # Store results as a list of tuples

        # Ensure R to Pandas conversion is active
        with localconverter(robjects.default_converter + pandas2ri.converter):
            for i, inner_list in enumerate(r_list):
                # Create a dictionary to hold data for all keys
                combined_data = {}

                for key in inner_list.keys():
                    # Convert each element of the inner list to a Pandas DataFrame
                    df_r = inner_list[key]
                    df_pandas = robjects.conversion.rpy2py(df_r).to_frame()

                    # Add the data to the combined dictionary
                    combined_data[key] = df_pandas.iloc[:, 0].values  # Extract the column as a list

                # Create a Polars DataFrame from the combined data
                pl_df = pl.DataFrame(combined_data)
                train_data.append(pl_df)
        print(f"total rat number for train is:{len(train_data)}")

        return train_data

    def load_test_data(self):
        # Load test data .RData file
        r_file = self.path + "test_data.RData"
        robjects.r(f'load("{r_file}")')  # Loads the objects into R's environment

        # Access the list object  
        r_list = robjects.r['test_data']

        test_data = []  # Store results as a list of tuples

        # Ensure R to Pandas conversion is active
        with localconverter(robjects.default_converter + pandas2ri.converter):
            for i, inner_list in enumerate(r_list):
                # Create a dictionary to hold data for all keys
                combined_data = {}

                for key in inner_list.keys():
                    # Convert each element of the inner list to a Pandas DataFrame
                    df_r = inner_list[key]
                    df_pandas = robjects.conversion.rpy2py(df_r).to_frame()

                    # Add the data to the combined dictionary
                    combined_data[key] = df_pandas.iloc[:, 0].values  # Extract the column as a list

                # Create a Polars DataFrame from the combined data
                pl_df = pl.DataFrame(combined_data)
                test_data.append(pl_df)
        print(f"total rat number for test is:{len(test_data)}")
        return test_data

    # def load_raw_spk_py(self):
    #     # load k's data
    #     raw_path = ['081106_barat/081106_barat_', 
    #             '090420_buchanan/090420_buchanan_',
    #             '080718_mitt/080718_mitt_',
    #             '090212_stella/090212_stella_',
    #             '090212_superchris/090212_superchris_']
                
    #     rat_name = ['Barat','Buchanan','Mitt','Stella','SuperChris']
    #     raw_spk_list = [] 
    #     epoched_spk_list = []

    #     for irat in range(0,5):
    #         current_path = raw_path[irat]
    #         spk = np.load('raw_data/' + current_path + 'spk.npz')
    #         behav = np.load('raw_data/' + current_path + 'bvr.npz')
    #         spk_dict = {
    #             'rat': rat_name[irat], 'keys': spk['keys'], 'spk': spk['data'],
    #             'behav_keys': behav['keys'], 'behav': behav['data']}

    #         raw_spk_list.append(spk_dict) 
    
    #         epo_spk = np.load('epoched_data/' + current_path + 'tsr_w-4000_spk.npz')
    #         epo_behav = np.load('epoched_data/' + current_path + 'tsr_w-4000_bvr.npz')
    #         epo_spk_dict = {
    #             'rat': rat_name[irat], 
    #             'keys': epo_spk['keys'], 'spk': epo_spk['data'],
    #             'behav_keys': epo_behav['keys'], 'behav': epo_behav['data']}

    #         epoched_spk_list.append(epo_spk_dict) 
    #         print(rat_name[irat], 'finish spk:', spk['data'].shape, 'behav:', behav['data'].shape )
    #     return raw_spk_list,epoched_spk_list
 