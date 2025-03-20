import pandas as pd 
#from data_process_commerce import CommerceDataset
from datetime import datetime 

class ComLoader:
    def __init__(self, training):
        self.user2id = {} # good practice to have if your user number is a coded number 
        self.user_list = []
        self.time_list = []
        self.item_list = []
        self.view_time_list = []
        self.click_rate_list = [] 
        self.training = training # dictate the creation of the dataset either training or testing 
        self._process_data() 
        
        
    def _process_data(self):
        self._read_data()
        self._process_user_data() 
        self._process_other_data() 
        
    def _read_data(self):
        self.df_user_item = pd.read_csv('user_item_interactions.csv')
        self.user_id_list = self.df_user_item['user_id'].values.tolist()
        self.timestamp_list = self.df_user_item['timestamp'].values.tolist() 
        self.item_id_list = self.df_user_item['item_id'].values.tolist()
        self.view_time_value_list = self.df_user_item['view_time'].values.tolist()
        self.click_rate_value_list = self.df_user_item['click_rate'].values.tolist() 
        
    
    def _process_user_data(self):
        prev_user = self.user_id_list[0] 
        count_user = 0
        for i in range(len(self.user_id_list)):
            user_id = self.user_id_list[i]
            if user_id == prev_user:
                count_user +=1
            else:
                if count_user >= 0:
                    self.user2id[prev_user] = len(self.user2id)
                prev_user = user_id 
                
    def _process_other_data(self):
        user_time, user_item, user_view_time, user_click_rate = [], [], [], []
        prev_user = self.user_id_list[0]
        prev_user_id = self.user2id.get(prev_user)
        for i in range(len(self.user_id_list)):
            user_id = self.user2id.get(self.user_id_list[i])
            time = (datetime.strptime(self.timestamp_list[i], '%Y-%m-%d %H:%M:%S.%f')-datetime(1970,1,1)).total_seconds()
            item = self.item_id_list[i] 
            view_time = self.view_time_value_list[i] 
            click_rate = self.click_rate_value_list[i] 
            if user_id == prev_user_id:
                user_time.insert(0, time)
                user_item.insert(0, item)
                user_view_time.insert(0, view_time)
                user_click_rate.insert(0, click_rate) 
            else:
                self.user_list.append(prev_user_id)
                self.time_list.append(user_time)
                self.item_list.append(user_item)
                self.view_time_list.append(user_view_time)
                self.click_rate_list.append(user_click_rate)
                
                # restart this process again for new user_id 
                prev_user_id = user_id 
                user_time = [time]
                user_item = [item]
                user_view_time = [view_time]
                user_click_rate = [click_rate] 
        
        # perform for the last user_id 
        self.user_list.append(prev_user_id)
        self.time_list.append(user_time)
        self.item_list.append(user_item)
        self.view_time_list.append(user_item)
        self.click_rate_list.append(user_click_rate)
                
        
            
            
        
            
        # since I have read the file separately
    #def create_data_set(self, batch_size, self.training):
        #return CommerceDataset(self.user_list, self.item_list, self.time_list, self.view_time_list, self.click_rate_list, batch_size, self.training)


## 
## from here extract individual user data 
