## this is the script to preprocess the data from the file user_item_interactions.csv
# - work on updating the script, the script will run and save a new csv named as processed_data.csv



##### - this works perfectly - working script

import pandas as pd
#from data_process_commerce import CommerceDataset
from datetime import datetime
import numpy as np


class ComLoader:
    def __init__(self, training=True):
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
        self._save_processed_data()

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
                count_user = 1
        # Add the last user
        if count_user >= 0:
            self.user2id[prev_user] = len(self.user2id)

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
        self.view_time_list.append(user_view_time)
        self.click_rate_list.append(user_click_rate)

    def _save_processed_data(self):
        # Create a dictionary for the processed data
        processed_data = {
            'user_id': [],
            'item_id': [],
            'timestamp': [],
            'view_time': [],
            'click_rate': []
        }

        # Flatten the nested lists and create the processed dataset
        for i in range(len(self.user_list)):
            user_id = [self.user_list[i]] * len(self.item_list[i])
            processed_data['user_id'].extend(user_id)
            processed_data['item_id'].extend(self.item_list[i])
            processed_data['timestamp'].extend(self.time_list[i])
            processed_data['view_time'].extend(self.view_time_list[i])
            processed_data['click_rate'].extend(self.click_rate_list[i])

        # Create DataFrame and save to CSV
        df_processed = pd.DataFrame(processed_data)
        df_processed.to_csv('data/processed_data.csv', index=False)
        print(f"Processed data saved to data/processed_data.csv")

    # since I have read the file separately
    def create_data_set(self, batch_size):
        return CommerceDataset(self.user_list, self.item_list, self.time_list, self.view_time_list, self.click_rate_list, batch_size, self.training)

# Example usage
if __name__ == "__main__":
    loader = ComLoader(training=True)
    print("Data processing completed!")

