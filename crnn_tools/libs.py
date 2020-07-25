import numpy as np
from tqdm import tqdm


def parser(single_str):
    single_str = single_str.replace("\n","")
    single_arr = single_str.split(" ")
    name = single_arr[0]
    arr = np.array(single_arr[1:]).astype(int)
    return name,arr
    

# 0 Get total_list for individual character 
with open("data/labels/char_std_5990.txt","r",encoding="utf-8") as f:
    total_list = f.readlines()
total_list = [items.replace("\n","") for items in total_list]
total_list = np.array(total_list)


# 1 Get train_dic and train_dic for train and test data
with open("data/labels/data_train.txt","r",encoding="utf-8") as f:
    train_list = f.readlines()
with open("data/labels/data_test.txt","r",encoding="utf-8") as f:
    test_list = f.readlines()
    
total_data = train_list+test_list
total_dic = {}    
try:
  with tqdm(total_data) as t:
    for i in t:
      name,value = parser(i)
      total_dic.update({name:value})
except KeyboardInterrupt:
      t.close()

    