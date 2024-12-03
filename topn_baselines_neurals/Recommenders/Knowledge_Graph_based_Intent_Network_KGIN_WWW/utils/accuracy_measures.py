import numpy as np
from statistics import mean 

class Recall: 
    def __init__(self, length=5):
        self.length = length
        self.score = 0
        self.numberOfUsers = 0
    def add(self, relevantItems, retrieveList ):
       
        self.score+=len(      relevantItems &     set(retrieveList[:self.length])    ) / len(relevantItems)
        self.numberOfUsers+=1
        
        
    def getScore(self):
        return self.score / self.numberOfUsers
    
    def metricName(self):
        return "Recall"
    

class NDCG: 
    def __init__(self, length=5):
        self.length = length
        self.score = 0
        self.numberOfUsers = 0
    def add(self, pos_items, ranked_list ):

        relevance= [1 for i in range(len(pos_items))]
        

        it2rel = {it: r for it, r in zip(pos_items, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:self.length]], dtype=np.float64)

        # DCG uses the relevance of the recommended items
        rank_dcg = self.dcg(rank_scores)

        ideal_dcg = self.dcg(np.sort(relevance)[::-1][:self.length])
        
        if rank_dcg!=0 and ideal_dcg !=0:
            self.score += rank_dcg / ideal_dcg

        self.numberOfUsers+=1
    
    def dcg(self, scores):
        return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)),
                    dtype=np.float64)
  
    def getScore(self):
        return self.score / self.numberOfUsers
    def metricName(self):
        return "NDCG"
    

class Precision: 
    def __init__(self, length=5):
        self.length = length
        self.score = 0
        self.numberOfUsers = 0
        
        
    def add(self, relevantItems, retrieveList ):
        
        self.score+= len(      relevantItems &     set(retrieveList[:self.length])    ) / len(set(retrieveList[:self.length]))
        self.numberOfUsers+=1
        
        
    def getScore(self):
        return self.score / self.numberOfUsers
    
    def metricName(self):
        return "Precision"

    

    
class Coverage:
    def __init__(self, length = 5, train_data = 0):
        self.length = length
        self.unique_items_appear_in_recommendation_list = set()
        
        # calculate_unique_items_in the traiing data
        self.total_unique_items = set()
        self.calUniqueItemTrainData(train_data)
        
    def calUniqueItemTrainData(self, train_data):
        for _item_set in train_data:
            if len(_item_set) == 0:
                continue
            self.total_unique_items.update( _item_set   )
        
    
    def add(self, retrieveList):
        items = set(  retrieveList[:self.length]    )
        self.unique_items_appear_in_recommendation_list.update( items   )
    
    
    def getScore(self):
        return len(self.unique_items_appear_in_recommendation_list) / len(self.total_unique_items)
    
    def metricName(self):
        return "Coverage"
    

class Novelty:
    def __init__(self, length = 5, train_data = 0):
        self.length = length
        
        self.item_popularity_dict = dict()
        self.cal_item_pop(train_data)
        self.n_interactions = sum( [value for key,value in self.item_popularity_dict.items() ])
        
        self.novelty = 0
        self.numberOfUser = 0
        
    
    def cal_item_pop(self, train_data):
        
        for _item_set in train_data:
            if len(_item_set) == 0:
                continue
            
            for _sItem in _item_set:
                if _sItem in self.item_popularity_dict:
                    self.item_popularity_dict[_sItem]+=1
                else:
                    self.item_popularity_dict[_sItem]=1
        
    
    def add(self, retrieveList):
        self.numberOfUser+=1
        retrieveList = retrieveList[:self.length]
        
        pop_list = []
        for item_ in retrieveList:
            if item_ in self.item_popularity_dict:
                pop_list.append(self.item_popularity_dict[item_])
                
        probability = np.array(pop_list)
        probability = probability/self.n_interactions
        probability = probability[probability!=0]
        self.novelty += np.sum(-np.log2(probability)/self.n_items)
        
        
    
    
    def getScore(self):
        return self.novelty / self.numberOfUser
    
    def metricName(self):
        return "Novelty"
    
    
    

    
    

    
    
    
    
    
    
    
    
    
    