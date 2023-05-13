import numpy as np
from sklearn import metrics

from pytorch_utils import forward
import torch


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        print(f'shape:{target.shape}num:{np.sum(target)}')
        index_tar = np.argmax(target, axis=1)
        index_out = np.argmax(clipwise_output, axis=1)
        target_names = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']

        macro_average_precision = metrics.average_precision_score(
            target, clipwise_output, average='macro')
        precision = metrics.average_precision_score(
            target, clipwise_output,average=None)
        acc = 0
        acc_list =[]
        for i in range(10):
            n_correct_pred_per_class = 0
            for n in range(len(index_out)):
                
                if index_out[n] == index_tar[n] == i:
                    n_correct_pred_per_class += 1 
            n_sample_per_class = (index_tar == i)
            n_sample_per_class = n_sample_per_class.sum()
            acc_per_class = n_correct_pred_per_class / n_sample_per_class
            weight = n_sample_per_class/len(index_tar)
            acc_list.append(acc_per_class)
            acc += acc_per_class 
        macro_acc = acc/10.

        
        acc = metrics.accuracy_score(index_tar,index_out)

        log_loss = metrics.log_loss(target, clipwise_output)

        
        statistics = {'acc_list': acc_list, 'precision': precision,'acc': acc, 'macro_acc':macro_acc, 'average_precision': macro_average_precision, 'log_loss':log_loss}

        return statistics