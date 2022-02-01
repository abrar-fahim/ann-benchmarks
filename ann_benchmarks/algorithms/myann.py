from ann_benchmarks.algorithms.base import BaseANN

import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt

import time


class MyANN(BaseANN):
    
    # vars to keep from base: res, name

    def __init__(self, n_bins, epochs, num_models, batch_size):
        print('initing')
        self.param_lr = 1e-3
        self.param_milestones = []
        self.custom_dataset = None
        self.do_training = True
        self.n_bins = n_bins
        self.epochs = epochs
        self.num_models = num_models
        self.batch_size = batch_size
        self.best_model = None
        self.best_model_list = None

        self.inference_list = None

        self.res = []

        self.bin_count_param = 1
        print('init done')
    

    def __str__(self):
        return 'MyANN(n_bins=%d, num_models=%d)' % (self.n_bins,
                                                   self.num_models)

    def fit(self, X):
        print('running with ', self.n_bins, 'bins')
        self.run(self.n_bins, self.epochs, self.param_lr, [], custom_dataset=X, do_training=True, data='sift', prepare_knn=True, num_models=self.num_models, batch_size=self.batch_size, continue_train=True)

    def run(self, n_bins, epochs, param_lr, param_milestones, custom_dataset=None, do_training=True, data='sift', prepare_knn=False, num_models=2, batch_size=1024, k_train=10, k_inference=10, bin_count=1, continue_train=True):

        print("prepping X tensor")
        # X = torch.tensor(custom_dataset, dtype=float, device='cuda')

        X = custom_dataset

        n_data = X.shape[0]

        # prepare knn ground truth
        class options(object):
            
            normalize_data=False
            sift=False
        pass

        Y = None
        if data == 'sift':
            # Y = torch.load('D:\\uni\\thesis\\learning-ann\\data\\raw\\' + data +  '-Y.pt')

            Y = torch.load('./data/cache/' + data +  '-Y.pt')
        else:
            Y = self.dist_rank(torch.tensor(X), k_train, opt=options, data=data)

        print("file y shape")
        print(Y.shape)
        
        

        # X = torch.tensor(X, dtype='float', device='cuda')

        pass
        print("prepping Y tensor")

        Y = torch.tensor(Y, dtype=float , device='cuda')

        print("Y shape", Y.shape)


        param_batch_size = n_data

        # trainloader = get_my_dataset(
        #     X, shuffle=False, param_batch_size=param_batch_size)
        X = torch.tensor(X, dtype=float, device='cuda')

        self.dataset = X
        # torch.save(X, data + '-X.pt')
        print("x tensor shape")
        print(X.shape)

        # build model
        m = Model
        n_bins = n_bins
        input_dim = X.size(1)
        # param_feat = input_dim
        param_feat = int(n_bins * 1.2)

        model_list = []
        optimizer_list = []
        input_weights = []  # ones for the main model (model_list[0]), but different booster_weights obtained from training model_list[i-1] for model_list[i]; no need to maintain seperate booster weights for each model since im training models sequentially

        for i in range(num_models):
            model = m(n_input=input_dim, n_hidden=param_feat,
                num_class=n_bins, opt=None).cuda()    #remove cuda() LATER
            model.train()
            model_list.append(model)


            optimizer = torch.optim.Adam(
            model.parameters(), lr=param_lr)
            optimizer_list.append(optimizer)

            # do schedulers_list LATER

            
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=param_milestones, gamma=0.1)
            # booster_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     booster_optimizer, milestones=param_milestones, gamma=0.1)

        pass

        # criterion
        crit = MyLoss()    # see if one crit object is enough or not after debugging LATER

    


        # start training!
        losses = []

        iterations = epochs
        max_points_per_bin = n_data / n_bins   
        

        n = X.shape[0]  # no of points

        target_b = n / n_bins

        
        print("starting training")

        # scheduler = torch.sched
        if do_training:

            

            for (i, model) in enumerate(model_list):

                if continue_train:
                    print('continuing training')
                    model.load_state_dict(torch.load('./data/models/' + data + '-best-model-' + str(i) + '-parameters.pt'))
                optimizer = optimizer_list[i]
                lowest_loss = 99999999999
                input_weights = torch.ones(n, device='cuda', requires_grad=False)

                print('\rtraining model ', i, ' / ', num_models, end='')
                print()

                


                #preparing input weights; batch this to make memory compatible LATER
                print('preparing input weights')
                for j in range(i):  # all models from 0 to (i-1)

                    print('\rmodel ', j, ' / ', i, end='')
                    print()

                    running_current_weights = torch.empty((0), device='cuda')  # empty tensor to store running current_weights
                    for k in range(0, n, batch_size):
                        print('\rbatch ', k / batch_size, ' / ', n / batch_size, end='')

                        
                        # start = k * batch_size
                        # end = (k + 1) * batch_size
                        # X_batch = X[start:end]
                        # Y_batch = Y[start:end]

                        X_batch = X[k: k + batch_size]
                        Y_batch = Y[k: k + batch_size]

                        input_weights_batch = input_weights[k: k + batch_size]
                        # print('X batch shape', X_batch.shape)
                        # processing batch

                        
                        knn_indices = torch.flatten(Y_batch)
                        knn_indices = torch.unique(knn_indices)
                        knn_indices, _ = torch.sort(knn_indices, descending=False)
                        knn_indices = knn_indices.type(dtype=torch.long)
                        # print('knns indices', knn_indices)
                        knns = torch.index_select(X, 0, knn_indices)
                        map_vector = torch.zeros(n, device='cuda', dtype=torch.long)
                        nks_vector = torch.arange(knn_indices.shape[0], device='cuda', dtype=torch.long)
                        map_vector = torch.scatter(map_vector, 0, knn_indices, nks_vector)
                        del nks_vector
                        
                        del knn_indices
                        map_vector = map_vector + X_batch.shape[0]
                        X_knn_batch = torch.cat((X_batch, knns), 0)
                        del X_batch
                        
                        del knns
                        y_pred = model_list[j](X_knn_batch)
                        Y_batch_shape = Y_batch.shape
                        Y_batch = torch.flatten(Y_batch)
                        Y_batch = Y_batch.type(dtype=torch.long)
                        Y_batch = torch.gather(map_vector, 0, Y_batch)  # y_batch[i] = map_vector[y_batch[i]]
                        del map_vector
                        Y_batch = torch.reshape(Y_batch, Y_batch_shape) # test this witchcraft LATER 
                        Y_batch = Y_batch.type(dtype=torch.double)

                        (_, _, _, current_weights) = crit(y_pred, Y_batch, input_weights_batch)
                        del Y_batch
                        del y_pred
                        # here, current_weights is a vector of size batch_size
                        current_weights = current_weights.detach() 
                        running_current_weights = torch.cat((running_current_weights, current_weights), 0)
                        # print('running length', running_current_weights.shape)
                        del current_weights

                    pass
                    # y_pred = model_list[j](X)
                    # (_, _, _, current_weights) = crit(y_pred, Y, input_weights) 
                    # current_weights = current_weights.detach()
                
                    input_weights *= running_current_weights    # keeping running product of all input weights for all models 0 to (i-1) in input_weights
                pass  
                print()

                print('training')

                losses = []
                for ep in range(1, iterations + 1):
                    
                    optimizer.zero_grad()
                    print('\repoch ', ep, ' / ', iterations, end='')
                    print()
                    loss_sum = 0  


                    # batch training

                    start = time.time()
                    for k in range(int(n / batch_size) + 1):

                        

                        # print('\rtraining batch ', k, '/ ', n / batch_size, end='')
                        print('\rtraining batch ', k, '/ ', n / batch_size, end='')



                        # FOR RANDOMLY SAMPLING batch_size DATA POINTS AND OPTIMIZING PER BATCH

                        # optimizer.zero_grad()

                        # randomly sampling batch_size data points to create X_batch corresponding Y_batch

                        random_indices = torch.randint(0, n, (batch_size,), device='cuda')

                        X_batch = torch.index_select(X, 0, random_indices)

                        Y_batch = torch.index_select(Y, 0, random_indices)

                        input_weights_batch = torch.index_select(input_weights, 0, random_indices)

                        

                        # FOR SAMPLING ALL DATA POINTS IN BATCHES, THEN OPTIMIZATION

                        # start = k * batch_size
                        # end = (k + 1) * batch_size
                        # X_batch = X[start:end]
                        # Y_batch = Y[start:end]

                        Y_batch = Y_batch[:, :3]    # taking first 3 nns only


                        knn_indices = torch.flatten(Y_batch)  # knn_indices contains indices of knns of all points in X_batch

                        # knn_indices, _ = torch.sort(knn_indices, descending=False) 
                        knn_indices = torch.unique(knn_indices, sorted=True) # sorting needed to make deterministic the order in which knns are present in (nks, d) shaped knns matrix
                        

                        
                        knn_indices = knn_indices.type(dtype=torch.long)
                        # print('knns indices', knn_indices)

                        # need to remove integers from knn_indices that appear in X_batch indices (random_indices)

                        # all_indices = torch.cat((random_indices, knn_indices), 0)

                        # (u, c) = torch.unique(all_indices, return_counts=True)

                        # repeats = u[(c == 2).nonzero().flatten()]

                        # def th_delete(tensor, indices):
                        #     mask = torch.ones(tensor.numel(), dtype=torch.bool)
                        #     mask[indices] = False
                        #     return tensor[mask]

                        # new_knn_indices = knn_indices
                        # # print('repeats ', repeats)
                        # # print('knn indices size before ', knn_indices.shape[0])
                        # for i in range(repeats.numel()):
                        #     index_to_remove = (knn_indices == repeats[i]).nonzero().flatten()

                        #     new_knn_indices = th_delete(new_knn_indices, index_to_remove)

                        # knn_indices = new_knn_indices

                        # print('knn indices size after ', knn_indices.shape[0])

                        knns = torch.index_select(X, 0, knn_indices)

                        map_vector = torch.zeros(n, device='cuda', dtype=torch.long) # maps from X index to index in knns

                        nks_vector = torch.arange(knn_indices.shape[0], device='cuda', dtype=torch.long) # just a vector containing numbers from 0 to nks - 1

                        map_vector = torch.scatter(map_vector, 0, knn_indices, nks_vector)  # map vector shape should be (n)
                        #position of point i in knns vector is map_vector[i]
                        # need to offset these positions by ns since map_vector is contatenated to X_batch, whose size is ns

                        del knn_indices
                        del nks_vector

                        map_vector = map_vector + X_batch.shape[0]



                        X_knn_batch = torch.cat((X_batch, knns), 0)

                        # print('X_knn_batch size', X_knn_batch.shape)
                        # print('X_batch', X_batch.shape)
                        
                        
                        del knns
                        
                        # print("===" + "MODEL " + str(i) + " EPOCH: " + str(ep) + " BATCH " + str(k) +  "===", end='\r')

                        # print('x knn batch size', X_knn_batch.shape, end='\r')
                        # print('workk', X_knn_batch)
                        y_pred = model(X_knn_batch)
                        # y_dist_pred = model(X_batch)
                        del X_batch

                        del X_knn_batch

                        # replace values in Y_batch with corresponding indexes from map_vector

                        Y_batch_shape = Y_batch.shape
                        Y_batch = torch.flatten(Y_batch)
                        Y_batch = Y_batch.type(dtype=torch.long)

                        Y_batch = torch.gather(map_vector, 0, Y_batch)  # y_batch[i] = map_vector[y_batch[i]]

                        del map_vector


                        Y_batch = torch.reshape(Y_batch, Y_batch_shape) # test this witchcraft LATER 
                        Y_batch = Y_batch.type(dtype=torch.double)  # since trunc in loss fn isnt implemented for int or long dtypes
                        # TRYING TO SEPERATE ACC AND DIST BACKWARDS
                        (loss, _, _, _) = crit(y_pred, Y_batch, input_weights_batch) 
                        # (_, _, dist, _) = crit(y_dist_pred, Y_batch, input_weights_batch, False) 

                        del input_weights_batch

                    
                        del Y_batch
                        del y_pred
                        

                        # acc = acc.detach()
                        # dist = dist.detach()
                        

                        # del acc
                    


                        loss.backward()
                        

                        
                        loss_sum += loss.detach()

                        losses.append(loss.detach())

                        del loss

                        if k % 10 == 0:

                            optimizer.step()
                            optimizer.zero_grad()

                        

                        
                    pass
                    end = time.time()
                    print('\ntime for batch: ', str(end - start))
                    
                    
                
                    if loss_sum < lowest_loss:
                        lowest_loss = loss_sum
                        # torch.save(model.state_dict(), 'D:\\uni\\thesis\\learning-ann\\trained-models\\' + data + '-best-model-' + str(i) + '-parameters.pt')
                        torch.save(model.state_dict(), './data/models/' + data + '-best-model-' + str(i) + '-parameters.pt')
                        # self.best_model = model
                    pass
                    # losses.append(loss_sum.item())
                    del loss_sum


                pass

            pass
            
        pass

        plt.plot(losses)
        plt.show()


        

        with torch.no_grad():
            # load model with lowest loss

            self.best_model_list = []

            for i in range(num_models):
                model = Model(n_input=input_dim, n_hidden=param_feat,
                num_class=n_bins, opt=None).cuda()

                # model.load_state_dict(torch.load('D:\\uni\\thesis\\learning-ann\\trained-models\\' + data + '-best-model-' + str(i) + '-parameters.pt'))
                model.load_state_dict(torch.load('./data/models/' + data + '-best-model-' + str(i) + '-parameters.pt'))



                model.eval()

                self.best_model_list.append(model)

                
            pass
        pass 

            
        self.inference_list = torch.empty((0), device='cuda', dtype=int)


        n = X.shape[0]  # total no of points in dataset


        # MAKING BINS DATA STRUCTURE
        for (i, model) in enumerate(self.best_model_list):
            # inference = model(X)
            # inference_list.append(inference)
            # assigned_bins = torch.argmax(inference, dim=1)  # (100, 1)
            # assigned_bins_list.append(assigned_bins)

            

            # make assigned_bins batchwise
            assigned_bins = torch.empty((0), device='cuda', dtype=int)  # doesn't need to be a tensor or in cuda
            n_bins = 0

            for j in range(0, n, batch_size):
                X_batch = X[j: j + batch_size]
                inference = model(X_batch)
                if j == 0:
                    n_bins = inference.shape[1]
                pass
                assigned_bins_batch = torch.argmax(inference, dim=1) # shape = (batch_size, 1)
                assigned_bins = torch.cat((assigned_bins, assigned_bins_batch), 0) # concat lists
            pass
            print('n', n)
            print('assigned bins shape', assigned_bins.shape)

            if i == 0:
                self.inference_list = assigned_bins
            else:
                self.inference_list = torch.vstack((self.inference_list, assigned_bins))
            pass
        pass

        if len(self.inference_list.shape) < 2:
            # this means that num_models is 1, so in prev if i == 0 if, control did not go to else statement to vstack
            self.inference_list = torch.unsqueeze(self.inference_list, 0)
            pass

        print('inference_list shape', self.inference_list.shape)


        # inference_list is the list for different models

        # BINS DATA STRUCTURE DONE
        pass

    def query(self, v, n):
        # n is k in knn
        # v is the point

        param_n = n

        # print('starting query')
        test_inference_list = []
        q = torch.tensor(v, dtype=float, device='cuda')

        if len(q.shape) == 1:
            # only one data point is given, so reshape into 2D tensor before feeding into model
            q = q.reshape(1, -1)
        
        for (i, model) in enumerate(self.best_model_list):
            inference = model(q)
            test_inference_list.append(inference)
        pass
        n = test_inference_list[0].shape[0]


        all_points_bins = []

        res = []

        # ignoring bins for now
        for bin_count in range(1, self.bin_count_param + 1):

            top_bins_list = []
            val_list = []

            # print('enuming over test inf list')
            for (i, inference) in enumerate(test_inference_list):
                (val, top_bins) = torch.topk(inference, bin_count, dim=1, sorted=True)
                val_list.append(val)
                top_bins_list.append(top_bins)

            pass

            del test_inference_list


     
            print('enuming over points')
            running_set_size = 0
            for point, _ in enumerate(top_bins):
                # print('res ', res)

                # if point % 500 == 0:
                print('\rpoint ' + str(point) + ' / ' + str(n), end='')
                # res.append([75] * 10)
                
                # print('res ', res)
                # continue

                # if point > 5:
                #     res.append([75] * 10)
                #     continue

                
        
                max_val = -1
                max_i = -1
                for (i, val) in enumerate(val_list):
                    if val[point][0] > max_val: # [0] since there will be bin_count values in val[point]
                        max_val = val[point][0]
                        max_i = i
                    pass

                pass
                # print('max i', max_i)
                assigned_bins = top_bins_list[max_i][point]

                all_points_bins.append(assigned_bins.item())

                # print('ab', assigned_bins)

                # print('assigned_bins ', assigned_bins, 'for bun cont', bin_count)
        
                # search for all instances of assigned_bins in this models inference
                # the for loop iterates over all selected bins to search for that are output by the model
                # no of bins to search for == bin_count_param
                candidate_set_points_indexes = sum(self.inference_list[max_i] == b for b in assigned_bins).nonzero(as_tuple=False).flatten() # this has indexes of candidate set points

                # print('candidate indexes ', candidate_set_points_indexes)

                # print('cand set size: ', candidate_set_points_indexes)

                

                candidate_set_points = torch.index_select(self.dataset, 0, candidate_set_points_indexes)

                

                candidate_set_size = candidate_set_points.shape[0]

                running_set_size += candidate_set_size

                # print('canddate set points ', candidate_set_points.shape)


                

                # print('candidate_set_points.shappe', candidate_set_points.shape)

                # print('v shape', v.shape)
                # print('v', v)
                # print('v point', v[point])
                P = torch.tensor(v[point], device='cuda')

                # print('p', P)


                batch_size = 500000
                # print('cand set points shape', candidate_set_points.shape)
                # print('cand set points', candidate_set_points)

                nearest_indices = []
                nearest_distances = []
                for k in range(0, candidate_set_size, batch_size):
                    # print('k: ', k)
                    candidate_set_points_batch = candidate_set_points[k: k + batch_size]
                    # print('cand set points batch shape', candidate_set_points_batch.shape)
                    # print('cand set points batch', candidate_set_points_batch)
                    sq_diff =  torch.square(P - candidate_set_points_batch)
                    dists = torch.sum(sq_diff, dim=1)
                    dists = torch.sqrt(dists)
                    del sq_diff
                    (nearest_distances_batch, nearest_indices_batch) = torch.topk(dists, min(param_n, dists.numel()), largest=False)
                    del dists
                    # print('nearest indices batch before ', nearest_indices_batch)
                    
                    nearest_indices_batch += k
                    # print('nearest indices batch after ', nearest_indices_batch)
                    nearest_indices.append(nearest_indices_batch.tolist())
                    nearest_distances.append(nearest_distances_batch.tolist())

                nearest_distances = torch.tensor(nearest_distances, device='cuda').flatten()
                nearest_indices = torch.tensor(nearest_indices, device='cuda').flatten()
                # print('nearest distances ', nearest_distances)

                (nearest_dist_2, nearest_indices_batchwise) = torch.topk(nearest_distances, min(param_n, nearest_distances.numel()), largest=False)

                nearest_indices = torch.index_select(nearest_indices, 0, nearest_indices_batchwise)

                # print('nearets indices ', nearest_indices)
                # print('nearets distances', nearest_dist_2)
                # print('nearest indices ', nearest_indices)
                nearest_points_X_indexes = torch.index_select(candidate_set_points_indexes, 0, nearest_indices) # convert indices wrt cand set to indices wrt X
                del candidate_set_points_indexes

                del nearest_indices

                # print('the point: ', v[point])

                # print('nearest neighbors of point ', str(point), ': ', nearest_points_X_indexes)

                # print('distances of nns: ', nearest_dist_2)

                
                # PREV 
                # squared_diff = torch.square(P - candidate_set_points)
                
                # dists = torch.sum(squared_diff, dim=1)
                # del squared_diff
            

                # # print('ditss shape', dists.shape)

                # (nearest_dists, nearest_indices) = torch.topk(dists, param_n, largest=False)
                # # nearest_indices indexes wrt cand set
                # del dists

                # del nearest_dists

                # nearest_points_X_indexes = torch.index_select(candidate_set_points_indexes, 0, nearest_indices) # convert indices wrt cand set to indices wrt X
                # del candidate_set_points_indexes

                # del nearest_indices

                # PREV END



                # cand_set_nn = self.dist_rank(candidate_set_points, k=param_n, opt=options)

                # shape of cand_set_nn = (n, k == param_n)




                

                

                # find knn among candidate_set_points
                # self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='l2')
                # self._nbrs.fit(candidate_set_points.cpu())

                # # if len(v.shape) == 1:
                # #     v = [v]
                # # self.res = list(self._nbrs.kneighbors(v, return_distance=False, n_neighbors=n)[0])
                # nearest_points_indexes = list(self._nbrs.kneighbors(self.dataset[point].reshape(1, -1).cpu(), return_distance=False, n_neighbors=param_n)[0]) # this indexes wrt the candidate set

                # nearest_points_X_indexes = torch.index_select(candidate_set_points_indexes, 0, torch.tensor(nearest_points_indexes, device='cuda')) # convert indices wrt cand set to indices wrt X

                res.append(nearest_points_X_indexes.tolist())
            pass

        print('avg cand set size: ', running_set_size / n)

        pass
        self.res = res


        return res
    pass

    def batch_query(self, X, n):
        # print('self dataset, ', self.dataset)
        return self.query(X, n)


    pass

    '''
    Memory-compatible. 
    Ranks of closest points not self.
    Uses l2 dist. But uses cosine dist if data normalized. 
    Input: 
    -data: tensors
    -data_y: data to search in
    -specify k if only interested in the top k results.
    -largest: whether pick largest when ranking. 
    -include_self: include the point itself in the final ranking.
    '''
    def dist_rank(self, data_x, k, data_y=None, largest=False, opt=None, include_self=False, data='mnist'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # device='cpu'

        if isinstance(data_x, np.ndarray):
            data_x = torch.from_numpy(data_x)

        if data_y is None:
            data_y = data_x
        else:
            if isinstance(data_y, np.ndarray):
                data_y = torch.from_numpy(data_y)
        k0 = k
        device_o = data_x.device
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        
        (data_x_len, dim) = data_x.size()
        data_y_len = data_y.size(0)
        #break into chunks. 5e6  is total for MNIST point size
        #chunk_sz = int(5e6 // data_y_len)
        chunk_sz = 16384
        chunk_sz = 300 #700 mem error. 1 mil points
        if data_y_len > 990000:
            chunk_sz = 150 #50 if over 1.1 mil; 150 takes up 3.2/4gb mem
            #chunk_sz = 500 #1000 if over 1.1 mil 
        else:
            chunk_sz = 3000    


        print('chunk size set to ', chunk_sz)
        if k+1 > len(data_y):
            k = len(data_y) - 1
        #if opt is not None and opt.sift:
        if device == 'cuda':
            dist_mx = torch.cuda.LongTensor(data_x_len, k+1)
        else:
            dist_mx = torch.LongTensor(data_x_len, k+1)
        data_normalized = True if opt is not None and opt.normalize_data else False
        largest = True if largest else (True if data_normalized else False)
        
        #compute l2 dist <--be memory efficient by blocking
        total_chunks = int((data_x_len-1) // chunk_sz) + 1
        #print('total chunks ', total_chunks)
        y_t = data_y.t()
        if not data_normalized:
            y_norm = (data_y**2).sum(-1).view(1, -1)
        del data_y

        print('total_chunks')
        print(total_chunks)
        
        for i in range(total_chunks):
            if i % 500 == 0:
                print(str(i) + '/' + str(total_chunks))
            pass
            
            base = i*chunk_sz
            upto = min((i+1)*chunk_sz, data_x_len)
            cur_len = upto-base
            x = data_x[base : upto]
            
            if not data_normalized:
                x_norm = (x**2).sum(-1).view(-1, 1)        
                #plus op broadcasts
                dist = x_norm + y_norm        
                dist -= 2*torch.mm(x, y_t)
                del x_norm
            else:
                dist = -torch.mm(x, y_t)
                
            topk = torch.topk(dist, k=k+1, dim=1, largest=largest)[1]
                    
            dist_mx[base:upto, :k+1] = topk #torch.topk(dist, k=k+1, dim=1, largest=largest)[1][:, 1:]
            del dist
            del x
            if i % 500 == 0:
                print('chunk ', i)

        topk = dist_mx
        if k > 3 and opt is not None and opt.sift:
            #topk = dist_mx
            #sift contains duplicate points, don't run this in general.
            identity_ranks = torch.LongTensor(range(len(topk))).to(topk.device)
            topk_0 = topk[:, 0]
            topk_1 = topk[:, 1]
            topk_2 = topk[:, 2]
            topk_3 = topk[:, 3]

            id_idx1 = topk_1 == identity_ranks
            id_idx2 = topk_2 == identity_ranks
            id_idx3 = topk_3 == identity_ranks

            if torch.sum(id_idx1).item() > 0:
                topk[id_idx1, 1] = topk_0[id_idx1]

            if torch.sum(id_idx2).item() > 0:
                topk[id_idx2, 2] = topk_0[id_idx2]

            if torch.sum(id_idx3).item() > 0:
                topk[id_idx3, 3] = topk_0[id_idx3]           

        
        if not include_self:
            topk = topk[:, 1:]
        elif topk.size(-1) > k0:
            topk = topk[:, :-1]
        topk = topk.to(device_o)
        

        return topk
        
pass


'''
Block of net
'''


class MyLoss(nn.Module):

    def __init__(self):

        super(MyLoss, self).__init__()
        print('loss changed')
        self.reduce_var = True
    pass


    '''
    weights has shape (n), multiply loss of point i with weights[i]
    '''
    def forward(self, outputs, y, weights, calculate_add = True):

        nns = torch.trunc(y)
        nns = nns.long()

        k = nns.shape[1]
        n = nns.shape[0]

        n = outputs.shape[0]

        batch_size = y.shape[0] # figuring out batch size from size of y matrix

        

        n_bins = outputs.shape[1]

        diff = 0
        booster_weights = 0
        # 1: accuracy
        if calculate_add:
            reshaped_nns = torch.movedim(nns, 1, 0)
            del nns
            reshaped_nns = torch.unsqueeze(reshaped_nns, 2)
            reshaped_nns = torch.movedim(reshaped_nns, 1, 2)
            reshaped_nns = reshaped_nns.repeat(1, n_bins, 1)

            
            refactored_outputs = torch.unsqueeze(outputs, 0)
            refactored_outputs = torch.movedim(refactored_outputs, 1, 2)
            refactored_outputs = refactored_outputs.repeat(k, 1, 1)

            cost_tensor_new = torch.gather(refactored_outputs, 2, reshaped_nns)
            del reshaped_nns
            del refactored_outputs

            reshaped_outputs = torch.transpose(outputs, 0, 1)
            reshaped_outputs = torch.reshape(reshaped_outputs, (1, n_bins, n))
            
            reshaped_outputs = reshaped_outputs[:, :, :batch_size] 
            add = cost_tensor_new + reshaped_outputs
            


            del reshaped_outputs
            del cost_tensor_new


            add, idx = torch.max(add, 1) 
            del idx

            booster_weights = torch.mean(add, 0)
 

            booster_weights = 2 - booster_weights 
            booster_weights = booster_weights / 2 
            booster_weights = torch.clamp(booster_weights, min=0.5) 
            add = add * (weights)
            
            add = torch.mean(add)


            diff = torch.square(2 - add)


        pass
        
        # 2: bins distribution

        target_b = n / n_bins

        batch_outputs = outputs[:batch_size, :]

        # print('batch outs shape ', batch_outputs.shape)
        b = torch.sum(batch_outputs, 0)
        b_max = torch.max(b) 
        b_min = torch.min(b)
        b = b_max - b_min

        del batch_outputs

       
        cost = b / target_b + diff

        b = b.detach()

        diff = diff.detach()

        booster_weights = booster_weights.detach()

        # print("loss")
        # print(cost)
        return (cost, diff, b / target_b, booster_weights)
    pass

    

class Model(nn.Module):

    def net_block(self, n_in, n_out):

        block = nn.Sequential(nn.Linear(n_in, n_out),
                          nn.BatchNorm1d(n_out),
                          nn.ReLU())
        return block
    def __init__(self, n_input, n_hidden, num_class, opt, toplevel=False):
        super(Model, self).__init__()
        # self.opt = opt
        # self.toplevel = toplevel

        self.block1 = self.net_block(n_input, n_hidden)
        # self.dropout = nn.Dropout(p=0.1)

        # if (opt.glove or opt.sift or opt.prefix10m):
        #     #if include skip connection:
        #     #self.block_mid = net_block(n_hidden + n_input, n_hidden)
        #     self.block_mid = net_block(n_hidden, n_hidden)
        # if toplevel:
        #     self.block2 = net_block(n_hidden, n_hidden)
        self.block2 = self.net_block(n_hidden, n_hidden)

        self.fc1 = nn.Linear(n_hidden, num_class)

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = x.float()
        y = self.block1(x)
        #y = self.dropout(x1)

        # if self.opt.glove or self.opt.sift or self.opt.prefix10m:
        # if include skip connection:
        #y = self.block_mid(torch.cat([x, y], dim=1))
        # y = self.block_mid(y)

        # if self.toplevel:
        #     y = self.block2(y)
        #     y = self.dropout(y)
        y = self.block2(y)   # i added this
        # y = self.block2(y)   # i added this


        out = self.fc1(y)
        out = self.softmax(out)
        return out
