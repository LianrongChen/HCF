import torch.optim as optim
import sys
import math
from Models import *
from utility.helper import *
from utility.batch_test import *

def lamb(epoch):
    epoch += 0
    return 0.95 ** (epoch / 14)

result = []
txt = open("./result.txt", "a")
alpha1=args.alpha1
alpha2=args.alpha2

def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)  # 获取对角线上的元素
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union)

class Model_Wrapper(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.record_alphas = False
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.layer_num = args.layer_num
        self.model_type += '_%s_%s_layers%d' % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        print('model_type is {}'.format(self.model_type))

        self.weights_save_path = '%sweights/%s/%s/emb_size%s/layer_num%s/mess_dropout%s/drop_edge%s/lr%s/reg%s' % (
            args.weights_path, args.dataset, self.model_type,
            str(args.embed_size), str(args.layer_num), str(args.mess_dropout), str(args.drop_edge), str(args.lr),
            '-'.join([str(r) for r in eval(args.regs)]))
        self.result_message = []

        print('----self.alg_type is {}----'.format(self.alg_type))

        if self.alg_type in ['hccf']:
            self.model = HCCF(self.n_users, self.n_items, self.emb_dim, self.layer_num, self.mess_dropout)
        else:
            raise Exception('Dont know which model to train')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2 = self.build_hyper_edge(
            args.data_path + args.dataset + '/TE.csv')

        self.model = self.model.cuda()
        self.norm_u1 = self.norm_u1.cuda()
        self.norm_u2 = self.norm_u2.cuda()
        self.norm_i1 = self.norm_i1.cuda()
        self.norm_i2 = self.norm_i2.cuda()
        self.lr_scheduler = self.set_lr_scheduler()

    def get_D_inv(self, Hadj):

        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()
        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.
        De_inv[np.isinf(De_inv)] = 0.

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    def build_hyper_edge(self, file):
        user_inter = np.zeros((USR_NUM, ITEM_NUM))
        items_inter = np.zeros((ITEM_NUM, USR_NUM))
        with open(file) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip("\n").split(" ")
                uid = int(l[0])
                items = [int(j) for j in l[1:]]
                user_inter[uid, items] = 1
                items_inter[items, uid] = 1

        # 用户相似度矩阵
        J_u = jaccard_similarity(user_inter)
        # 每条超边的下标
        indices = np.where(J_u > alpha1)
        # 每条超边节点的权重
        values = J_u[indices]
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (USR_NUM, USR_NUM))
        self.HuEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HuEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HuEdge = self.sparse_mx_to_torch_sparse_tensor(self.HuEdge)
        self.HuEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HuEdge)
        self.norm_u1 = sparse.mm(spm1, De_1)
        self.norm_u2 = self.HuEdge_T

        J_i = jaccard_similarity(items_inter)
        # 每条超边的下标
        indices = np.where(J_i >alpha2)
        # 每条超边节点的权重
        values = J_i[indices]
        # 生成超边矩阵
        HEdge = sp.coo_matrix((values, indices), (ITEM_NUM, ITEM_NUM))
        self.HiEdge = (HEdge).T
        Dv_1, De_1 = self.get_D_inv(self.HiEdge)

        Dv_1 = self.sparse_mx_to_torch_sparse_tensor(Dv_1)
        De_1 = self.sparse_mx_to_torch_sparse_tensor(De_1)
        self.HiEdge = self.sparse_mx_to_torch_sparse_tensor(self.HiEdge)
        self.HiEdge_T = self.sparse_mx_to_torch_sparse_tensor(HEdge)

        spm1 = sparse.mm(Dv_1, self.HiEdge)
        self.norm_i1 = sparse.mm(spm1, De_1)
        self.norm_i2 = self.HiEdge_T

        return self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2

    def set_lr_scheduler(self):  # lr_scheduler：学习率调度器
        fac = lamb
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        # 每次的lr值：来自优化器的初始lr乘上一个lambda
        return scheduler

    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path))

    def test(self, users_to_test, drop_flag=False, batch_test_flag=False):
        self.model.eval()  # 评估模式，batchnorm和Drop层不起作用，相当于self.model.train(False)
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1, self.norm_i2)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger = [], [], [], [], [], [], [], []
        stopping_step = 10
        should_stop = False
        cur_best_pre_0 = 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        for epoch in range(args.epoch):
            t1 = time()

            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.

            for idx in range(n_batch):
                self.model.train()  # 模型为训练模式，像Dropout，Normalize这些层就会起作用，测试模式不会

                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()  # 采样正相关与负相关的物品id
                sample_time += time() - sample_t1

                ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1,
                                                          self.norm_i2)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            self.lr_scheduler.step()  # 学习率更新

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)

            t2 = time()
            # users_to_test:测试数据集用户
            users_to_test = list(data_generator.test_set.keys())
            ret = self.test(users_to_test, drop_flag=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            map_loger.append(ret['map'])
            mrr_loger.append(ret['mrr'])
            fone_loger.append(ret['fone'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1], ret['map'][0], ret['map'][-1], ret['mrr'][0],
                            ret['mrr'][-1], ret['fone'][0], ret['fone'][-1])
                result.append(perf_str + "\n")

                global txt
                txt.write(perf_str + "\n")
                txt.close()
                txt = open("./result.txt", "a")
                print(perf_str)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=15)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop:
                break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                self.save_model()
                if self.record_alphas:
                    self.best_alphas = [i for i in self.model.get_alphas()]
                print('save the weights in path: ', self.weights_save_path)

        # print the final recommendation results to csv files.
        if args.save_recom:
            results_save_path = r'./output/%s/rec_result.csv' % (args.dataset)
            self.save_recResult(results_save_path)

        if rec_loger != []:
            self.print_final_results(rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                                     training_time_list)

    def norm(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def save_recResult(self, outputPath):
        # used for reverve the recommendation lists
        recommendResult = {}
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

        # get all apps (users)
        users_to_test = list(data_generator.test_set.keys())
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1
        count = 0

        # calculate the result by our own
        # get the latent factors
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_u1, self.norm_u2, self.norm_i1,
                                                      self.norm_i2)

        # get result in batch
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start: end]
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]
            # get the ratings
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))
            # move from GPU to CPU
            rate_batch = rate_batch.detach().cpu().numpy()
            # contact each user's ratings with his id
            user_rating_uid = zip(rate_batch, user_batch)
            # now for each user, calculate his ratings and recommendation
            for x in user_rating_uid:
                # user u's ratings for user u
                rating = x[0]
                # uid
                u = x[1]
                training_items = data_generator.train_items[u]
                user_pos_test = data_generator.test_set[u]
                all_items = set(range(ITEM_NUM))
                test_items = list(all_items - set(training_items))
                item_score = {}
                for i in test_items:
                    item_score[i] = rating[i]
                K_max = max(Ks)
                K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
                recommendResult[u] = K_max_item_score

        # output the result to csv file.
        ensureDir(outputPath)
        with open(outputPath, 'w') as f:
            print("----the recommend result has %s items." % (len(recommendResult)))
            for key in recommendResult.keys():  # due to that all users have been used for test and the subscripts start from 0.
                outString = ""
                for v in recommendResult[key]:
                    outString = outString + "," + str(v)
                f.write("%s%s\n" % (key, outString))

    def print_final_results(self, rec_loger, pre_loger, ndcg_loger, hit_loger, map_loger, mrr_loger, fone_loger,
                            training_time_list):
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        map = np.array(map_loger)
        mrr = np.array(mrr_loger)
        fone = np.array(fone_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], map=[%s],mrr=[%s], f1=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcg_loger[idx]]),
                      '\t'.join(['%.5f' % r for r in map[idx]]),
                      '\t'.join(['%.5f' % r for r in mrr[idx]]),
                      '\t'.join(['%.5f' % r for r in fone[idx]]))
        result.append(final_perf + "\n")
        txt.write(final_perf + "\n")
        print(final_perf)

    # pos_items：正相关物品的id
    # neg_items：负相关物品的id
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # torch.mul():对应元素相乘
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)  # torch.mul():对应元素相乘

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items


    t0 = time()

    pretrain_data = None

    Engine = Model_Wrapper(data_config=config, pretrain_data=pretrain_data)
    if args.pretrain:
        print('pretrain path: ', Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            users_to_test = list(data_generator.test_set.keys())
            ret = Engine.test(users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], map=[%.5f, %.5f], mrr=[%.5f, %.5f], f1=[%.5f, %.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1],
                            ret['map'][0], ret['map'][-1],
                            ret['mrr'][0], ret['mrr'][-1],
                            ret['fone'][0], ret['fone'][-1])
            print(pretrain_ret)
        else:
            print('Cannot load pretrained model. Start training from stratch')
    else:
        print('without pretraining')
    Engine.train()
