import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HCCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--alpha1', type=float, default=0.45,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--alpha2', type=float, default=0.45,
                        help='choose the degree of the neighbors similiar to themselves')
    parser.add_argument('--data_path', nargs='?', default='training_dataset/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default=r'',
                        help='Project path.')
    parser.add_argument('--save_recom', type=int, default=1,
                        help='Whether save the recommendation results.')
    parser.add_argument('--dataset', nargs='?', default='qiaoji_5_6',
                        help='Choose a dataset from given folder')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--model_type', nargs='?', default='hccf',
                        help='Specify the name of model (hccf).')
    parser.add_argument('--adj_type', nargs='?', default='norm_adj',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='hccf')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. ')
    #probability of an element to be zeroed.
    parser.add_argument('--drop_edge',type=float,default=0.95,
                        help="perserve the percent of edges")
    parser.add_argument('--Ks', nargs='?', default='[5, 10]',
                        help='Output sizes of every layer')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
