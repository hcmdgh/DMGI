from model import * 

from gh import * 
import genghao_lightning as gl 
from sklearn.linear_model import LogisticRegression

DATASET = pickle_load('/Dataset/Pickle/DBLP-from-DMGI.dict.pkl')
METAPATH_G_LIST = [
    dgl.add_self_loop(
        dgl.remove_self_loop(
            dgl.to_bidirected(g) 
        )
    )
    for g in DATASET['metapath_graph_dict'].values()
] 
FEAT = DATASET['feat']
LABEL = DATASET['label']
TRAIN_MASK = DATASET['train_mask']
VAL_MASK = DATASET['val_mask']
TEST_MASK = DATASET['test_mask']

HYPER_PARAM = dict(
    embed_dim = 64, 
    dropout = 0.5,
    loss_alpha = 0.001, 
    
    num_epochs = 2000,     
    lr = 0.0005, 
    weight_decay = 0.0001, 
)

        
def train_epoch(model, dataset):
    metapath_g_list = dataset['metapath_g_list']
    feat = dataset['feat']
    
    loss = model(metapath_g_list=metapath_g_list, feat=feat)
    
    return dict(loss=loss) 


def eval_epoch(model, dataset):
    label = dataset['label'].cpu().numpy() 
    train_mask = dataset['train_mask'].cpu().numpy() 
    val_mask = dataset['val_mask'].cpu().numpy() 
    test_mask = dataset['test_mask'].cpu().numpy() 
    
    embed = model.Z.detach().cpu().numpy() 
    
    train_embed = embed[train_mask]
    val_embed = embed[val_mask]
    test_embed = embed[test_mask]

    train_label = label[train_mask]
    val_label = label[val_mask]
    test_label = label[test_mask]

    clf = LogisticRegression()
    clf.fit(X=train_embed, y=train_label)
    val_acc = clf.score(X=val_embed, y=val_label)
    test_acc = clf.score(X=test_embed, y=test_label)
    
    return dict(val_acc=val_acc, test_acc=test_acc)


def main():
    num_nodes, feat_dim = FEAT.shape
    
    model = DMGI(
        in_dim = feat_dim,
        out_dim = HYPER_PARAM['embed_dim'],
        num_nodes = num_nodes,
        num_metapaths = len(METAPATH_G_LIST),
        dropout = HYPER_PARAM['dropout'],
        loss_alpha = HYPER_PARAM['loss_alpha'],
    )
    
    task = gl.UnsupervisedRepresentationLearningTask(
        model = model, 
    )
    
    task.train_and_eval(
        dataset = dict(
            feat = FEAT, 
            label = LABEL, 
            train_mask = TRAIN_MASK,
            val_mask = VAL_MASK,
            test_mask = TEST_MASK,
            metapath_g_list = METAPATH_G_LIST, 
        ),
        train_epoch = train_epoch,
        eval_epoch = eval_epoch, 
        optimizer_type = 'Adam',
        optimizer_param = dict(lr=HYPER_PARAM['lr'], weight_decay=HYPER_PARAM['weight_decay']),
        max_epochs = HYPER_PARAM['num_epochs'], 
    )


if __name__ == '__main__':
    main() 
