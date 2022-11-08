from model import * 

from gh import * 
import genghao_lightning as gl 

HG = pickle_load('/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl')
INFER_NTYPE = 'author'

# HG = pickle_load('/Dataset/PyG/ogbn-mag/Processed/ogbn-mag-TransE.dglhg.pkl')
# INFER_NTYPE = 'paper'

# HG = pickle_load('/Dataset/OAG-from-HGT/Processed/OAG-CS/OAG-Venue/OAG-Venue.dglhg.pkl')
# INFER_NTYPE = 'paper'

HYPER_PARAM = dict(
    hidden_dim = 256,
    num_layers = 4, 
    activation = nn.PReLU(), 
    
    use_gpu = True, 
    num_epochs = 100,
    lr = 0.001,
    weight_decay = 0.00001, 
    save_model_interval = -1,
    save_embedding_interval = -1,  
)


def train_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][train_mask]
    target = label[train_mask]
    
    return dict(pred=pred, target=target)


def val_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    full_pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype]
    pred = full_pred[val_mask]
    target = label[val_mask]
    
    save_embedding_interval = HYPER_PARAM['save_embedding_interval']
    if save_embedding_interval > 0 and epoch % save_embedding_interval == 0:
        np.save(
            arr = full_pred.detach().cpu().numpy(), 
            file = f"./saved_embedding/embedding_epoch_{epoch}.npy",
        )
    
    return dict(pred=pred, target=target)


def test_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][test_mask]
    target = label[test_mask]
    
    return dict(pred=pred, target=target)
               

def main():
    set_cwd(__file__)
    device = gl.auto_select_gpu(use_gpu=HYPER_PARAM['use_gpu'])
    
    hg = HG.to(device)
    feat_dict = dict(hg.ndata['feat']) 
    in_dim_dict = { ntype: feat.shape[-1] for ntype, feat in feat_dict.items() }
    label = hg.nodes[INFER_NTYPE].data['label']
    num_classes = len(label.unique())
    train_mask = hg.nodes[INFER_NTYPE].data['train_mask']
    val_mask = hg.nodes[INFER_NTYPE].data['val_mask']
    test_mask = hg.nodes[INFER_NTYPE].data['test_mask']
    ntypes = set(hg.ntypes)
    etypes = set(hg.canonical_etypes)

    model = RGCN(
        in_dim_dict = in_dim_dict,
        hidden_dim = HYPER_PARAM['hidden_dim'],
        out_dim = num_classes,
        num_layers = HYPER_PARAM['num_layers'],
        ntypes = ntypes, 
        etypes = etypes,
        activation = HYPER_PARAM['activation'],
    )
    
    trainer = gl.FullBatchTrainer(
        model = model, 
        project_name = 'RGCN',
    )
    
    trainer.train_and_eval(
        dataset = dict(
            hg = hg, 
            infer_ntype = INFER_NTYPE, 
            feat_dict = feat_dict, 
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
        ), 
        train_func = train_func,
        val_func = val_func,
        test_func = test_func,
        evaluator = gl.MultiClassClassificationEvaluator(),
        optimizer_type = 'Adam',
        optimizer_param = dict(lr=HYPER_PARAM['lr'], weight_decay=HYPER_PARAM['weight_decay']),
        num_epochs = HYPER_PARAM['num_epochs'],
        save_model_interval = HYPER_PARAM['save_model_interval'],
    )


if __name__ == '__main__':
    main() 
