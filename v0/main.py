from model import * 

from gh import * 
import genghao_lightning as gl 

HG = pickle_load('/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl')
HG.nodes['conference'].data['feat'] = torch.eye(HG.num_nodes('conference'))
INFER_NTYPE = 'author'

HYPER_PARAM = dict(
    hidden_dim = 128,
    num_layers = 3, 
    activation = nn.PReLU(), 
    
    num_epochs = 500,
    lr = 0.0001,
    weight_decay = 0.00001, 
)


def train_func(model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][train_mask]
    target = label[train_mask]
    
    return dict(pred=pred, target=target)


def val_func(model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][val_mask]
    target = label[val_mask]
    
    return dict(pred=pred, target=target)


def test_func(model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][test_mask]
    target = label[test_mask]
    
    return dict(pred=pred, target=target)
               

def main():
    device = gl.auto_select_gpu()
    
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
    )


if __name__ == '__main__':
    main() 
