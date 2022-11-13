from model import * 

from gh import * 
import genghao_lightning as gl 


def train_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][train_mask]
    target = label[train_mask]
    
    return dict(pred=pred, target=target)


def val_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    full_pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype]
    pred = full_pred[val_mask]
    target = label[val_mask]
    
    # save_embedding_interval = HYPER_PARAM['save_embedding_interval']
    # if save_embedding_interval > 0 and epoch % save_embedding_interval == 0:
    #     np.save(
    #         arr = full_pred.detach().cpu().numpy(), 
    #         file = f"./saved_embedding/embedding_epoch_{epoch}.npy",
    #     )
    
    return dict(pred=pred, target=target)


def test_func(epoch, model, hg, infer_ntype, feat_dict, label, train_mask, val_mask, test_mask):
    pred = model(hg=hg, feat_dict=feat_dict)[infer_ntype][test_mask]
    target = label[test_mask]
    
    return dict(pred=pred, target=target)
               

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hg_path', type=str, required=True)
    parser.add_argument('--infer_ntype', type=str, required=True)
    parser.add_argument('--use_cpu', action='store_true')
    
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--activation', type=str, required=True)

    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    
    
    args = parser.parse_args()
    print(args)
    
    set_cwd(__file__)
    device = gl.auto_select_gpu(use_gpu=not args.use_cpu)
    
    hg = pickle_load(args.hg_path).to(device)
    infer_ntype = args.infer_ntype 
    feat_dict = dict(hg.ndata['feat']) 
    in_dim_dict = { ntype: feat.shape[-1] for ntype, feat in feat_dict.items() }
    label = hg.nodes[infer_ntype].data['label']
    num_classes = len(label.unique())
    train_mask = hg.nodes[infer_ntype].data['train_mask']
    val_mask = hg.nodes[infer_ntype].data['val_mask']
    test_mask = hg.nodes[infer_ntype].data['test_mask']
    ntypes = set(hg.ntypes)
    etypes = set(hg.canonical_etypes)

    model = RGCN(
        in_dim_dict = in_dim_dict,
        hidden_dim = args.hidden_dim,
        out_dim = num_classes,
        num_layers = args.num_layers,
        ntypes = ntypes, 
        etypes = etypes,
        activation = get_activation_func(args.activation),
    )
    
    trainer = gl.FullBatchTrainer(
        model = model, 
        project_name = 'RGCN',
    )
    
    trainer.train_and_eval(
        dataset = dict(
            hg = hg, 
            infer_ntype = infer_ntype, 
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
        optimizer_param = dict(lr=args.lr, weight_decay=args.weight_decay),
        num_epochs = args.num_epochs,
    )


if __name__ == '__main__':
    main() 
