from model import * 

from gh import * 
from sklearn.manifold import TSNE
import seaborn as sns 

HG = pickle_load('/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl')
INFER_NTYPE = 'author'


def main():
    set_cwd(__file__)
    
    hg = HG 
    feat_dict = dict(hg.ndata['feat']) 
    in_dim_dict = { ntype: feat.shape[-1] for ntype, feat in feat_dict.items() }
    label = hg.nodes[INFER_NTYPE].data['label']
    num_classes = len(label.unique())
    train_mask = hg.nodes[INFER_NTYPE].data['train_mask']
    val_mask = hg.nodes[INFER_NTYPE].data['val_mask']
    test_mask = hg.nodes[INFER_NTYPE].data['test_mask']
    ntypes = set(hg.ntypes)
    etypes = set(hg.canonical_etypes)

    feat = np.load('./saved_embedding/embedding_epoch_115.npy')
    label = label.numpy() 
    
    tsne = TSNE(n_components=2)
    
    feat_tsne = tsne.fit_transform(feat)
    
    print(feat.shape)    
    print(feat_tsne.shape)    
    
    X_tsne_data = np.hstack([feat_tsne, label.reshape(-1, 1)]) 
    df_tsne = pd.DataFrame(X_tsne_data, columns=['x1', 'x2', 'label']) 

    plt.figure(figsize=(8, 8)) 
    sns.scatterplot(data=df_tsne, hue='label', x='x1', y='x2') 
    plt.savefig('./visualize.png')
    

if __name__ == '__main__':
    main() 
