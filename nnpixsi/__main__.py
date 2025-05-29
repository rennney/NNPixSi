import sys
import json
import click
import nnpixsi


@click.group()
@click.option("-s","--store",type=click.Path(),
              envvar="NNPIXSI_STORE",
              help="File for primary data storage (input/output)")
@click.option("-o","--outstore",type=click.Path(),
              help="File for output (primary only input)")
@click.pass_context
def cli(ctx, store, outstore):
    '''
    NNPixSi command line interface
    '''
    if not store:
        store = "."
    ctx.obj = pixsi.main.Main(store, outstore)


@cli.command()
@click.option("-i","--input",type=click.Path(),required=False,
              help="Path to Data")
@click.pass_context
def train(ctx,input):
    '''
        Train Model on TRED input
    '''
    meas , true_charges = util.extract_measurement_truth_lists(input)
    from torch.utils.data import DataLoader  # ← standard one

    from torch_geometric.data import Batch

    def collate_graphs_with_x_raw(batch):
        batched = Batch.from_data_list(batch)
        batched.x_raw = [x for g in batch for x in g.x_raw]
        return batched
        
    dataset = dataprep.PixelPatchDataset(meas[0:400] , true_charges[0:400])
    train_size = int(0.8 * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:int(0.8 * len(dataset))]
    test_indices = indices[int(0.8 * len(dataset)):]

    train_set = [dataset[i] for i in train_indices]
    test_set = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_graphs_with_x_raw)
    test_loader = DataLoader(test_set, batch_size=32,  collate_fn=collate_graphs_with_x_raw)
    
    model = model_gnn.PixelGNN()
    time_grid = torch.linspace(0, 10.0, steps=8)
    train.train_with_regularization(model, train_loader, test_loader, time_grid,nepochs=400)
    #torch.save(model.state_dict(), "pixel_gnn_latest.pth")



@cli.command()
@click.option("-i","--input",type=click.Path(),required=False,
              help="Path to Data")
@click.option("-m","--modelpath",type=click.Path(),required=False,
              help="Path to Model")
@click.pass_context
def eval(ctx,input,modelpath):
    '''
    Evaluate Model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_gnn.PixelGNN(embed_dim=32, gnn_dim=64).to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()
    
    meas , true_charges = util.extract_measurement_truth_lists(input)
    
    means_meas=[]
    means_pred=[]
    
    for i in tqdm(range(400,830)):
        if len(meas[i])<300 : continue # lets evaluate only longer tracks
        m_m,m_p = evaluate_single_event(model,
                                    meas[i] ,
                                    true_charges[i],
                                    device="cuda",
                                    patch_size=5,
                                    L_samples=10,
                                       ploting = False)
        means_meas.append(m_m)
        means_pred.append(m_p)
    m=np.array(means_meas)
    p=np.array(means_pred)
    plt.figure(figsize=(6, 4))
    plt.hist(means_meas, bins=50, range=(-1,1), alpha=0.6, label=f"Meas, $\mu=${m[(m > -1) & (m < 1)].mean():.3f}, $\sigma=${m[(m > -1) & (m < 1)].std():.3f}")
    plt.hist(means_pred, bins=50, range=(-1,1), alpha=0.6, label=f"Pred, $\mu=${p[(p > -1) & (p < 1)].mean():.3f}, $\sigma=${p[(p > -1) & (p < 1)].std():.3f}")
    plt.xlabel("Mean per track")
    plt.ylabel("Track Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    

def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
