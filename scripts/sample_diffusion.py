import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
from rdkit import Chem

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from utils import reconstruct, transforms


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_pos0_traj, all_pred_v_traj = [], [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        ligand_pos, ligand_mask, ligand_v = None, None, None
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref' or sample_num_atoms == 'inpainting_ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(device)
                if sample_num_atoms == 'inpainting_ref':
                    assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch, 'ligand_mask')
                    ligand_mask = batch.ligand_mask
                    ligand_pos = batch.ligand_pos
                    ligand_v = batch.ligand_atom_feature_full
            elif sample_num_atoms == 'inpainting_prior':
                assert hasattr(batch, 'ligand_pos') and hasattr(batch, 'ligand_atom_feature_full') and hasattr(batch, 'ligand_mask')
                ligand_num_atoms = []
                ligand_mask, ligand_pos, ligand_v = [], [], []
                while len(ligand_num_atoms) < len(batch):
                    data_id = len(ligand_num_atoms)
                    data = batch[data_id]
                    pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                    atom_nums = atom_num.sample_atom_num(pocket_size).astype(int)
                    if atom_nums < len(data.ligand_pos):
                        continue
                    ligand_num_atoms.append(atom_nums)
                    # update ligand_pos and ligand_v, ligand_mask
                    data.ligand_pos = torch.cat([data.ligand_pos, torch.zeros(atom_nums - len(data.ligand_pos), 3, device=data.ligand_pos.device)])
                    data.ligand_atom_feature_full = torch.cat([data.ligand_atom_feature_full, torch.zeros(atom_nums - len(data.ligand_atom_feature_full), device=data.ligand_atom_feature_full.device)])
                    data.ligand_mask = torch.cat([data.ligand_mask, torch.zeros(atom_nums - len(data.ligand_mask), dtype=torch.bool, device=data.ligand_mask.device)])
                    batch[data_id] = data
                    # print(f"sampled {atom_nums} atoms for ligand {data_id} in inpainting_prior mode")
                    assert len(data.ligand_pos) == atom_nums and len(data.ligand_atom_feature_full) == atom_nums and len(data.ligand_mask) == atom_nums
                    ligand_mask.append(data.ligand_mask)
                    ligand_pos.append(data.ligand_pos)
                    ligand_v.append(data.ligand_atom_feature_full)
                # print(f"ligand_num_atoms: {sum(ligand_num_atoms)}")
                batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(device)
                ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=device)
                ligand_mask = torch.cat(ligand_mask, dim=0).bool()
                ligand_pos = torch.cat(ligand_pos, dim=0)
                ligand_v = torch.cat(ligand_v, dim=0).long()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,

                ligand_pos=ligand_pos,   # for inpainting
                ligand_v=ligand_v,       # for inpainting
                ligand_mask=ligand_mask, # for inpainting
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj, ligand_pos0_traj = r['v0_traj'], r['vt_traj'], r['pos0_traj']
            # unbatch pos
            ligand_cum_atoms = torch.cat([
                torch.tensor([0], dtype=torch.long, device=device), 
                ligand_num_atoms.cumsum(dim=0)
            ]).cpu().numpy()
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            all_step_pos0 = [[] for _ in range(n_data)]
            for p in ligand_pos0_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos0[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos0 = [np.stack(step_pos) for step_pos in
                            all_step_pos0]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos0_traj += [p for p in all_step_pos0]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_pos0_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--change_scaffold',action='store_true')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    if os.path.exists(os.path.join(args.result_path, f'result_{args.data_id}.pt')):
        print(f"Result {args.data_id} already exists!")
        exit(0)
        
    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if 'inpainting' in config.sample.sample_num_atoms:
        transform_list.append(trans.AddScaffoldMask('/sharefs/share/sbdd_data/Mask_cd_test.pkl', change_scaffold=args.change_scaffold))
    transform = Compose(transform_list)

    # Load dataset
    ckpt['config'].data.path = '/sharefs/share/sbdd_data/crossdocked_v1.1_rmsd1.0_pocket10'
    ckpt['config'].data.split = '/sharefs/share/sbdd_data/crossdocked_pocket10_pose_split.pt'
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]

    # pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, pred_pos0_traj, time_list = sample_diffusion_ligand(
    #     model, data, config.sample.num_samples,
    #     batch_size=args.batch_size, device=args.device,
    #     num_steps=config.sample.num_steps,
    #     pos_only=config.sample.pos_only,
    #     center_pos_mode=config.sample.center_pos_mode,
    #     sample_num_atoms=config.sample.sample_num_atoms
    # )
    # result = {
    #     'data': data,
    #     'pred_ligand_pos': pred_pos,
    #     'pred_ligand_v': pred_v,
    #     'pred_ligand_pos_traj': pred_pos_traj,
    #     'pred_ligand_v_traj': pred_v_traj,
    #     'pred_ligand_pos0_traj':pred_pos0_traj,
    #     'pred_ligand_v0_traj': pred_v0_traj,
    #     'time': time_list
    # }

    valid_pred_pos = []
    valid_pred_v = []
    valid_pred_pos_traj = []
    valid_pred_v_traj = []
    valid_time_list = []

    num_tries = 0
    while len(valid_pred_pos_traj) < config.sample.num_samples and num_tries < 10:
        num_tries += 1
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, (config.sample.num_samples - len(valid_pred_pos_traj)),
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )

        # Add a reconstruction step to check if it can be successfully reconstructed
        for sample_idx, (pred_pos_, pred_v_) in enumerate(zip(pred_pos_traj, pred_v_traj)):
            pred_pos_, pred_v_ = pred_pos_[-1], pred_v_[-1]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v_, mode="add_aromatic")

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v_, mode="add_aromatic")
                mol = reconstruct.reconstruct_from_generated(pred_pos_, pred_atom_type, pred_aromatic)
            except Exception as e:
                continue

            # incomplete molecule
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:
                continue

            valid_pred_pos.append(pred_pos[sample_idx])
            valid_pred_v.append(pred_v[sample_idx])
            valid_pred_pos_traj.append(pred_pos_traj[sample_idx])
            valid_pred_v_traj.append(pred_v_traj[sample_idx])
        
        valid_time_list += time_list
        logger.info(f"Sample {len(valid_pred_pos_traj)} done!")

    result = {
        'data': data,
        'pred_ligand_pos': valid_pred_pos,
        'pred_ligand_v': valid_pred_v,
        'pred_ligand_pos_traj': valid_pred_pos_traj,
        'pred_ligand_v_traj': valid_pred_v_traj,
        'time': valid_time_list
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
