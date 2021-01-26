import os.path as osp
import glob

import multiprocessing as mp
from tqdm import tqdm
import random
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data, Dataset


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (string): Root directory where the dataset should be saved.

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

        n_events (int): Number of events in the raw folder to process



    GRAPH CONSTRUCTION PARAMETERS
    ###########################################################################

        volume_layer_ids (List): List of the volume and layer ids to be included
            in the graph. Layers get indexed by increasing volume and layer id.
            Refer to the following map for the layer indices, and compare them
            to the chart at https://www.kaggle.com/c/trackml-particle-identification/data

                                            41
                        34 --- 39            |        42 --- 47
                                            40

                                            27
                        18 --- 23            |        28 --- 33
                                            24

                                            10
                         0 ---  6            |        11 --- 17
                                             7

        layer_pairs (List): List of which pairs of layers can have edges between them.
            Uses the layer indices described above to reference layers.
            Example for Barrel Only:
            [[7,8],[8,9],[9,10],[10,24],[24,25],[25,26],[26,27],[27,40],[40,41]]

        pt_min (float32): A truth cut applied to reduce the number of nodes in the graph.
            Only nodes associated with particles above this momentum are included.

        eta_range ([min, max]): A cut applied to nodes to select a specific eta

        phi_slope_max (float32): A cut applied to edges to limit the change in phi between
            the two nodes.

        z0_max (float32): A cut applied to edges that limits how far from the center of
            the detector the particle edge can originate from.

        n_phi_sections (int): Break the graph into multiple segments in the phi direction.

        n_eta_sections (int): Break the graph into multiple segments in the eta direction.

        augments (bool): Toggle for turning data augmentation on and off

        intersect (bool): Toggle for interseting lines cut. When connecting Barrel
            edges to the inner most endcap layer, sometimes the edge passes through
            the layer above, this cut removes those edges.

        hough (bool): Toggle for using a hough transform to construct an edge weight.
            Each node in the graph casts votes into an accumulator for a linear
            parameter space. The edge is then used to address this accumulator and
            extract the vote count.

        tracking (bool): Toggle for building truth tracks. Track data is a tensor with
            dimensions (Nx5) with the following columns:
            [r coord, phi coord, z coord, layer index, track number]

        directed (bool): Edges are directed, for an undirected graph, edges are
            duplicated and in reverse direction.

        layer_pairs_plus (bool): Allows for edge connections within the same layer


    MULTIPROCESSING PARAMETERS
    ###########################################################################

        n_workers (int): Number of worker nodes for multiprocessing

        n_tasks (int): Break the processing into a number of tasks

    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None, n_events=0,
                 directed=False, layer_pairs_plus=False,
                 volume_layer_ids=[[8, 2], [8, 4], [8, 6], [8, 8]], #Layers Selected
                 layer_pairs=[[7, 8], [8, 9], [9, 10]],             #Connected Layers
                 pt_min=2.0, eta_range=[-5, 5],                     #Node Cuts
                 phi_slope_max=0.0006, z0_max=150,                  #Edge Cuts
                 n_phi_sections=1, n_eta_sections=1,                #N Sections
                 augments=False, intersect=False,                   #Toggle Switches
                 hough=False, tracking=False,                       #Toggle Switches
                 no_edge_features=True,                             #Toggle Switches
                 n_workers=mp.cpu_count(), n_tasks=1                #multiprocessing
                 ):
        events = glob.glob(osp.join(osp.join(root, 'raw'), 'event*-hits.csv'))
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events = sorted(events)
        if (n_events > 0):
            self.events = self.events[:n_events]


        self.directed         = directed
        self.layer_pairs_plus = layer_pairs_plus
        self.volume_layer_ids = torch.tensor(volume_layer_ids)
        self.layer_pairs      = torch.tensor(layer_pairs)
        self.pt_min           = pt_min
        self.eta_range        = eta_range
        self.phi_slope_max    = phi_slope_max
        self.z0_max           = z0_max
        self.n_phi_sections   = n_phi_sections
        self.n_eta_sections   = n_eta_sections
        self.augments         = augments
        self.intersect        = intersect
        self.hough            = hough
        self.tracking         = tracking
        self.no_edge_features = no_edge_features
        self.n_workers        = n_workers
        self.n_tasks          = n_tasks

        self.accum0_m          = [-30, 30, 2000]          # cot(theta) [eta]
        self.accum0_b          = [-20, 20, 2000]          # z0
        self.accum1_m          = [-.0003, .0003, 2000]    # phi-slope  [qA/pT]
        self.accum1_b          = [-3.3, 3.3, 2000]        # phi0

        # bin = 2000
        # m = torch.cot(2*torch.atan(torch.e^(-eta_range)))
        # self.accum0_m          = [m[0], m[1], bin]                      # cot(theta) [eta]
        # # self.accum0_b          = [-z0_max, z0_max, bin]                 # z0
        # self.accum0_b          = [-20, 20, bin]                 # z0
        # self.accum1_m          = [-phi_slope_max, phi_slope_max, bin]   # phi-slope  [qA/pT]
        # self.accum1_b          = [-np.pi, np.pi, bin]                   # phi0

        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)


    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.csv'))
        return [f.split('/')[-1] for f in self.input_files]


    @property
    def processed_file_names(self):
        N_sections = self.n_phi_sections*self.n_eta_sections
        if not hasattr(self,'processed_files'):
            proc_names = ['event{}_section{}.pt'.format(idx, i) for idx in self.events for i in range(N_sections)]
            if(self.augments):
                proc_names_aug = ['event{}_section{}_aug.pt'.format(idx, i) for idx in self.events for i in range(N_sections)]
                proc_names = [x for y in zip(proc_names, proc_names_aug) for x in y]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files


    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.csv files to {}'.format(self.url, self.raw_dir))


    def len(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def __len__(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def read_hits(self, idx):
        hits_filename = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        hits = pandas.read_csv(
            hits_filename, usecols=['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id'],
            dtype={
                'hit_id': np.int64,
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
                'module_id': np.int64
            })
        return hits


    def read_cells(self, idx):
        cells_filename = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cells = pandas.read_csv(
            cells_filename, usecols=['hit_id', 'ch0', 'ch1', 'value'],
            dtype={
                'hit_id': np.int64,
                'ch0': np.int64,
                'ch1': np.int64,
                'value': np.float32
            })
        return cells


    def read_particles(self, idx):
        particles_filename = osp.join(self.raw_dir, f'event{idx}-particles.csv')
        particles = pandas.read_csv(
            particles_filename, usecols=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits'],
            dtype={
                'particle_id': np.int64,
                'vx': np.float32,
                'vy': np.float32,
                'vz': np.float32,
                'px': np.float32,
                'py': np.float32,
                'pz': np.float32,
                'q': np.int64,
                'nhits': np.int64
            })
        return particles


    def read_truth(self, idx):
        truth_filename = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        truth = pandas.read_csv(
            truth_filename, usecols=['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight'],
            dtype={
                'hit_id': np.int64,
                'particle_id': np.int64,
                'tx': np.float32,
                'ty': np.float32,
                'tz': np.float32,
                'tpx': np.float32,
                'tpy': np.float32,
                'tpz': np.float32,
                'weight': np.float32
            })
        return truth


    def select_hits(self, hits, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        n_det_layers = len(valid_layer)

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        index = layer.unique(return_inverse=True)[1]
        hits = hits[['hit_id', 'x', 'y', 'z']].assign(layer=layer, index=index)

        valid_groups = hits.groupby(['layer'])
        hits = pandas.concat([valid_groups.get_group(valid_layer.numpy()[i]) for i in range(n_det_layers)])

        pt = np.sqrt(particles['px'].values**2 + particles['py'].values**2)
        particles = particles[pt > self.pt_min]

        hits = (hits[['hit_id', 'x', 'y', 'z', 'index']].merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits.merge(particles[['particle_id']], on='particle_id'))

        r = np.sqrt(hits['x'].values**2 + hits['y'].values**2)
        phi = np.arctan2(hits['y'].values, hits['x'].values)
        theta = np.arctan2(r,hits['z'].values)
        eta = -1*np.log(np.tan(theta/2))
        hits = hits[['z', 'index', 'particle_id']].assign(r=r, phi=phi, eta=eta)

        # Remove duplicate hits
        if not self.layer_pairs_plus:
            hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin()]

        r = torch.from_numpy(hits['r'].values)
        phi = torch.from_numpy(hits['phi'].values)
        z = torch.from_numpy(hits['z'].values)
        eta = torch.from_numpy(hits['eta'].values)
        layer = torch.from_numpy(hits['index'].values)
        particle = torch.from_numpy(hits['particle_id'].values)
        pos = torch.stack([r, phi, z], 1)

        return pos, layer, particle, eta


    def compute_edge_index(self, pos, layer):
        # print("Constructing Edge Index")
        edge_indices = torch.empty(2,0, dtype=torch.long)

        layer_pairs = self.layer_pairs
        if self.layer_pairs_plus:
            layers = layer.unique()
            layer_pairs_plus = torch.tensor([[layers[i],layers[i]] for i in range(layers.shape[0])])
            layer_pairs = torch.cat((layer_pairs, layer_pairs_plus), 0)

        for (layer1, layer2) in layer_pairs:
            mask1 = layer == layer1
            mask2 = layer == layer2
            nnz1 = mask1.nonzero().flatten()
            nnz2 = mask2.nonzero().flatten()

            dr   = pos[:, 0][mask2].view(1, -1) - pos[:, 0][mask1].view(-1, 1)
            dphi = pos[:, 1][mask2].view(1, -1) - pos[:, 1][mask1].view(-1, 1)
            dz   = pos[:, 2][mask2].view(1, -1) - pos[:, 2][mask1].view(-1, 1)
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi

            # Calculate phi_slope and z0 which will be cut on
            phi_slope = dphi / dr
            z0 = pos[:, 2][mask1].view(-1, 1) - pos[:, 0][mask1].view(-1, 1) * dz / dr

            # Check for intersecting edges between barrel and endcap connections
            intersected_layer = dr.abs() < -1
            if (self.intersect):
                if((layer1 == 7 and (layer2 == 6 or layer2 == 11)) or
                   (layer2 == 7 and (layer1 == 6 or layer1 == 11))):
                    z_int =  71.56298065185547 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975
                elif((layer1 == 8 and (layer2 == 6 or layer2 == 11)) or
                     (layer2 == 8 and (layer1 == 6 or layer1 == 11))):
                    z_int = 115.37811279296875 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975

            adj = (phi_slope.abs() < self.phi_slope_max) & (z0.abs() < self.z0_max) & (intersected_layer == False)

            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)
            
            edge_indices = torch.cat((edge_indices, edge_index), 1)

        return edge_indices
    
    
    def compute_edge_attr(self, x, edge_index):
        # print("Constructing Edge Attr")
        row, col = edge_index
        in_r, in_phi, in_z    = x[row][:,0], x[row][:,1], x[row][:,2]
        out_r, out_phi, out_z = x[col][:,0], x[col][:,1], x[col][:,2]
        in_r3 = np.sqrt(in_r**2 + in_z**2)
        out_r3 = np.sqrt(out_r**2 + out_z**2)
        in_theta = np.arccos(in_z/in_r3)
        in_eta = -np.log(np.tan(in_theta/2.0))
        out_theta = np.arccos(out_z/out_r3)
        out_eta = -np.log(np.tan(out_theta/2.0))
        deta = out_eta - in_eta
        dphi = out_phi - in_phi
        dphi[dphi > np.pi] -= 2 * np.pi
        dphi[dphi < -np.pi] += 2 * np.pi
        dR = np.sqrt(deta**2 + dphi**2)
        dZ = in_z - out_z
        edge_attr = torch.stack([deta, dphi, dR, dZ], dim=1)
 
        return edge_attr


    def compute_y_index(self, edge_indices, particle):
        # print("Constructing y Index")
        pid1 = [ particle[i].item() for i in edge_indices[0] ]
        pid2 = [ particle[i].item() for i in edge_indices[1] ]
        y = np.zeros(edge_indices.shape[1], dtype=np.int64)
        for i in range(edge_indices.shape[1]):
            if pid1[i] == pid2[i]:
                y[i] = 1

        return torch.from_numpy(y)



    def split_detector_sections(self, pos, layer, particle, eta, phi_edges, eta_edges):
        pos_sect, layer_sect, particle_sect = [], [], []

        for i in range(len(phi_edges) - 1):
            phi_mask1 = pos[:,1] > phi_edges[i]
            phi_mask2 = pos[:,1] < phi_edges[i+1]
            phi_mask  = phi_mask1 & phi_mask2
            phi_pos      = pos[phi_mask]
            phi_layer    = layer[phi_mask]
            phi_particle = particle[phi_mask]
            phi_eta      = eta[phi_mask]

            for j in range(len(eta_edges) - 1):
                eta_mask1 = phi_eta > eta_edges[j]
                eta_mask2 = phi_eta < eta_edges[j+1]
                eta_mask  = eta_mask1 & eta_mask2
                phi_eta_pos = phi_pos[eta_mask]
                phi_eta_layer = phi_layer[eta_mask]
                phi_eta_particle = phi_particle[eta_mask]
                pos_sect.append(phi_eta_pos)
                layer_sect.append(phi_eta_layer)
                particle_sect.append(phi_eta_particle)

        return pos_sect, layer_sect, particle_sect


    def read_event(self, idx):
        hits      = self.read_hits(idx)
        # cells     = self.read_cells(idx)
        particles = self.read_particles(idx)
        truth     = self.read_truth(idx)

        return hits, particles, truth


    def process(self, reprocess=False):
        print('Constructing Graphs using n_workers = ' + str(self.n_workers))
        task_paths = np.array_split(self.processed_paths, self.n_tasks)
        for i in range(self.n_tasks):
            if reprocess or not self.files_exist(task_paths[i]):
                self.process_task(i)


    def process_task(self, idx):
        print('Running task ' + str(idx))
        task_events = np.array_split(self.events, self.n_tasks)
        with mp.Pool(processes = self.n_workers) as pool:
            pool.map(self.process_event, tqdm(task_events[idx]))


    def process_event(self, idx):
        hits, particles, truth = self.read_event(idx)
        pos, layer, particle, eta = self.select_hits(hits, particles, truth)

        tracks = torch.empty(0, dtype=torch.long)
        if(self.tracking):
            tracks = self.build_tracks(hits, particles, truth)

        phi_edges = np.linspace(*(-np.pi, np.pi), num=self.n_phi_sections+1)
        eta_edges = np.linspace(*self.eta_range, num=self.n_eta_sections+1)
        pos_sect, layer_sect, particle_sect = self.split_detector_sections(pos, layer, particle, eta, phi_edges, eta_edges)

        for i in range(len(pos_sect)):
            edge_index = self.compute_edge_index(pos_sect[i], layer_sect[i])
            y = self.compute_y_index(edge_index, particle_sect[i])

            if self.no_edge_features: 
                edge_attr = torch.zeros(edge_index.shape[1], 1, dtype=torch.float32)
            elif(self.hough):
                accumulator0, accumulator1 = self.build_accumulator(pos_sect[i])
                edge_attr  = self.extract_votes(accumulator0, accumulator1, pos_sect[i], edge_index)
            else:
                edge_attr = self.compute_edge_attr(pos_sect[i], edge_index)
                

            data = Data(x=pos_sect[i], edge_index=edge_index, edge_attr=edge_attr, y=y, tracks=tracks)

            if not self.directed and not data.is_undirected():
                rows,cols = data.edge_index
                temp = torch.stack((cols,rows))
                data.edge_index = torch.cat([data.edge_index,temp],dim=-1)
                data.y = torch.cat([data.y,data.y])
                data.edge_attr = torch.cat([data.edge_attr,data.edge_attr])

            torch.save(data, osp.join(self.processed_dir, 'event{}_section{}.pt'.format(idx, i)))

            if (self.augments):
                data.x[:,1]= -data.x[:,1]
                torch.save(data, osp.join(self.processed_dir, 'event{}_section{}_aug.pt'.format(idx, i)))

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue
        #
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)


    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data


    def draw(self, idx, dpi=500):
        # print("Making plots for " + str(self.processed_files[idx]))
        width1 = .1   #blue edge (false)
        width2 = .2   #black edge (true)
        points = .25  #hit points
        dpi   = 500

        X = self[idx].x.cpu().numpy()
        index = self[idx].edge_index.cpu().numpy()
        y = self[idx].y.cpu().numpy()
        true_index = index[:,y > 0]

        r_co = X[:,0]
        phi_co = X[:,1]
        z_co = X[:,2]
        x_co = X[:,0]*np.cos(X[:,1])
        y_co = X[:,0]*np.sin(X[:,1])

        # scale = 12*z_co.max()/r_co.max()
        fig0, (ax0) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig1, (ax1) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig2, (ax2) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig3, (ax3) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))

        # Adjust axes
        ax0.set_xlabel('Z [mm]')
        ax0.set_ylabel('R [mm]')
        ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
        ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_xlim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
        ax2.set_xlabel('R [mm]')
        ax2.set_ylabel('Z [mm]')
        ax2.set_xlim(0, 1.1*r_co.max())
        ax2.set_ylim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
        ax3.set_xlabel('R [mm]')
        ax3.set_ylabel('Phi [mm]')
        ax3.set_xlim(0, 1.1*r_co.max())
        ax3.set_ylim(-np.pi, np.pi)


        #plot points
        r_co[X[:,1] < 0] *= -1
        ax0.scatter(z_co, r_co, s=points, c='k')
        ax0.plot([z_co[index[0]], z_co[index[1]]],
                 [r_co[index[0]], r_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax0.plot([z_co[true_index[0]], z_co[true_index[1]]],
                 [r_co[true_index[0]], r_co[true_index[1]]],
                 '-', c='black', linewidth=width2)
        r_co[X[:,1] < 0] *= -1

        ax1.scatter(x_co, y_co, s=points, c='k')
        ax1.plot([x_co[index[0]], x_co[index[1]]],
                 [y_co[index[0]], y_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax1.plot([x_co[true_index[0]], x_co[true_index[1]]],
                 [y_co[true_index[0]], y_co[true_index[1]]],
                 '-', c='black', linewidth=width2)

        ax2.scatter(r_co, z_co, s=points, c='k')
        ax2.plot([r_co[index[0]], r_co[index[1]]],
                 [z_co[index[0]], z_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax2.plot([r_co[true_index[0]], r_co[true_index[1]]],
                 [z_co[true_index[0]], z_co[true_index[1]]],
                 '-', c='black', linewidth=width2)

        ax3.scatter(r_co, phi_co, s=points, c='k')
        ax3.plot([r_co[index[0]], r_co[index[1]]],
                 [phi_co[index[0]], phi_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax3.plot([r_co[true_index[0]], r_co[true_index[1]]],
                 [phi_co[true_index[0]], phi_co[true_index[1]]],
                 '-', c='black', linewidth=width2)


        fig0_name = self.processed_files[idx].split('.')[0] + '_zr_signed.png'
        fig1_name = self.processed_files[idx].split('.')[0] + '_xy.png'
        fig2_name = self.processed_files[idx].split('.')[0] + '_rz.png'
        fig3_name = self.processed_files[idx].split('.')[0] + '_rphi.png'
        fig0.savefig(fig0_name, dpi=dpi)
        fig1.savefig(fig1_name, dpi=dpi)
        fig2.savefig(fig2_name, dpi=dpi)
        fig3.savefig(fig3_name, dpi=dpi)

        fig0_name = self.processed_files[idx].split('.')[0] + '_zr_signed.pdf'
        fig1_name = self.processed_files[idx].split('.')[0] + '_xy.pdf'
        fig2_name = self.processed_files[idx].split('.')[0] + '_rz.pdf'
        fig3_name = self.processed_files[idx].split('.')[0] + '_rphi.pdf'
        fig0.savefig(fig0_name, dpi=dpi)
        fig1.savefig(fig1_name, dpi=dpi)
        fig2.savefig(fig2_name, dpi=dpi)
        fig3.savefig(fig3_name, dpi=dpi)


    def build_tracks(self, hits, particles, truth):
        # print('Building Tracks')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id']]
                .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'particle_id']]
                .merge(particles[['particle_id', 'px', 'py', 'pz']], on='particle_id'))

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        pt = torch.from_numpy(np.sqrt(hits['px'].values**2 + hits['py'].values**2))
        particle = torch.from_numpy(hits['particle_id'].values)

        layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        pt_mask = pt > self.pt_min
        # mask = layer_mask & pt_mask
        mask = pt_mask

        layer = layer.unique(return_inverse=True)[1]
        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        particle = particle[mask]
        layer = layer[mask]

        particle, indices = torch.sort(particle)
        particle = particle.unique(return_inverse=True)[1]
        pos = pos[indices]
        layer = layer[indices]

        tracks = torch.empty(0,5, dtype=torch.float32)
        for i in range(particle.max()+1):
            track_pos   = pos[particle == i]
            track_layer = layer[particle == i]
            track_particle = particle[particle == i]
            track_layer, indices = torch.sort(track_layer)
            track_pos = track_pos[indices]
            track_layer = track_layer[:, None]
            track_particle = track_particle[:, None]
            track = torch.cat((track_pos, track_layer.type(torch.float32)), 1)
            track = torch.cat((track, track_particle.type(torch.float32)), 1)
            tracks = torch.cat((tracks, track), 0)

        return tracks


    def files_exist(self, files):
        return len(files) != 0 and all([osp.exists(f) for f in files])


    def shuffle(self):
        random.shuffle(self.processed_files)


    def sort(self):
        self.processed_files.sort()


    def build_accumulator(self, pos):
        # print("build_accumulator")
        accumulator0 = torch.zeros(self.accum0_b[2] , self.accum0_m[2], dtype=torch.long)
        accumulator1 = torch.zeros(self.accum1_b[2] , self.accum1_m[2], dtype=torch.long)

        # for i in tqdm(range(pos.shape[0])):
        # # for i in range(pos.shape[0]):
        #     self.cast_vote(accumulator0, pos[i,0], pos[i,2], 0) #R-Z   Plane
        #     self.cast_vote(accumulator1, pos[i,0], pos[i,1], 1) #R-Phi Plane

        # accumulator = torch.stack([self.cast_vote(pos[i,0], pos[i,2]) for i in range(pos.shape[0])], dim=0).sum(dim=0)
        # self.draw_accumulator(accumulator0, accumulator1)
        return accumulator0, accumulator1


    def draw_accumulator(self, accumulator0, accumulator1):
        fig0, ax0 = plt.subplots()
        img0 = ax0.imshow(accumulator0, cmap="hot", extent=[self.accum0_m[0],self.accum0_m[1],self.accum0_b[0],self.accum0_b[1]], aspect="auto")
        ax0.set_xlabel(r"$m$")
        ax0.set_ylabel(r"$b$")
        fig0.colorbar(img0)
        plt.title("Hough Transform Accumulator (RZ)")
        fig0.savefig("accumulator_rz.pdf", dpi=600)

        fig1, ax1 = plt.subplots()
        img1 = ax1.imshow(accumulator1, cmap="hot", extent=[self.accum1_m[0],self.accum1_m[1],self.accum1_b[0],self.accum1_b[1]], aspect="auto")
        ax1.set_xlabel(r"$m$")
        ax1.set_ylabel(r"$b$")
        fig1.colorbar(img1)
        plt.title("Hough Transform Accumulator (RPhi)")
        fig1.savefig("accumulator_rphi.pdf", dpi=600)



    def cast_vote(self, accumulator, x_co, y_co, switch=0):
    # def cast_vote(self, x_co, y_co):
        # accumulator = torch.zeros(self.accum_b[2] , self.accum_m[2], dtype=torch.long)

        # print(switch)
        # print(accumulator)

        if switch == 0:
            b_min = self.accum0_b[0]
            b_max = self.accum0_b[1]
            b_bin = self.accum0_b[2]
            m_min = self.accum0_m[0]
            m_max = self.accum0_m[1]
            m_bin = self.accum0_m[2]
        elif switch == 1:
            b_min = self.accum1_b[0]
            b_max = self.accum1_b[1]
            b_bin = self.accum1_b[2]
            m_min = self.accum1_m[0]
            m_max = self.accum1_m[1]
            m_bin = self.accum1_m[2]

        m_step = (m_max - m_min) / m_bin
        b_step = (b_max - b_min) / b_bin
        m_lo = torch.tensor([m_min +  i   *m_step for i in range(m_bin)])
        m_hi = torch.tensor([m_min + (i+1)*m_step for i in range(m_bin)])
        b_lo = y_co - m_lo * x_co
        b_hi = y_co - m_hi * x_co
        j_lo = torch.floor(b_bin * (b_lo - b_max) / (b_min - b_max))
        j_hi = torch.floor(b_bin * (b_hi - b_max) / (b_min - b_max))
        j_min = torch.min(j_lo, j_hi)
        j_max = torch.max(j_lo, j_hi)

        for i in range(m_bin):
            min = int(j_min[i].item())
            max = int(j_max[i].item())

            if min < 0 and max >= 0 and max < b_bin:
                accumulator[:,i][:max+1] = accumulator[:,i][:max+1] + 1
                # accumulator[:,i][:max+1] = 1
            elif min >= 0 and max < b_bin:
                accumulator[:,i][min:max+1] = accumulator[:,i][min:max+1] + 1
                # accumulator[:,i][min:max+1] = 1
            elif min >= 0 and min < b_bin and max >= b_bin:
                accumulator[:,i][min:] = accumulator[:,i][min:] + 1
                # accumulator[:,i][min:] = 1
            elif min < 0 and max >= b_bin:
                accumulator[:,i] = accumulator[:,i] + 1
                # accumulator[:,i] = 1

        return accumulator
        # for i in range(self.accum_m[2]):
        #     m_lo = self.accum_m[0] +  i   *m_step
        #     m_hi = self.accum_m[0] + (i+1)*m_step
        #     b_lo = y_co - m_lo * x_co
        #     b_hi = y_co - m_hi * x_co
        #
        #     if ((b_lo >= b_min and b_lo < b_max) or (b_hi >= b_min and b_hi < b_max)):
        #         j_lo = torch.floor(b_bin * (b_lo - b_max) / (b_min - b_max))
        #         j_hi = torch.floor(b_bin * (b_hi - b_max) / (b_min - b_max))
        #         if j_lo > j_hi:
        #             a = j_lo
        #             j_lo = j_hi
        #             j_hi = a
        #
        #         for k in range(torch.tensor(j_hi-j_lo+1, dtype=torch.int64)):
        #             j = torch.tensor(j_lo + k, dtype=torch.int64)
        #             if (j >= 0 and j < b_bin):
        #                 accumulator[j,i] = accumulator[j,i] + 1


    def extract_votes(self, accumulator0, accumulator1, pos, edge_index):
        # print("extract_votes")

        r_in = pos[edge_index[0], 0]
        p_in = pos[edge_index[0], 1]
        z_in = pos[edge_index[0], 2]
        r_ot = pos[edge_index[1], 0]
        p_ot = pos[edge_index[1], 1]
        z_ot = pos[edge_index[1], 2]

        dr = r_ot-r_in
        dp = p_ot-p_in
        dz = z_ot-z_in
        dp[dp >  np.pi] -= 2 * np.pi
        dp[dp < -np.pi] += 2 * np.pi

        m0 = dz/dr
        b0 = z_in - m0*r_in
        m1 = dp/dr
        b1 = p_in - m1*r_in

        return torch.stack([m0, b0, m1, b1], 1)


        i0 = torch.floor(self.accum0_m[2] * (m0 - self.accum0_m[0]) / (self.accum0_m[1] - self.accum0_m[0]))
        j0 = torch.floor(self.accum0_b[2] * (b0 - self.accum0_b[1]) / (self.accum0_b[0] - self.accum0_b[1]))
        i1 = torch.floor(self.accum1_m[2] * (m1 - self.accum1_m[0]) / (self.accum1_m[1] - self.accum1_m[0]))
        j1 = torch.floor(self.accum1_b[2] * (b1 - self.accum1_b[1]) / (self.accum1_b[0] - self.accum1_b[1]))

        # plt.hist(j0, 100, [-100, 2100])
        # plt.savefig('debug_hist.pdf', dpi=600)

        edge_votes = torch.empty(2,0, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            i0_int = int(i0[i].item())
            j0_int = int(j0[i].item())
            i1_int = int(i1[i].item())
            j1_int = int(j1[i].item())

            if (i0_int >= 0 and i0_int < self.accum0_m[2] and j0_int >= 0 and j0_int < self.accum0_b[2]):
                vote0 = accumulator0[j0_int, i0_int]
            else:
                vote0 = 0

            if (i1_int >= 0 and i1_int < self.accum1_m[2] and j1_int >= 0 and j1_int < self.accum1_b[2]):
                vote1 = accumulator1[j1_int,i1_int]
            else:
                vote1 = 0

            votes = torch.tensor([[vote0], [vote1]])
            edge_votes = torch.cat((edge_votes, votes), 1)

        return edge_votes.T
        