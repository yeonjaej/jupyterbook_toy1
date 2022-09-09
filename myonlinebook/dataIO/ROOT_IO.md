```python
import ROOT as ROOT
```

    Welcome to JupyROOT 6.22/09



```python
from ROOT import TFile, TChain
```


```python
import os
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME')
DATA_DIR = '/sdf/home/y/yjwa/lartpc_mlreco3d/book/data'

```


```python
import sys
# set software directory
sys.path.insert(0, SOFTWARE_DIR)


```


```python
import numpy as np
import yaml
from mlreco.main_funcs import process_config, prepare

from larcv import larcv
```

    /usr/local/lib/python3.8/dist-packages/MinkowskiEngine/__init__.py:36: UserWarning: The environment variable `OMP_NUM_THREADS` not set. MinkowskiEngine will automatically set `OMP_NUM_THREADS=16`. If you want to set `OMP_NUM_THREADS` manually, please export it on the command line before running a python script. e.g. `export OMP_NUM_THREADS=12; python your_program.py`. It is recommended to set it below 24.
      warnings.warn(



```python
example = TFile(os.path.join(DATA_DIR, 'mpvmpr_062022_test_small.root'))
```

    Warning in <TClass::Init>: no dictionary for class larcv::EventNeutrino is available
    Warning in <TClass::Init>: no dictionary for class larcv::NeutrinoSet is available
    Warning in <TClass::Init>: no dictionary for class larcv::Neutrino is available



```python
example.ls()
```


```python
nu_mpv_tree = example.Get("neutrino_mpv_tree")
nu_mpv_tree.Show(1)

list_vtx_x = []
list_vtx_y = []
list_vtx_z = []

for entry in range(nu_mpv_tree.GetEntries()):
    nu_mpv_tree.GetEntry(entry)
    event = nu_mpv_tree.neutrino_mpv_branch
    vtx_x = nu_mpv_tree.GetLeaf("_part_v._vtx._x")
    vtx_y = nu_mpv_tree.GetLeaf("_part_v._vtx._y")
    vtx_z = nu_mpv_tree.GetLeaf("_part_v._vtx._z")
    list_vtx_x.append(vtx_x.GetValue())
    list_vtx_y.append(vtx_y.GetValue())
    list_vtx_z.append(vtx_z.GetValue())
```

    ======> EVENT:1
     neutrino_mpv_branch = (larcv::EventNeutrino*)0xcfcd6f0
     _producer       = mpv
     _run            = 1
     _subrun         = 1
     _event          = 2
     _part_v         = (vector<larcv::Neutrino>*)0xcfcd740
     _part_v._id     = 0
     _part_v._mcst_index = 65535
     _part_v._mct_index = 0
     _part_v._nu_trackid = 0
     _part_v._lepton_trackid = 4294967295
     _part_v._current_type = 0
     _part_v._interaction_mode = 0
     _part_v._interaction_type = 1001
     _part_v._target = 0
     _part_v._nucleon = 0
     _part_v._quark  = 0
     _part_v._w      = -1
     _part_v._x      = -1
     _part_v._y      = -1
     _part_v._qsqr   = -1
     _part_v._theta  = 0
     _part_v._pdg    = 16
     _part_v._px     = 0.96765
     _part_v._py     = -0.0214221
     _part_v._pz     = 1.09055
     _part_v._vtx._x = -169.037
     _part_v._vtx._y = 5.78837
     _part_v._vtx._z = 406.125
     _part_v._vtx._t = 340.494
     _part_v._dist_travel = -1
     _part_v._energy_init = 4.1198
     _part_v._energy_deposit = 0
     _part_v._process = primary
     _part_v._num_voxels = 0
     _part_v._children_id = (vector<unsigned short>*)0xb11f950
     _part_v._traj_x = (vector<double>*)0xb11f968
     _part_v._traj_y = (vector<double>*)0xb11f980
     _part_v._traj_z = (vector<double>*)0xb11f998
     _part_v._traj_t = (vector<double>*)0xb11f9b0
     _part_v._traj_px = (vector<double>*)0xb11f9c8
     _part_v._traj_py = (vector<double>*)0xb11f9e0
     _part_v._traj_pz = (vector<double>*)0xb11f9f8
     _part_v._traj_e = (vector<double>*)0xb11fa10



```python
import matplotlib.pyplot as plt

```


```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.hist(list_vtx_x)
ax1.set_xlabel("vtx_x")
ax2.hist(list_vtx_y)
ax2.set_xlabel("vtx_y")
ax3.hist(list_vtx_z)
ax3.set_xlabel("vtx_z")
```




    Text(0.5, 0, 'vtx_z')




```{figure} https://github.com/yeonjaej/jupyterbook_toy1/blob/main/myonlinebook/dataIO/output_10_1.png
---
height: 300px
name: vertex-rootio-fig
---
```
    


Using ml-reco configuration now.


```python
cluster_tree = example.Get("cluster3d_pcluster_tree")
```

import mlreco.iotools.parsers as parsers

nus_asis = parsers.parse_neutrino_asis(nu_mpv_tree) #no

nus_asis = parsers.parse_neutrino_asis(nu_mpv_tree, cluster_tree) #no you can't call parsers directly.


```python

```


```python
cfg = """
iotool:
  batch_size: 1
  shuffle: False
  num_workers: 4
  collate_fn: CollateSparse
  dataset:
    name: LArCVDataset
    data_keys:
      - /sdf/home/y/yjwa/lartpc_mlreco3d/book/data/mpvmpr_062022_test_small.root
    limit_num_files: 10
    #event_list: '[6436, 562, 3802, 6175, 15256]'
    schema:
      neutrino_asis:
        parser: parse_particle_asis
        args:
          particle_event: particle_pcluster
          cluster_event: cluster3d_pcluster
""".replace('DATA_DIR', DATA_DIR)

```


```python
cfg=yaml.load(cfg,Loader=yaml.Loader)
# pre-process configuration (checks + certain non-specified default settings)
process_config(cfg)
# prepare function configures necessary "handlers"
hs=prepare(cfg)
```

    
    Config processed at: Linux rome0034 3.10.0-1160.42.2.el7.x86_64 #1 SMP Tue Sep 7 14:49:57 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux
    
    $CUDA_VISIBLE_DEVICES="None"
    
    iotool:
      batch_size: 1
      collate_fn: CollateSparse
      dataset:
        data_keys: [/sdf/home/y/yjwa/lartpc_mlreco3d/book/data/mpvmpr_062022_test_small.root]
        limit_num_files: 10
        name: LArCVDataset
        schema:
          neutrino_asis:
            args: {cluster_event: cluster3d_pcluster, particle_event: particle_pcluster}
            parser: parse_particle_asis
      minibatch_size: 1
      num_workers: 4
      shuffle: false
    
    Loading file: /sdf/home/y/yjwa/lartpc_mlreco3d/book/data/mpvmpr_062022_test_small.root
    Loading tree particle_pcluster
    Loading tree cluster3d_pcluster
    Found 101 events in file(s)



```python
event=next(hs.data_io_iter)
```


```python
hs.__dict__
```




    {'cfg': {'iotool': {'batch_size': 1,
       'shuffle': False,
       'num_workers': 4,
       'collate_fn': 'CollateSparse',
       'dataset': {'name': 'LArCVDataset',
        'data_keys': ['/sdf/home/y/yjwa/lartpc_mlreco3d/book/data/mpvmpr_062022_test_small.root'],
        'limit_num_files': 10,
        'schema': {'neutrino_asis': {'parser': 'parse_particle_asis',
          'args': {'particle_event': 'particle_pcluster',
           'cluster_event': 'cluster3d_pcluster'}}}},
       'minibatch_size': 1}},
     'data_io': <torch.utils.data.dataloader.DataLoader at 0x7f2de546a3d0>,
     'data_io_iter': <generator object cycle at 0x7f2dcd1b0ac0>}




```python
help(event['neutrino_asis'][0][0])
```

    Help on Particle in module multiprocessing.queues object:
    
    class Particle(cppyy.gbl.CPPInstance)
     |  cppyy object proxy (internal)
     |  
     |  Method resolution order:
     |      Particle
     |      cppyy.gbl.CPPInstance
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __assign__(...)
     |      larcv::Particle& larcv::Particle::operator=(const larcv::Particle&)
     |  
     |  __init__(...)
     |      Particle::Particle(larcv::ShapeType_t shape = larcv::kShapeUnknown)
     |      Particle::Particle(const larcv::Particle&)
     |  
     |  __str__(...) from builtins.NoneType
     |      CPyCppyy custom instance method (internal)
     |  
     |  ancestor_creation_process(...)
     |      const string& larcv::Particle::ancestor_creation_process()
     |      void larcv::Particle::ancestor_creation_process(const string& proc)
     |  
     |  ancestor_pdg_code(...)
     |      int larcv::Particle::ancestor_pdg_code()
     |      void larcv::Particle::ancestor_pdg_code(int code)
     |  
     |  ancestor_position(...)
     |      const larcv::Vertex& larcv::Particle::ancestor_position()
     |      void larcv::Particle::ancestor_position(const larcv::Vertex& vtx)
     |      void larcv::Particle::ancestor_position(double x, double y, double z, double t)
     |  
     |  ancestor_t(...)
     |      double larcv::Particle::ancestor_t()
     |  
     |  ancestor_track_id(...)
     |      unsigned int larcv::Particle::ancestor_track_id()
     |      void larcv::Particle::ancestor_track_id(unsigned int id)
     |  
     |  ancestor_x(...)
     |      double larcv::Particle::ancestor_x()
     |  
     |  ancestor_y(...)
     |      double larcv::Particle::ancestor_y()
     |  
     |  ancestor_z(...)
     |      double larcv::Particle::ancestor_z()
     |  
     |  boundingbox_2d(...)
     |      const larcv::BBox2D& larcv::Particle::boundingbox_2d(unsigned short id)
     |      const vector<larcv::BBox2D>& larcv::Particle::boundingbox_2d()
     |      void larcv::Particle::boundingbox_2d(const vector<larcv::BBox2D>& bb_v)
     |      void larcv::Particle::boundingbox_2d(const larcv::BBox2D& bb, unsigned short id)
     |  
     |  boundingbox_3d(...)
     |      const larcv::BBox3D& larcv::Particle::boundingbox_3d()
     |      void larcv::Particle::boundingbox_3d(const larcv::BBox3D& bb)
     |  
     |  children_id(...)
     |      const vector<unsigned short>& larcv::Particle::children_id()
     |      void larcv::Particle::children_id(unsigned short id)
     |      void larcv::Particle::children_id(const vector<unsigned short>& id_v)
     |  
     |  creation_process(...)
     |      const string& larcv::Particle::creation_process()
     |      void larcv::Particle::creation_process(const string& proc)
     |  
     |  distance_travel(...)
     |      double larcv::Particle::distance_travel()
     |      void larcv::Particle::distance_travel(double dist)
     |  
     |  dump(...)
     |      string larcv::Particle::dump()
     |  
     |  end_position(...)
     |      const larcv::Vertex& larcv::Particle::end_position()
     |      void larcv::Particle::end_position(const larcv::Vertex& vtx)
     |      void larcv::Particle::end_position(double x, double y, double z, double t)
     |  
     |  energy_deposit(...)
     |      double larcv::Particle::energy_deposit()
     |      void larcv::Particle::energy_deposit(double e)
     |  
     |  energy_init(...)
     |      double larcv::Particle::energy_init()
     |      void larcv::Particle::energy_init(double e)
     |  
     |  first_step(...)
     |      const larcv::Vertex& larcv::Particle::first_step()
     |      void larcv::Particle::first_step(const larcv::Vertex& vtx)
     |      void larcv::Particle::first_step(double x, double y, double z, double t)
     |  
     |  group_id(...)
     |      unsigned short larcv::Particle::group_id()
     |      void larcv::Particle::group_id(unsigned short id)
     |  
     |  id(...)
     |      unsigned short larcv::Particle::id()
     |      void larcv::Particle::id(unsigned short id)
     |  
     |  interaction_id(...)
     |      unsigned short larcv::Particle::interaction_id()
     |      void larcv::Particle::interaction_id(unsigned short id)
     |  
     |  last_step(...)
     |      const larcv::Vertex& larcv::Particle::last_step()
     |      void larcv::Particle::last_step(const larcv::Vertex& vtx)
     |      void larcv::Particle::last_step(double x, double y, double z, double t)
     |  
     |  mcst_index(...)
     |      unsigned short larcv::Particle::mcst_index()
     |      void larcv::Particle::mcst_index(unsigned short id)
     |  
     |  mct_index(...)
     |      unsigned short larcv::Particle::mct_index()
     |      void larcv::Particle::mct_index(unsigned short id)
     |  
     |  momentum(...)
     |      void larcv::Particle::momentum(double px, double py, double pz)
     |  
     |  nu_current_type(...)
     |      short larcv::Particle::nu_current_type()
     |      void larcv::Particle::nu_current_type(short curr)
     |  
     |  nu_interaction_type(...)
     |      short larcv::Particle::nu_interaction_type()
     |      void larcv::Particle::nu_interaction_type(short itype)
     |  
     |  num_voxels(...)
     |      int larcv::Particle::num_voxels()
     |      void larcv::Particle::num_voxels(int count)
     |  
     |  p(...)
     |      double larcv::Particle::p()
     |  
     |  parent_creation_process(...)
     |      const string& larcv::Particle::parent_creation_process()
     |      void larcv::Particle::parent_creation_process(const string& proc)
     |  
     |  parent_id(...)
     |      unsigned short larcv::Particle::parent_id()
     |      void larcv::Particle::parent_id(unsigned short id)
     |  
     |  parent_pdg_code(...)
     |      int larcv::Particle::parent_pdg_code()
     |      void larcv::Particle::parent_pdg_code(int code)
     |  
     |  parent_position(...)
     |      const larcv::Vertex& larcv::Particle::parent_position()
     |      void larcv::Particle::parent_position(const larcv::Vertex& vtx)
     |      void larcv::Particle::parent_position(double x, double y, double z, double t)
     |  
     |  parent_t(...)
     |      double larcv::Particle::parent_t()
     |  
     |  parent_track_id(...)
     |      unsigned int larcv::Particle::parent_track_id()
     |      void larcv::Particle::parent_track_id(unsigned int id)
     |  
     |  parent_x(...)
     |      double larcv::Particle::parent_x()
     |  
     |  parent_y(...)
     |      double larcv::Particle::parent_y()
     |  
     |  parent_z(...)
     |      double larcv::Particle::parent_z()
     |  
     |  pdg_code(...)
     |      int larcv::Particle::pdg_code()
     |      void larcv::Particle::pdg_code(int code)
     |  
     |  position(...)
     |      const larcv::Vertex& larcv::Particle::position()
     |      void larcv::Particle::position(const larcv::Vertex& vtx)
     |      void larcv::Particle::position(double x, double y, double z, double t)
     |  
     |  px(...)
     |      double larcv::Particle::px()
     |  
     |  py(...)
     |      double larcv::Particle::py()
     |  
     |  pz(...)
     |      double larcv::Particle::pz()
     |  
     |  shape(...)
     |      larcv::ShapeType_t larcv::Particle::shape()
     |      void larcv::Particle::shape(larcv::ShapeType_t shape)
     |  
     |  t(...)
     |      double larcv::Particle::t()
     |  
     |  track_id(...)
     |      unsigned int larcv::Particle::track_id()
     |      void larcv::Particle::track_id(unsigned int id)
     |  
     |  x(...)
     |      double larcv::Particle::x()
     |  
     |  y(...)
     |      double larcv::Particle::y()
     |  
     |  z(...)
     |      double larcv::Particle::z()
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from cppyy.gbl.CPPInstance:
     |  
     |  __add__(self, value, /)
     |      Return self+value.
     |  
     |  __bool__(self, /)
     |      self != 0
     |  
     |  __destruct__(...)
     |  
     |  __dispatch__(...)
     |      dispatch to selected overload
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __hash__(self, /)
     |      Return hash(self).
     |  
     |  __invert__(self, /)
     |      ~self
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __mul__(self, value, /)
     |      Return self*value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __neg__(self, /)
     |      -self
     |  
     |  __pos__(self, /)
     |      +self
     |  
     |  __radd__(self, value, /)
     |      Return value+self.
     |  
     |  __reduce__(...) from builtins.NoneType
     |      CPyCppyy custom instance method (internal)
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __rmul__(self, value, /)
     |      Return value*self.
     |  
     |  __rsub__(self, value, /)
     |      Return value-self.
     |  
     |  __rtruediv__(self, value, /)
     |      Return value/self.
     |  
     |  __smartptr__(...)
     |      get associated smart pointer, if any
     |  
     |  __sub__(self, value, /)
     |      Return self-value.
     |  
     |  __truediv__(self, value, /)
     |      Return self/value.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from cppyy.gbl.CPPInstance:
     |  
     |  __new__(*args, **kwargs) from cppyy.CPPScope
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from cppyy.gbl.CPPInstance:
     |  
     |  __python_owns__
     |      If true, python manages the life time of this object
    



```python

parse_list_vtx_x = []
parse_list_vtx_y = []
parse_list_vtx_z = []

for i in range (101):
    event=next(hs.data_io_iter)
    #print (event['neutrino_asis'][0][0].pdg_code(), event['neutrino_asis'][0][0].parent_id(), event['neutrino_asis'][0][0].parent_x())
    parse_list_vtx_x.append(event['neutrino_asis'][0][0].parent_x())
    parse_list_vtx_y.append(event['neutrino_asis'][0][0].parent_y())
    parse_list_vtx_z.append(event['neutrino_asis'][0][0].parent_z())
```


```python
fig_parse, (ax1_parse, ax2_parse, ax3_parse) = plt.subplots(1, 3)
ax1_parse.hist(parse_list_vtx_x)
ax1_parse.set_xlabel("vtx_x")
ax2_parse.hist(parse_list_vtx_y)
ax2_parse.set_xlabel("vtx_y")
ax3_parse.hist(parse_list_vtx_z)
ax3_parse.set_xlabel("vtx_z")
```




```{figure} https://github.com/yeonjaej/jupyterbook_toy1/blob/main/myonlinebook/dataIO/output_21_1.png
---
height: 300px
name: vertex-rootio-fig2
---
```
    

    Text(0.5, 0, 'vtx_z')





    


Good sanity check!


```python

```
