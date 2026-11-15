# NaSn CT application

Data acquisition, PAD-PA/Rivers preprocessing, FullCT Gridrec, scale calibration,
corpus construction, pMF training/resume, EMA export and reconstruction commands
are described in [the NaSn guide](../../../../docs/ct/nasn.md).

Application orchestration lives here. Models, random streams, DDS and CUDA
operators are shared with the CT library. Runtime data and weights are supplied
through local paths. Packaged configurations and release manifests describe
upstream resources and their access conditions.
