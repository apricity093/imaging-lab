# fastMRI network components

inv_framework/mri/models/varnet.py and unet.py are adapted from
facebookresearch/fastMRI (archived main, retrieved 2026-11-06).
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Public benchmark data

The public_standard benchmark uses scikit-image 0.26.0 sample images:
camera by Lav Varshney (CC0), astronaut from NASA (public domain, no known
copyright restrictions), and cell by the quantitative phase imaging contributors
listed in skimage.data.cell (CC0). Image conversions and source URLs are retained
in each case.json and docs/public_benchmarks.md. Cell is not fluorescence data.

Levin09 kernels are attributed to Anat Levin, Yair Weiss, Fredo Durand and
William T. Freeman, CVPR 2009, obtained from cszn/DPIR/kernels/Levin09.mat.
The DPIR repository MIT notice, Copyright (c) 2020 Kai Zhang, is preserved in
tests/data/deconvolution/sources/DPIR_LICENSE.txt. Kernels retain original support;
no separate original-author data license is asserted by this notice.

The MR tissue phantom is generated with phantominator 0.7.0 (MIT), Nicholas
McKibben, implementing Gach, Tanase and Boada's 2008 MR Shepp-Logan standard.
This dependency is used for data preparation, not the reconstruction runtime.
No real fastMRI patient data is redistributed. fastMRI source-code licensing
does not replace the dataset use agreement.

## Industrial CT integration

CT geometry adapters and example datasets derive from peterw634/inv_framework,
commit dc181fb65afbf300a9935c3c770231c44ffc0c78.
The CUDA projector derives from nasn-truncated-ct commit
35bb7088273dddb07495cafbf9fcc612c07ae930 under BSD-3-Clause.
Gridrec derives from TomoPy 1.15.x under BSD-3-Clause.
Licenses are retained in LICENSES/NaSn-BSD-3-Clause.txt and
LICENSES/TomoPy-BSD-3-Clause.txt. Dataset licenses are listed in
examples/ct_data/README.md.

## Deep-prior integration

DIP, INR, diffusion solvers, conditioning, CG correction, LoRA and pMF models are migrated from peterw634/inv_framework at dc181fb65afbf300a9935c3c770231c44ffc0c78. Existing source notices and LICENSES apply. DM4CT model revisions and pMF release assets are declared in docs/ct/deep_priors.md and inv_framework/solvers/deep/models/resources/model-v0.1.0.json. Source card measurements retain their original evaluation conditions. LoDoInd data retain CC BY-NC 4.0 terms.

## NaSn application

Data preparation, training, EMA export and benchmark orchestration in inv_framework/applications/nasn_ct derive from the same fixed NaSn integration above. Dataset manifests retain upstream publication and license declarations. No raw NaSn scans or trained checkpoints are redistributed.

## Retained model, scheduler and dataset notices

The pMF model source retains the upstream MIT license, Copyright (c) Yiyang Lu,
in LICENSES/pMF-MIT.txt. The ReSample DDIM scheduler adaptation derives from
Hugging Face Diffusers 0.32.2 under Apache-2.0, retained in
LICENSES/diffusers-Apache-2.0.txt. The NaSn dataset attribution declaration is
retained in LICENSES/CC-BY-4.0-DATA.txt. The local data manifest currently has
published=false. Dataset access and publication status follow that manifest.

The diffusion inference reference is DM4CT/DM4CT commit
49b3e5907178b56338d55c17944de17168a7a0a1 (pipelines.py and condition_methods.py).
The fixed source snapshot reports no standalone license for that reference
material. This notice does not assign a new license to that upstream material.
The MIT declarations for downloaded model weights apply to those model assets.
The migrated source branch also supplies no repository-wide LICENSE file.
Its provenance and component notices are retained without assigning it a new
upstream license.

Source algorithm measurements are retained in
docs/algorithms/source_cards/results.json and method cards. Those records keep
the upstream evaluation protocol and environment. Current local validation is
recorded separately in docs/ct.
