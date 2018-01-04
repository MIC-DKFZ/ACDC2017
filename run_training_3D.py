# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib
matplotlib.use('agg')
import imp
import sys
import os
sys.path.append("configs")

def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    net = cf.get_network(mode='train')

    out_folder = os.path.join(cf.results_dir, "fold%d" % fold)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    gen_tr, gen_val = cf.get_train_val_generators(fold)
    net.plot_architecture(os.path.join(out_folder, "architecture.png"))

    net.train(gen_tr, gen_val, cf.n_epochs, cf.n_batches_per_epoch, cf.base_lr, cf.lr_decay, seed=False, loss=cf.loss,
              solver=cf.solver, l2_penalty=cf.l2_penalty, best_params_file=os.path.join(out_folder, "best_params.pkl"),
              latest_params_file=os.path.join(out_folder, "latest_params.pkl"), patience=cf.patience,
              plot_fname=os.path.join(out_folder, "progress.png"))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)