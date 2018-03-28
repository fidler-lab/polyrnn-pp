import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import os
import json
import matplotlib.pyplot as plt
from poly_utils import vis_polys
import skimage.io as io
import numpy as np
import tqdm

def main(pred_dir, show_ggnn):
    preds_path = glob.glob(os.path.join(pred_dir, '*.json'))

    fig, axes = plt.subplots(1, 2 if show_ggnn else 1, num=0,figsize=(12,6))
    axes = np.array(axes).flatten()
    for pred_path in tqdm.tqdm(preds_path):
        pred = json.load(open(pred_path, 'r'))
        file_name = pred_path.split('/')[-1].split('.')[0]

        im_crop, polys = io.imread(pred['img_source']), np.array(pred['polys'])
        vis_polys(axes[0], im_crop, polys, title='PolygonRNN++ : %s ' % file_name)
        if show_ggnn:
            vis_polys(axes[1], im_crop, np.array(pred['polys_ggnn']), title=' PolygonRNN++ + GGNN : %s' % file_name)

        fig_name = os.path.join(pred_dir, file_name) + '.png'
        fig.savefig(fig_name)
        
        [ax.cla() for ax in axes]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred_dir', default='output/', help='dir with the predicted json files')
    parser.add_argument('--show_ggnn', action="store_true", default=False, help='visualize ggnn')
    # --
    args = parser.parse_args()
    main(args.pred_dir, args.show_ggnn)
