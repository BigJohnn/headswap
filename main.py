from video2img import video2img
from image_seg import seg
from mask import gen_sample as gs

import sys
sys.path.append('F:/head_swap/inpainting')
import inpainting.cv_inpainting as ip
import render.run as run

def main():
    # video2img.run("../madao.mp4")

    # seg.segmentation()    # Get segmentation [Matlab]

    # gs.run()   # Get ROI

    # ip.img_sequence_roi_inpaint("./dataset")

    run.Go()

if __name__ == "__main__":
    main()
