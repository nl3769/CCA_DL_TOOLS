"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import argparse
import importlib
from package_evaluation.evaluationHandler           import evaluationHandler

# ----------------------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()
    sets = p.SET

    for set in sets:
        p.SET = set
        eval=evaluationHandler(p)
        eval.get_hausdorff()
        eval.get_PDM()
        eval.compute_bias_pdm()
        eval.get_diff()
        eval.write_metrics_to_cvs(p.PRESCSV, p.SET + '_')
        eval.mk_plot_seaborn(p.PLOT, p.SET + '_')
        eval.write_unprocessed_images(p.PUNPROCESSED, p.SET)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This function evaluates segmentation results based on CUBS database. It load results availale online (https://data.mendeley.com/datasets/m7ndn58sv6/1) 
    """
    main()

# ----------------------------------------------------------------------------------------------------------------------------------------------------