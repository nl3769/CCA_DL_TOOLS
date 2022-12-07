import argparse
import importlib
import package_utils.fold_handler                   as fh
from package_evaluation.evaluationHandler           import evaluationHandler


# ----------------------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    # --- get project parameters
    my_parser = argparse.ArgumentParser(description='Name of set_parameters_*.py')
    my_parser.add_argument('--Parameters', '-param', required=True, help='List of parameters required to execute the code.')
    arg = vars(my_parser.parse_args())
    param = importlib.import_module('package_parameters.' + arg['Parameters'].split('.')[0])
    p = param.setParameters()

    eval = evaluationHandler(p)

    keys = ['A1bis', 'A2'] + list(eval.annotation_methods.keys())

    for key in keys:
        eval.get_diff(key)
        eval.get_MAE(key)


    eval.write_metrics_to_cvs(p.PRES, p.SET + '_')


# ----------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
